"""Metrics recorder script."""
from __future__ import annotations
import yaml
from pathlib import Path
import argparse
import json
import logging
import signal
import threading
import time
import uuid
from typing import Any
import os
import pyarrow as pa
import pyarrow.parquet as pq

from swchmonclient import (
    query_metric_values,
    query_metric_values_raw,
    subscribe_metric,
    subscribe_metric_raw,
    unsubscribe_metric,
)

# --------------------------------------------------------------------------
# Configuration -- edit these for your metric set.
# --------------------------------------------------------------------------

# Standard mode: shared listener on the EMS broker (MON_CLIENT_STOMP_HOST).
# Use this for composite metrics and SLO/constraint topics, which only
# exist on the broker. Samples are stamped at poll time.

package_root = Path(__file__).resolve().parent.parent.parent
with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    DEFAULT_OUT_DIR = os.path.expanduser(config["data_dir"])
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    DEFAULT_POLL_INTERVAL_SECONDS = config["poll_interval_seconds"]
    DEFAULT_FLUSH_MAX_ROWS = config["flush_max_rows"]
    DEFAULT_FLUSH_MAX_SECONDS = config["flush_max_seconds"]

#get these from config.yaml instead of hardcoding them here!!
STANDARD_METRICS = config.get("STANDARD_METRICS", [])

# Raw mode: direct connections to node IPs; samples keep real timestamps.
# Selector per metric: explicit IP list (simplest outside Kubernetes),
# "all", or "local" (both require in-cluster K8s API access).
RAW_METRICS = config.get("RAW_METRICS", {})

logger = logging.getLogger("recorder")


SCHEMA = pa.schema(
    [
        pa.field("node", pa.string()),
        pa.field("mode", pa.string()),
        pa.field("timestamp", pa.float64()),
        pa.field("value", pa.float64()),
        pa.field("value_json", pa.string()),
        # partition columns
        pa.field("metric", pa.string()),
        pa.field("date", pa.string()),
    ]
)


def _utc_date(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(epoch_seconds))


def _normalize_timestamp(raw_timestamp: float) -> float:
    """Millisecond epochs -> seconds (same heuristic swchmonclient uses)."""
    if raw_timestamp > 10_000_000_000:
        return float(raw_timestamp) / 1000.0
    return float(raw_timestamp)


def _coerce_value(payload: Any) -> tuple[float | None, str | None]:
    """Return (numeric value, json fallback) for an arbitrary sample payload."""
    if isinstance(payload, bool):  # bool is an int subclass; keep it as JSON
        return None, json.dumps(payload)
    if isinstance(payload, (int, float)):
        return float(payload), None
    if isinstance(payload, str):
        try:
            return float(payload), None
        except ValueError:
            return None, json.dumps(payload)
    if isinstance(payload, dict):
        for key in ("metricValue", "value"):
            if key in payload:
                value, _ = _coerce_value(payload[key])
                if value is not None:
                    return value, json.dumps(payload, default=str)
    try:
        return None, json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return None, json.dumps(str(payload))


def _make_row(
    metric: str,
    node: str | None,
    mode: str,
    timestamp: float,
    payload: Any,
) -> dict[str, Any]:
    value, value_json = _coerce_value(payload)
    timestamp = _normalize_timestamp(timestamp)
    return {
        "node": node,
        "mode": mode,
        "timestamp": float(timestamp),
        "value": value,
        "value_json": value_json,
        "metric": metric,
        "date": _utc_date(timestamp),
    }

class ParquetSink:
    """Buffers rows in memory and flushes them as hive-partitioned Parquet."""

    def __init__(self, out_dir: str, max_rows: int, max_seconds: float) -> None:
        self._out_dir = out_dir
        self._max_rows = max_rows
        self._max_seconds = max_seconds
        self._rows: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self.total_rows_written = 0

    def append(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            self._rows.extend(rows)

    def maybe_flush(self, force: bool = False) -> None:
        with self._lock:
            due = (
                force
                or len(self._rows) >= self._max_rows
                or (
                    self._rows
                    and time.monotonic() - self._last_flush >= self._max_seconds
                )
            )
            if not due or not self._rows:
                if force:
                    self._last_flush = time.monotonic()
                return
            rows, self._rows = self._rows, []
            self._last_flush = time.monotonic()

        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_to_dataset(
            table,
            root_path=self._out_dir,
            partition_cols=["metric", "date"],
            basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        self.total_rows_written += len(rows)
        logger.info(
            "Flushed %d rows to %s (total written: %d)",
            len(rows),
            self._out_dir,
            self.total_rows_written,
        )


def _poll_standard(metric: str) -> list[dict[str, Any]]:
    poll_time = time.time()
    try:
        values = query_metric_values(metric)
        logger.info("DEBUG: Polled %s directly. Received array: %s", metric, values)
    except Exception:
        logger.exception("Failed to query standard metric %s", metric)
        return []
    return [_make_row(metric, None, "standard", poll_time, v) for v in values]

def _poll_raw(metric: str, window_seconds: int) -> list[dict[str, Any]]:
    try:
        values_by_node = query_metric_values_raw(metric, window_seconds)
    except Exception:
        logger.exception("Failed to query raw metric %s", metric)
        return []
    rows: list[dict[str, Any]] = []
    for node, samples in values_by_node.items():
        for sample in samples:
            timestamp = sample.get("timestamp") or time.time()
            rows.append(_make_row(metric, node, "raw", timestamp, sample.get("value")))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS
    )
    parser.add_argument("--flush-max-rows", type=int, default=DEFAULT_FLUSH_MAX_ROWS)
    parser.add_argument(
        "--flush-max-seconds", type=float, default=DEFAULT_FLUSH_MAX_SECONDS
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds (0 = run until SIGINT/SIGTERM).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overlap = set(STANDARD_METRICS) & set(RAW_METRICS)
    if overlap:
        logger.error(
            "Metrics configured in both standard and raw mode (not allowed): %s",
            sorted(overlap),
        )
        return 2
    if not STANDARD_METRICS and not RAW_METRICS:
        logger.error("No metrics configured.")
        return 2

    stop_event = threading.Event()

    def _request_stop(signum: int, _frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # Every raw sample must be inside the query window at least once between
    # polls, so the window must exceed the poll interval; 3x adds slack for
    # scheduling jitter (re-reads are impossible: returned samples are consumed).
    raw_window_seconds = max(int(args.poll_interval * 3), 10)

    sink = ParquetSink(args.out_dir, args.flush_max_rows, args.flush_max_seconds)
    subscribed_metrics: list[str] = []

    try:
        for metric in STANDARD_METRICS:
            subscribe_metric(metric)
            subscribed_metrics.append(metric)
            logger.info("Subscribed (standard): %s", metric)
        for metric, node_selector in RAW_METRICS.items():
            threads = subscribe_metric_raw(metric, node_selector)
            subscribed_metrics.append(metric)
            logger.info("Subscribed (raw): %s -> %s", metric, threads)

        started_at = time.monotonic()
        logger.info(
            "Recording to %s (poll every %.1fs, flush at %d rows / %.0fs)",
            args.out_dir,
            args.poll_interval,
            args.flush_max_rows,
            args.flush_max_seconds,
        )

        while not stop_event.is_set():
            rows: list[dict[str, Any]] = []
            for metric in STANDARD_METRICS:
                rows.extend(_poll_standard(metric))
            for metric in RAW_METRICS:
                rows.extend(_poll_raw(metric, raw_window_seconds))
            if rows:
                logger.debug("Collected %d rows this poll", len(rows))
            sink.append(rows)
            sink.maybe_flush()

            if args.run_seconds and time.monotonic() - started_at >= args.run_seconds:
                logger.info("Run duration reached, stopping.")
                break
            stop_event.wait(args.poll_interval)
    finally:
        # Final drain: one last poll so buffered samples are not lost.
        final_rows: list[dict[str, Any]] = []
        for metric in STANDARD_METRICS:
            final_rows.extend(_poll_standard(metric))
        for metric in RAW_METRICS:
            final_rows.extend(_poll_raw(metric, raw_window_seconds))
        sink.append(final_rows)
        sink.maybe_flush(force=True)

        for metric in subscribed_metrics:
            try:
                unsubscribe_metric(metric)
                logger.info("Unsubscribed: %s", metric)
            except Exception:
                logger.exception("Failed to unsubscribe %s", metric)

    logger.info("Done. Total rows written: %d", sink.total_rows_written)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

