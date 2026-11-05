#injest metrics that were received from query functions into the parquet_file
import pandas as pd
import yaml
from ..helpers.config_helper import validate_config
import os

def injest_metrics(metric_name, metrics, mode = "raw"):
    if mode not in ["raw", "composite"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'raw' or 'composite'.")
    config = validate_config()
    parquet_path = os.path.expanduser(config["parquet_path"])
    if not os.path.exists(parquet_path):
        df = pd.DataFrame(columns=["ts", "node_ip", "metric_name", "metric_value"])
        df.to_parquet(parquet_path, index=False)
    df = pd.read_parquet(parquet_path)
    datapoints_to_injest = []
    if mode == "raw":
        for node_ip in metrics.keys():
            for metric in metrics[node_ip]:
                new_row = {
                    "ts": metric["timestamp"],
                    "node_ip": node_ip,
                    "metric_name": metric_name,
                    "metric_value": metric["value"],
                }
                datapoints_to_injest.append(new_row)
    elif mode == "composite":
        for row in metrics:
            new_row = {
                "ts": row["timestamp"],
                "node_ip": "composite",
                "metric_name": metric_name,
                "metric_value": row["value"],
            }
            datapoints_to_injest.append(new_row)
    new_df = pd.DataFrame(datapoints_to_injest)
    new_df["ts"] = pd.to_datetime(new_df["ts"], unit="s", utc=True)
    if df.empty:
        df = new_df
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = pd.concat([df, new_df], ignore_index=True)

    df = df.drop_duplicates(subset=["ts", "node_ip", "metric_name"], keep="last")
    df = df.sort_values(by="ts").reset_index(drop=True)
    df.to_parquet(parquet_path, index=False)