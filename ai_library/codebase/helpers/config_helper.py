("""Utility: validate and load configuration.

This module provides a single convenience function `validate_config`
which accepts a path to a YAML config file, verifies it exists,
parses it, and performs basic sanity checks on required keys and
value types.

The function returns the parsed config dict when validation succeeds
and raises clear exceptions (`FileNotFoundError`, `ValueError`) for
common problems so callers can fail fast.
""")

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import re
import yaml

def validate_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if required keys are missing or types/values are invalid.
    """
    # Default to the package's `config.yaml` located at the package root
    if config_path is None:
        package_root = Path(__file__).resolve().parent.parent.parent
        p = package_root / "config.yaml"
    else:
        p = Path(config_path)

    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Check key presence first
    required_keys = [
        "window",
        "horizon",
        "prediction_target",
        "pipeline_type",
        "splits",
        "cols_to_drop",
        "stride",
        "parquet_path",
        "data_frequency",
        "retrain_frequency",
        "STANDARD_METRICS",
        "data_dir",
        "saved_files_dir",
        "poll_interval_seconds",
        "flush_max_rows",
        "flush_max_seconds",
    ]
    
    # 1. Verify all required keys are present in config dict
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")

    # 2. Verify non-target required keys are not None or empty string
    empty_keys = [
        k for k in required_keys 
        if k != "prediction_target" and config[k] in (None, "")
    ]
    if empty_keys:
        raise ValueError(f"Config keys cannot be empty or None: {', '.join(empty_keys)}")

    # 3. Explicitly validate prediction_target (None, str, or list/tuple of str)
    target = config.get("prediction_target")
    if target is not None:
        if not isinstance(target, (str, list, tuple)):
            raise ValueError("prediction_target must be None, a string, or a list of strings")
        if isinstance(target, (list, tuple)) and not all(isinstance(t, str) for t in target):
            raise ValueError("All items in prediction_target list must be strings")

    pipeline_type = config["pipeline_type"]
    if pipeline_type not in {"gmlp", "xgb", "itransformer"}:
        raise ValueError(f"Unsupported pipeline_type: {pipeline_type}")

    for key in ("window", "horizon", "stride"):
        value = config[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")

    splits = config["splits"]
    if not isinstance(splits, (list, tuple)) or len(splits) != 3:
        raise ValueError(f"splits must be a list or tuple of 3 values, got {splits!r}")
    if any(not isinstance(v, (int, float)) or not 0 < v < 1 for v in splits):
        raise ValueError(f"Each split value must be between 0 and 1, got {splits!r}")
    total = float(sum(splits))
    eps = 1e-8
    if total > 1.0 + eps:
        raise ValueError(f"Split values sum to more than 1.0 (got {total})")

    if not isinstance(config["cols_to_drop"], list):
        raise ValueError("cols_to_drop must be a list")

    if not isinstance(config.get("STANDARD_METRICS"), list):
        raise ValueError("STANDARD_METRICS must be a list of metric names")

    for key in ("data_frequency", "retrain_frequency"):
        value = config.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d+[sm]", value):
            raise ValueError(f"{key} must be a string like '30s' or '5m', got {value!r}")

    # data_dir and saved_files_dir should be non-empty strings
    for dir_key in ("data_dir", "saved_files_dir", "parquet_path"):
        if not isinstance(config.get(dir_key), str) or not config.get(dir_key).strip():
            raise ValueError(f"{dir_key} must be a non-empty string path")

    if not isinstance(config.get("poll_interval_seconds"), (int, float)) or config.get("poll_interval_seconds") <= 0:
        raise ValueError("poll_interval_seconds must be a positive number")

    if not isinstance(config.get("flush_max_rows"), int) or config.get("flush_max_rows") <= 0:
        raise ValueError("flush_max_rows must be a positive integer")

    if not isinstance(config.get("flush_max_seconds"), (int, float)) or config.get("flush_max_seconds") <= 0:
        raise ValueError("flush_max_seconds must be a positive number")

    parquet_train_size = config.get("parquet_train_size")
    if parquet_train_size is not None:
        if not isinstance(parquet_train_size, int) or parquet_train_size <= 0:
            raise ValueError("parquet_train_size must be a positive integer or None")

    # Optional: verify model-specific sections are mappings if present
    for model_key in ("gmlp", "xgb"):
        if model_key in config and config[model_key] is not None and not isinstance(config[model_key], dict):
            raise ValueError(f"{model_key} section must be a mapping of parameters")

    # Validate common hyperparameters for supported model sections
    gmlp_params = config.get("gmlp", {}).get("model_params", {}) if isinstance(config.get("gmlp"), dict) else {}
    xgb_params = config.get("xgb", {}).get("model_params", {}) if isinstance(config.get("xgb"), dict) else {}

    if gmlp_params:
        for key in ("d_model", "d_ffn", "depth", "patch_size", "epochs"):
            if key in gmlp_params and (not isinstance(gmlp_params[key], int) or gmlp_params[key] <= 0):
                raise ValueError(f"gmlp.model_params.{key} must be a positive integer")
        for key in ("lr", "patience"):
            if key in gmlp_params and (not isinstance(gmlp_params[key], (int, float)) or gmlp_params[key] <= 0):
                raise ValueError(f"gmlp.model_params.{key} must be a positive number")

    if xgb_params:
        for key in ("n_estimators", "max_depth", "min_child_weight"):
            if key in xgb_params and (not isinstance(xgb_params[key], int) or xgb_params[key] <= 0):
                raise ValueError(f"xgb.model_params.{key} must be a positive integer")
        for key in ("learning_rate", "subsample", "colsample_bytree", "gamma", "reg_lambda", "reg_alpha"):
            if key in xgb_params and (not isinstance(xgb_params[key], (int, float)) or xgb_params[key] < 0):
                raise ValueError(f"xgb.model_params.{key} must be a positive number")
        if "objective" in xgb_params and not isinstance(xgb_params["objective"], str):
            raise ValueError("xgb.model_params.objective must be a string")
        if "tree_method" in xgb_params and not isinstance(xgb_params["tree_method"], str):
            raise ValueError("xgb.model_params.tree_method must be a string")
        if "predictor" in xgb_params and not isinstance(xgb_params["predictor"], str):
            raise ValueError("xgb.model_params.predictor must be a string")
        if "random_state" in xgb_params and (not isinstance(xgb_params["random_state"], int) or xgb_params["random_state"] < 0):
            raise ValueError("xgb.model_params.random_state must be a non-negative integer")
        if "n_jobs" in xgb_params and (not isinstance(xgb_params["n_jobs"], int)):
            raise ValueError("xgb.model_params.n_jobs must be an integer")

    print(f"Configuration file {p} validated successfully.")
    return config

def _update_nested_key(config_dict: dict, key_path: str, new_value: Any) -> None:
    """Traverse nested dict using dot notation and set value if key path exists."""
    keys = key_path.split(".")
    current = config_dict
    
    # Traverse down to the target dictionary parent
    for i, k in enumerate(keys[:-1]):
        if not isinstance(current, dict) or k not in current:
            path_so_far = ".".join(keys[: i + 1])
            raise KeyError(f"Config key path not found: '{path_so_far}'")
        current = current[k]

    # Validate leaf key existence
    leaf_key = keys[-1]
    if not isinstance(current, dict) or leaf_key not in current:
        raise KeyError(f"Config key not found: '{key_path}'")
        
    current[leaf_key] = new_value


def update_config(config_path: Optional[Path], updates: Any) -> None:
    """Update one or more keys (including nested dot-notation keys) in the YAML config file."""
    # 1. Resolve configuration path
    if config_path is None:
        package_root = Path(__file__).resolve().parent.parent.parent
        p = package_root / "config.yaml"
    else:
        p = Path(config_path)

    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    # 2. Read existing configuration
    with p.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 3. Normalize input updates into a list of (key_path, value) tuples
    pairs = []
    if isinstance(updates, (list, tuple)) and len(updates) == 2 and isinstance(updates[0], str) and not isinstance(updates[1], (list, tuple, dict)):
        pairs.append((updates[0], updates[1]))
    elif isinstance(updates, dict):
        pairs.extend(updates.items())
    elif isinstance(updates, (list, tuple)):
        for item in updates:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each update item must be a [key, value] pair")
            pairs.append((item[0], item[1]))
    else:
        raise TypeError("updates must be a [key, value] pair, a dict, or a list/tuple of pairs")

    # 4. Apply updates
    for key_path, new_value in pairs:
        if not isinstance(key_path, str):
            raise TypeError("Config keys must be strings")
        _update_nested_key(config, key_path, new_value)

    # 5. Write updated configuration back to disk
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

def show_config(config_path: Optional[Path] = None) -> None:
	"""Print the current configuration to stdout."""
	config = validate_config(config_path)
	print(yaml.dump(config, sort_keys=False, default_flow_style=False))
	
