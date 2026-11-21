from ..helpers.config_helper import validate_config
import os
import pandas as pd
import pickle 
from river import drift

def load_parquet(parquet_path):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"No data found in parquet file: {parquet_path}")
    
    if "ts" not in df.columns:
        raise ValueError(f"Parquet file {parquet_path} missing required 'ts' column")
    
    df = df.sort_values(by="ts").reset_index(drop=True)
    return df

def load_state(state_path):
    if not os.path.exists(state_path):
        return {}, None
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    return state.get("detectors", {}), state.get("last_ts")

def save_state_atomic(state_path, detectors, last_ts):
    tmp_path = f"{state_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump({"detectors": detectors, "last_ts": last_ts}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, state_path)

def apply_adwin():
    config = validate_config()
    saved_files_dir = os.path.expanduser(config.get("saved_files_dir"))
    os.makedirs(saved_files_dir, exist_ok=True)
    state_path = os.path.join(saved_files_dir, "adwin_state.pkl")
    parquet_path = os.path.expanduser(config.get("parquet_path"))
    parquet_path_inference = os.path.expanduser(config.get("inference_parquet_path"))
    df_true = load_parquet(parquet_path)
    df_preds = load_parquet(parquet_path_inference)

    #find an adwin window where both dataframes have the same timestamps
    common_timestamps = pd.merge(df_true, df_preds, on="ts", how="inner")["ts"]
    if common_timestamps.empty:
        raise ValueError("No common timestamps found between true and predicted dataframes.")
    common_column_names = [col for col in df_true.columns if col in df_preds.columns and col != "ts"]
    merged = (
        pd.merge(
            df_true[["ts"] + common_column_names],
            df_preds[["ts"] + common_column_names],
            on="ts",
            how="inner",
            suffixes=("_true", "_pred"),
        )
        .sort_values(by="ts")
        .reset_index(drop=True)
    )

    if merged.empty:
        raise ValueError("No common timestamps found between true and predicted dataframes.")

    detectors, last_ts = load_state(state_path)
    if last_ts is not None:
        merged = merged[merged["ts"] > last_ts].reset_index(drop=True)

    for col in common_column_names:
        if col not in detectors:
            detectors[col] = drift.ADWIN(delta=config.get("adwin_delta"))

    drift_detected_any = False

    for row in merged.itertuples(index=False):
        for col in common_column_names:
            t_val = getattr(row, f"{col}_true")
            p_val = getattr(row, f"{col}_pred")

            if pd.isna(t_val) or pd.isna(p_val):
                continue

            error_val = abs(float(t_val) - float(p_val))
            detectors[col].update(error_val)

            if detectors[col].drift_detected:
                drift_detected_any = True

    if drift_detected_any:
        print("drift detected")

    new_last_ts = merged["ts"].iloc[-1] if not merged.empty else last_ts
    save_state_atomic(state_path, detectors, new_last_ts)