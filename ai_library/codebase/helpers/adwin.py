from ..helpers.config_helper import validate_config
import os
import pandas as pd
import pickle 
from river import drift
from ..helpers.pivot_df import pivot_df
from ..helpers.to_saved_files import atomic_save

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

def apply_adwin():
    '''returns True, if drift was detected, else False'''
    config = validate_config()
    saved_files_dir = os.path.expanduser(config.get("saved_files_dir"))
    os.makedirs(saved_files_dir, exist_ok=True)
    state_path = os.path.join(saved_files_dir, "adwin_state.pkl")
    parquet_path = os.path.expanduser(config.get("parquet_path"))
    parquet_path_inference = os.path.expanduser(config.get("inference_parquet_path"))
    try:
        df_true = load_parquet(parquet_path)
        df_true = pivot_df(df_true)
        df_true = df_true.set_index("ts").interpolate("linear").bfill().ffill().reset_index()
        df_preds = load_parquet(parquet_path_inference)
    except:
        print ("Dataframes are empty. Cannot apply adwin")
        return False
    #find an adwin window where both dataframes have the same timestamps
    common_timestamps = pd.merge(df_true, df_preds, on="ts", how="inner")["ts"]
    if common_timestamps.empty:
        print("No common timestamps found between true and predicted dataframes.")
        return False
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
        return "No common timestamps found between true and predicted dataframes."

    detectors, last_ts = load_state(state_path)
    if last_ts is not None:
        merged = merged[merged["ts"] > last_ts].reset_index(drop=True)

    for col in common_column_names:
        if col not in detectors:
            detectors[col] = drift.ADWIN(delta=config.get("adwin_delta"))

    drift_detected_any = False

    true_arrays = {col: merged[f"{col}_true"].to_numpy() for col in common_column_names}
    pred_arrays = {col: merged[f"{col}_pred"].to_numpy() for col in common_column_names}

    for idx in range(len(merged)):
        for col in common_column_names:
            t_val = true_arrays[col][idx]
            p_val = pred_arrays[col][idx]

            if pd.isna(t_val) or pd.isna(p_val):
                continue

            error_val = abs(float(t_val) - float(p_val))
            detectors[col].update(error_val)

            if detectors[col].drift_detected:
                drift_detected_any = True

    new_last_ts = merged["ts"].iloc[-1] if not merged.empty else last_ts
    atomic_save({"detectors": detectors, "last_ts": new_last_ts}, final_path = state_path)

    if drift_detected_any:
        return True
    return False