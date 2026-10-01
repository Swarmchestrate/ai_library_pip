import pickle
import os

import pandas as pd
import torch
import yaml

# Use package-relative imports to be import-safe when the package is installed.
from ..models.iTransformer_class import iTransformer_pipeline
from ..models.gmlp_class import gMLP_pipeline
from ..models.xgb_class import XGBoost_pipeline
from ..helpers.pivot_df import pivot_df
from ..helpers.config_helper import validate_config

def inference():
    from . import read
    config = validate_config()
    saved_files_dir = os.path.expanduser(config.get("saved_files_dir"))
    os.makedirs(saved_files_dir, exist_ok=True)
    pipeline_type = config["pipeline_type"]
    window = config["window"]
    targets = config["prediction_target"]
    cols_to_drop = config.get("cols_to_drop", [])
    resample_frequency = config.get("data_frequency")
    if pipeline_type == "gmlp":
        pipeline = gMLP_pipeline()
        print("loading gMLP model from", os.path.join(saved_files_dir, "trained_model_gmlp.pth"))
        pipeline.model = torch.load(
            os.path.join(saved_files_dir, "trained_model_gmlp.pth"), 
            map_location="cpu", 
            weights_only=False)
        pipeline.scaler_x = pickle.load(open(os.path.join(saved_files_dir, "scaler_x_gmlp.pkl"), "rb"))
        pipeline.scaler_y = pickle.load(open(os.path.join(saved_files_dir, "scaler_y_gmlp.pkl"), "rb"))
        pipeline.clipping_min = pickle.load(open(os.path.join(saved_files_dir, "clipping_min_gmlp.pkl"), "rb"))
        pipeline.clipping_max = pickle.load(open(os.path.join(saved_files_dir, "clipping_max_gmlp.pkl"), "rb"))
        pipeline.conformal_q = pickle.load(open(os.path.join(saved_files_dir, "conformal_q_gmlp.pkl"), "rb"))
    elif pipeline_type == "xgb":
        pipeline = XGBoost_pipeline()
        print("loading xgboost model from", os.path.join(saved_files_dir, "trained_model_xgb.pkl"))
        pipeline.model = pickle.load(open(os.path.join(saved_files_dir, "trained_model_xgb.pkl"), "rb"))
        pipeline.scaler_x = pickle.load(open(os.path.join(saved_files_dir, "scaler_x_xgb.pkl"), "rb"))
        pipeline.scaler_y = pickle.load(open(os.path.join(saved_files_dir, "scaler_y_xgb.pkl"), "rb"))
        pipeline.clipping_min = pickle.load(open(os.path.join(saved_files_dir, "clipping_min_xgb.pkl"), "rb"))
        pipeline.clipping_max = pickle.load(open(os.path.join(saved_files_dir, "clipping_max_xgb.pkl"), "rb"))
        pipeline.conformal_q = pickle.load(open(os.path.join(saved_files_dir, "conformal_q_xgb.pkl"), "rb"))
    elif pipeline_type == "itransformer":
        pipeline = iTransformer_pipeline()
        print("loading iTransformer model from", os.path.join(saved_files_dir, "trained_model_itransformer.pth"))
        pipeline.model = torch.load(
            os.path.join(saved_files_dir, "trained_model_itransformer.pth"), 
            map_location="cpu", 
            weights_only=False)
        pipeline.scaler_x = pickle.load(open(os.path.join(saved_files_dir, "scaler_x_itransformer.pkl"), "rb"))
        pipeline.scaler_y = pickle.load(open(os.path.join(saved_files_dir, "scaler_y_itransformer.pkl"), "rb"))
        target_cols = list(targets) if isinstance(targets, (list, tuple)) else [targets]
        pipeline.target_columns = target_cols
        pipeline.n_past = window
        pipeline.n_targets = len(target_cols)
        pipeline.time_column = "ts"
        pipeline.conformal_q = pickle.load(open(os.path.join(saved_files_dir, "conformal_q_itransformer.pkl"), "rb"))
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    data_frequency = pd.to_timedelta(config.get("data_frequency"))
    df = pd.read_parquet(config["parquet_path"])
    max_ts = df["ts"].max().floor(data_frequency)
    pivot_df_bool = config.get("pivot_df")
    if pivot_df_bool:
        df = pivot_df(df)
    if "ts" not in df.columns and df.index.name == "ts":
        df = df.reset_index()
    elif "ts" not in df.columns:
        raise KeyError(f"Column 'ts' missing from data after pivoting. Available columns: {list(df.columns)}")

    # Forward fill trailing values (so the newest polled metrics extend to the latest timestamp)
    df = df.set_index("ts").resample(resample_frequency).mean(numeric_only=True).interpolate("linear").bfill().ffill().reset_index()
    print(df.head())
    last_window = df.iloc[-window:]
    horizon = config.get("horizon")
    first_predicted = last_window.iloc[-1]["ts"] + pd.Timedelta(seconds=data_frequency.total_seconds())
    last_predicted = first_predicted + pd.Timedelta(seconds=data_frequency.total_seconds() * (horizon - 1))

    if pipeline_type == "itransformer":
        excluded = set(cols_to_drop + pipeline.target_columns + ([pipeline.time_column] if pipeline.time_column else []))
        pipeline.feature_columns = [col for col in last_window.columns if col not in excluded]

    window_tensor = pipeline.preprocess_inference(
        last_window,
        targets,
        n_past=window,
        exclude_columns=cols_to_drop,
    )

    predictions = pipeline.infer(window_tensor)
    #print(f"Predictions from {pipeline_type} model: {predictions}")
    #print(f"First predicted timestamp: {first_predicted}")
    #print(f"Last predicted timestamp: {last_predicted}")
    #print(f"Data frequency: {data_frequency}")
    return predictions, first_predicted, last_predicted, data_frequency, pipeline.conformal_q
