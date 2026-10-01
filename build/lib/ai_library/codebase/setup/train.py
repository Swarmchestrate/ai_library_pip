import pandas as pd
from ..models.gmlp_class import gMLP_pipeline
from ..models.xgb_class import XGBoost_pipeline
import yaml
from ..models.iTransformer_class import iTransformer_pipeline
from ..helpers.pivot_df import pivot_df
from ..helpers.config_helper import validate_config


def get_last_window_data_and_train():
    from . import read
    config = validate_config()
    train_window = config["window"]
    train_horizon = config["horizon"]
    splits = config.get("splits", [0.7, 0.15, 0.15])
    stride = config.get("stride", train_horizon)
    parquet_path = config["parquet_path"]
    resample_frequency = config["data_frequency"]
    pipeline_type = config["pipeline_type"]
    targets = config["prediction_target"]
    cols_to_drop = config["cols_to_drop"]
    parquet_train_size = config.get("parquet_train_size")
    if pipeline_type == "gmlp":
        pipeline = gMLP_pipeline()
    elif pipeline_type == "xgb":
        pipeline = XGBoost_pipeline()
    elif pipeline_type == "itransformer":
        pipeline = iTransformer_pipeline()
    df = pd.read_parquet(parquet_path)
    print(df.head())
    # Basic safety checks: parquet should contain time column and enough rows
    if df.empty:
        raise ValueError(f"No data found in parquet file: {parquet_path}")
    if "ts" not in df.columns:
        raise ValueError(f"Parquet file {parquet_path} missing required 'ts' column")
    required_rows = int((train_window + train_horizon + stride) / min(splits))
    pivot_df_bool = config.get("pivot_df")
    if pivot_df_bool:
        df = pivot_df(df)
    print(df.head())
    df = df.set_index("ts").resample(resample_frequency).mean(numeric_only=True).interpolate("linear").bfill().ffill().reset_index()
    if parquet_train_size:
        df = df.tail(parquet_train_size)
    print(df.head())
    #if len(df) < required_rows:
    #        raise ValueError(
    #            f"Not enough rows in parquet file: found {len(df)}, need at least {required_rows} (window + horizon + stride)/min(splits)"
    #        )
    if pipeline_type == "gmlp":
        dls, test_dl = pipeline.preprocess_splits(df, targets, splits, train_horizon, train_window, stride, cols_to_drop)
        model, rmse, conformal_q = pipeline.train(dls, test_dl)
    elif pipeline_type == "xgb":
        train_ds, val_ds, test_ds = pipeline.preprocess_splits(df,targets, splits, train_horizon, train_window, stride, cols_to_drop)
        model, rmse, conformal_q = pipeline.train(train_ds, val_ds, test_ds)
    elif pipeline_type == "itransformer":
        dls, test_dls = pipeline.preprocess_splits(df, targets, splits, train_horizon, train_window, stride, cols_to_drop)
        model, rmse, conformal_q = pipeline.train(dls, test_dls)
    print(f"Trained {pipeline_type} model with RMSE: {rmse}")

if __name__ == "__main__":
    get_last_window_data_and_train()