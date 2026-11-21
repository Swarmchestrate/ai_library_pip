from ..helpers.config_helper import validate_config
import os
import pandas as pd

def get_inference_parquet_path():
    config = validate_config()
    inference_parquet_path = os.path.expanduser(config.get("inference_parquet_path"))
    return inference_parquet_path

def ingest_to_inference_parquet(predictions, first_prediction_timestamp, last_prediction_timestamp, frequency):
    inference_parquet_path = get_inference_parquet_path()
    new_data = pd.DataFrame(predictions)
    timestamps = pd.date_range(start=first_prediction_timestamp, end=last_prediction_timestamp, freq=frequency)
    new_data["ts"] = pd.to_datetime(timestamps, unit="s", utc=True)
    if not os.path.exists(inference_parquet_path):
        new_data.to_parquet(inference_parquet_path, index=False)
    else:
        existing_data = pd.read_parquet(inference_parquet_path)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=["ts"], keep="last")
        combined_data = combined_data.sort_values(by="ts").reset_index(drop=True)
        combined_data.to_parquet(inference_parquet_path, index=False)
    