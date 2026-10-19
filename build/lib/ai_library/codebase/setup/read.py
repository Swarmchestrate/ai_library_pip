import pandas as pd
import pyarrow.dataset as ds
import yaml
import os
from pathlib import Path

package_root = Path(__file__).resolve().parent.parent.parent
with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    parquet_path = os.path.expanduser(config["parquet_path"])
    DATA_DIR = os.path.expanduser(config["data_dir"]) 
    resample_frequency = config["data_frequency"]

# 2. Create the directory on disk if it doesn't exist yet
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

# Get both standard and raw metrics from config.yaml
STANDARD_METRICS = config.get("STANDARD_METRICS", [])
RAW_METRICS = config.get("RAW_METRICS", {})

# Combine all metrics you want to read/process from storage
ALL_METRICS = list(set(STANDARD_METRICS) | set(RAW_METRICS.keys()))

# 1. Load the dataset
dataset = ds.dataset(DATA_DIR, format="parquet", partitioning="hive")

# 2. Filter for ALL configured metrics (both standard and raw) using partition pruning
table = dataset.to_table(
    filter=(ds.field("metric").isin(ALL_METRICS))
)

# 3. Process the DataFrame
df = table.to_pandas()

# Clean up NaN values in the node column:
# Standard mode metrics don't have node labels, but raw metrics do.
df["node"] = df["node"].fillna("all_nodes")

df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
df = df.sort_values("ts").reset_index(drop=True)

# 4. For forecasting: Resample per metric and node combination 
# (Since raw metrics have specific node IDs, grouping by both metric and node 
# ensures you don't mix up different node streams if you have multiple nodes later).
series = (
    df.set_index("ts")
    .groupby(["metric", "node"])["value"]
    .resample(f"{resample_frequency}")
    .mean()
)

# 5. Save the full dataframe to a Parquet file without the row index numbers
df.to_parquet(parquet_path, index=False)