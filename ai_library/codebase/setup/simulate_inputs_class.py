import time
import math
import yaml

import pandas as pd

#with open("config.yaml", "r") as f:
#    config = yaml.safe_load(f)
#    parquet_path = config["parquet_path"]
parquet_path = "metrics_output.parquet"  # Default path if config.yaml is not used
parquet = pd.read_parquet(parquet_path)
import matplotlib.pyplot as plt
plt.plot(parquet["cpu_usage_comp"])
plt.show()
class MetricStreamSimulator:
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval  # How often a new sample arrives (in seconds)
        self._buffers = {}
        self._last_produced = {}

    def query_metric_values(self, metric: str) -> list[dict]:
        """
        Simulates query_metric_values(metric: str) -> list[dict[str, Any]]
        Returns the currently buffered sample and consumes it.
        """
        now = time.time()

        # Initialize tracking for a new metric name
        if metric not in self._last_produced:
            self._last_produced[metric] = 0
            self._buffers[metric] = None

        # Simulate a background producer generating a new sample every `update_interval` seconds
        if now - self._last_produced[metric] >= self.update_interval:
            # Generate a smooth, non-random, changing value using a sine wave 
            # (e.g., cycling gracefully between ~20.0 and ~80.0)
            base_value = 50.0 + 30.0 * math.sin(now / 5.0)
            
            self._buffers[metric] = {
                "timestamp": now,
                "value": round(base_value, 2)
            }
            self._last_produced[metric] = now

        # Retrieve and consume the buffer (mimicking the one-entry consumption rule)
        sample = self._buffers[metric]
        if sample is not None:
            self._buffers[metric] = None  # Consume the sample
            return [sample]
        
        return []  # Empty when no new sample has arrived[cite: 1]

import time
import math

class RawMetricStreamSimulator:
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._raw_buffers = {}  # Format: {metric: [{ "source": str, "timestamp": float, "value": float }, ...]}
        self._last_produced = {}

    def query_metric_values_raw(self, metric: str, seconds: int) -> dict[str, list[dict]]:
        """
        Simulates query_metric_values_raw(metric: str, seconds: int) -> dict[str, list[dict[str, Any]]]
        Returns buffered raw metric values from the last `seconds` and consumes them.
        """
        now = time.time()

        # Initialize storage for new metrics
        if metric not in self._last_produced:
            self._last_produced[metric] = 0
            self._raw_buffers[metric] = []

        # Simulate background producers adding raw samples periodically for multiple sources (e.g., two IPs)
        sources = ["10.0.0.1", "10.0.0.2"]
        if now - self._last_produced[metric] >= self.update_interval:
            for i, src in enumerate(sources):
                # Generate a smooth, non-random wave value unique per source
                base_value = 50.0 + 25.0 * math.sin((now / 4.0) + i)
                
                self._raw_buffers[metric].append({
                    "source": src,
                    "timestamp": now - (i * 0.1), # slight offset
                    "value": round(base_value, 2)
                })
            self._last_produced[metric] = now

        # Define the time window cutoff
        cutoff_time = now - seconds

        # Filter samples that fall within the requested time window
        retained_samples = []
        result_dict = {}

        for sample in self._raw_buffers[metric]:
            if sample["timestamp"] >= cutoff_time:
                # Matches the time window -> include in output
                src = sample["source"]
                if src not in result_dict:
                    result_dict[src] = []
                result_dict[src].append({
                    "timestamp": sample["timestamp"],
                    "value": sample["value"]
                })
            else:
                # Too old, keep it in the buffer for future queries covering wider windows
                retained_samples.append(sample)

        # Update buffer by consuming the returned samples (keeping only un-queried or out-of-window ones)
        self._raw_buffers[metric] = retained_samples

        return result_dict

import time
import math
import pandas as pd

# --- Include your simulator classes here ---
# (Assuming MetricStreamSimulator and RawMetricStreamSimulator are defined as above)

def collect_metrics_to_parquet(filename="metrics_output.parquet", duration_seconds=15, poll_interval=1.0, metric_name="metric_comp"):
    # Initialize simulators
    comp_simulator = MetricStreamSimulator(update_interval=1.0)
    raw_simulator = RawMetricStreamSimulator(update_interval=1.0)
    
    collected_data = []
    
    print(f"Starting metric collection for {duration_seconds} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        timestamp = time.time()
        
        # 1. Query composite data (metric_comp)
        comp_result = comp_simulator.query_metric_values(metric_name)
        comp_value = comp_result[0]["value"] if comp_result else None
        print(comp_result)
        # 2. Query raw node data (metric_node)
        # We look at a small window (e.g., 2 seconds) to catch incoming samples
        raw_result = raw_simulator.query_metric_values_raw(metric_name, seconds=2)
        print(raw_result)
        # Build the row dictionary with the timestamp and composite value
        row = {
            "timestamp": timestamp,
            f"{metric_name}_comp": comp_value
        }
        
        # Unpack raw data sources into columns matching `metric_node_<node_id>`
        for source, samples in raw_result.items():
            # Clean source IP/name for a valid column name (e.g., "10.0.0.1" -> "10_0_0_1")
            safe_node_name = source.replace(".", "_")
            column_name = f"{metric_name}_node_{safe_node_name}"
            
            # If samples arrived in this window, grab the latest value
            if samples:
                row[column_name] = samples[-1]["value"]
            else:
                row[column_name] = None
                
        collected_data.append(row)
        
        # Wait for the next polling interval
        time.sleep(poll_interval)

    # Convert collected rows into a Pandas DataFrame
    df = pd.DataFrame(collected_data)
    
    # Optional: Set timestamp as index or keep it as a column
    df.set_index("timestamp", inplace=True)
    
    # Save to Parquet format
    print(f"\nCollection complete! Data successfully saved to '{parquet_path}'.")
    print("\nPreview of the DataFrame:")
    print(df.head())
    return df
import os

if __name__ == "__main__":
    while True:
        # 1. Collect individual metric DataFrames
        df_cpu_usage = collect_metrics_to_parquet(duration_seconds=0.1, poll_interval=0.1, metric_name="cpu_usage")
        df_cpu_temp = collect_metrics_to_parquet(duration_seconds=0.1, poll_interval=0.1, metric_name="cpu_temp")
        df_mem_usage = collect_metrics_to_parquet(duration_seconds=0.1, poll_interval=0.1, metric_name="mem_usage")

        # 2. Convert indices to datetime so Pandas can perform time-based resampling
        for df in [df_cpu_usage, df_cpu_temp, df_mem_usage]:
            if not df.empty:
                df.index = pd.to_datetime(df.index, unit='s')

        # 3. Combine all metrics side-by-side using an outer join on the timestamp index
        combined_df = pd.concat([df_cpu_usage, df_cpu_temp, df_mem_usage], axis=1)

        # 4. Resample to 3-second intervals (taking the mean value across each bucket)
        resampled_df = combined_df.resample('3s').mean()

        # 5. Append or merge with existing Parquet data instead of replacing it
        if os.path.exists(parquet_path):
            try:
                existing_df = pd.read_parquet(parquet_path)
                
                # Combine existing data with new data
                resampled_df = pd.concat([existing_df, resampled_df])
                
                # Remove duplicate timestamps if they overlap, keeping the latest entry
                resampled_df = resampled_df[~resampled_df.index.duplicated(keep='last')]
                
                # Sort chronologically by timestamp index
                resampled_df.sort_index(inplace=True)
            except Exception as e:
                print(f"Warning: Could not read existing parquet file ({e}). Creating a new one.")

        # 6. Flush/save the cumulative dataframe to Parquet
        resampled_df.to_parquet(parquet_path, engine="pyarrow")
        
        print(f"\nSuccessfully combined, resampled, appended, and flushed to '{parquet_path}'.")
        print(f"Total rows in Parquet file now: {len(resampled_df)}")
        print("\nPreview of Cumulative DataFrame:")
        print(resampled_df.tail()) # Show tail to see latest appended rows

        print("\nWaiting for the next collection cycle...\n")
        #time.sleep(3) # Adjust sleep time as needed