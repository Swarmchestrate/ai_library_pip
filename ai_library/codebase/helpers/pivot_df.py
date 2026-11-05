import yaml
from pathlib import Path
def pivot_df(df):
    package_root = Path(__file__).resolve().parent.parent.parent
    with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_frequency = config.get("data_frequency")  # Default fallback if not specified
    df = df.copy()
    df["feature"] = df["metric_name"] + "_" + df["node_ip"].astype(str)
    df_indexed = df.set_index("ts")
    resampled = df_indexed.groupby("feature")["metric_value"].resample(data_frequency).mean(numeric_only=True)
    pivoted_df = resampled.unstack(level="feature")
    pivoted_df = pivoted_df.reset_index()
    pivoted_df.columns.name = None
    return pivoted_df
import matplotlib.pyplot as plt

def plot_complete_timesteps(pivoted_df):
    # 1. Drop any rows that contain NaN in ANY column
    complete_df = pivoted_df.dropna()
    
    if complete_df.empty:
        print("No timesteps exist where all metrics are non-NaN simultaneously!")
        return

    # 2. Set 'ts' as index if it isn't already, so it plots nicely on the X-axis
    if 'ts' in complete_df.columns:
        complete_df = complete_df.set_index('ts')

    # 3. Plot the complete periods
    plt.figure(figsize=(12, 6))
    
    # Loop through each metric column and plot it
    for column in complete_df.columns:
        plt.plot(complete_df.index, complete_df[column], label=column, marker='.', linestyle='-')
        
    plt.title("Metrics Over Complete Timesteps (No NaN Rows)")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save or display it
    plt.savefig("complete_metrics_plot.png")
