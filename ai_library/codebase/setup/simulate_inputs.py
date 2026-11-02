import pandas as pd
import numpy as np

def get_timestamps(seconds, mode):
    now = pd.Timestamp.now().floor('s')
    if mode == "train":
        return pd.date_range(end=now, periods=seconds, freq='s')
    elif mode == "infer":
        return pd.date_range(start=now, periods=seconds, freq='s')
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'infer'.")

def query_metric_values_raw(node_id, metric_names, seconds, mode):
    """
    Simulates matching the original shape/structure where a dictionary 
    wraps the lists of sample dicts per node/metric.
    """
    ts = get_timestamps(seconds, mode)
    x = np.linspace(0, 35 * np.pi, seconds)
    y = np.sin(x)
    
    result = {}
    for node in node_id:
        for metric in metric_names:
            key = f"{metric}_{node}"
            # Format matching the sample structure [{"timestamp": ..., "value": ...}, ...]
            samples = [
                {"timestamp": t, "value": val} 
                for t, val in zip(ts, y)
            ]
            result[key] = samples
            
    return result

def query_metric_values_composite(metric_names, seconds, mode):
    ts = get_timestamps(seconds, mode)
    x = np.linspace(0, 35 * np.pi, seconds)
    y = np.sin(x)
    
    result = {}
    for metric in metric_names:
        key = f"{metric}_composite"
        samples = [
            {"timestamp": t, "value": val} 
            for t, val in zip(ts, y)
        ]
        result[key] = samples
        
    return result

def preprocess(node_ids, raw_metrics, composite_metrics, seconds, mode):
    raw_data = query_metric_values_raw(node_ids, raw_metrics, seconds, mode)
    composite_data = query_metric_values_composite(composite_metrics, seconds, mode)

    series_dict = {}
    
    # Unpack the dictionary of lists into pandas Series
    for key, samples in raw_data.items():
        timestamps = [s["timestamp"] for s in samples]
        values = [s["value"] for s in samples]
        series_dict[key] = pd.Series(values, index=pd.to_datetime(timestamps))
    
    for key, samples in composite_data.items():
        timestamps = [s["timestamp"] for s in samples]
        values = [s["value"] for s in samples]
        series_dict[key] = pd.Series(values, index=pd.to_datetime(timestamps))
        
    df = pd.DataFrame(series_dict)
    
    # Extract useful numeric components from the timestamp
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['minute'] = df.index.minute
    df['ts'] = df.index
    return df

if __name__ == "__main__":
    node_ids = ['node1', 'node2', 'node3']
    raw_metrics = ['metric1', 'metric2', 'metric3']
    composite_metrics = ['metric4', 'metric5']
    seconds = 10
    
    df = preprocess(node_ids, raw_metrics, composite_metrics, seconds, "train")
    #print(df.head())

    df = preprocess(node_ids, raw_metrics, composite_metrics, seconds, "infer")
    #print(df.head())

    print(query_metric_values_composite(composite_metrics, seconds, "train"))