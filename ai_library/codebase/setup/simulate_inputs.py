import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_timestamps(seconds, mode):
    now = pd.Timestamp.now().floor('s')
    if mode == "train":
        return pd.date_range(end=now, periods=seconds, freq='15s')
    elif mode == "infer":
        return pd.date_range(start=now, periods=seconds, freq='15s')
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'infer'.")

def query_metric_values_raw(node_id, metric_names, seconds, mode):
    ts = get_timestamps(seconds, mode)
    dictlist = []
    for node in node_id:
        for metric in metric_names:
            x = np.linspace(0, 35 * np.pi, seconds)
            y = np.sin(x)
            dictlist.append({
                "node": node,
                "metric": metric,
                "timestamps": ts,
                "values": y
            })
    return dictlist

def query_metric_values_composite(metric_names, seconds, mode):
    ts = get_timestamps(seconds, mode)
    dictlist = []
    for metric in metric_names:
        x = np.linspace(0, 35 * np.pi, seconds)
        y = np.sin(x)
        dictlist.append({
            "timestamps": ts,
            "metric": metric,
            "values": y
        })
    return dictlist

def synthesize_metric_values(seconds, mode):
    # Kept for backward compatibility if called elsewhere
    ts = get_timestamps(seconds, mode)
    x = np.linspace(0, 35 * np.pi, seconds)
    y = np.sin(x)
    return y, ts

def preprocess(node_ids, raw_metrics, composite_metrics, seconds, mode):
    raw_data = query_metric_values_raw(node_ids, raw_metrics, seconds, mode)
    composite_data = query_metric_values_composite(composite_metrics, seconds, mode)
    
    series_dict = {}
    for item in raw_data:
        col_name = f"{item['metric']}_{item['node']}"
        series_dict[col_name] = pd.Series(item['values'], index=pd.to_datetime(item['timestamps']))
    
    for item in composite_data:
        col_name = f"{item['metric']}_composite"
        series_dict[col_name] = pd.Series(item['values'], index=pd.to_datetime(item['timestamps']))
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
    print(df.head())

    df = preprocess(node_ids, raw_metrics, composite_metrics, seconds, "infer")
    print(df.head())
