from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tsai.data.preparation import apply_sliding_window
from tsai.data.preparation import SlidingWindow
import numpy as np

def apply_sliding_window(
    df,
    targets,
    n_future,
    n_past,
    stride,
    exclude_columns=None,
    scaler=None,
    scaler_labels=None,
    splits=[0.7, 0.1, 0.2],
):
    if exclude_columns is None:
        exclude_columns = []

    # Keep only target/feature columns
    df = df.drop(columns=exclude_columns)
    target_cols = list(targets) if isinstance(targets, (list, tuple)) else [targets]

    train_cutoff = int(splits[0] * df.shape[0])

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df.iloc[:train_cutoff])

    scaled_features = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

    # If targets cover all columns, target scaler is identical to feature scaler
    if set(target_cols) == set(df.columns):
        scaler_labels = scaler
    elif scaler_labels is None:
        scaler_labels = MinMaxScaler()
        scaler_labels.fit(df[target_cols].iloc[:train_cutoff])

    X_data, y_data = SlidingWindow(
        window_len=n_past,
        horizon=n_future,
        get_y=target_cols,
        stride=stride,
    )(scaled_df)

    return X_data.astype(np.float32), y_data.astype(np.float32), scaler, scaler_labels