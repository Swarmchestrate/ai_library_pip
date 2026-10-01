from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tsai.data.preparation import apply_sliding_window
from tsai.data.preparation import SlidingWindow
import numpy as np

def apply_sliding_window(df, targets, n_future, n_past, stride, exclude_columns, scaler=None, scaler_labels=None, splits=[0.7,0.1,0.2]):
    '''exclude time steps as well here. just features per row'''

    target = df[pd.Index(targets) if isinstance(targets, (list, tuple)) else [targets]]
    if exclude_columns is None:
        exclude_columns = []

    exclude_columns = list(exclude_columns) + list(target.columns)
    df = df.drop(columns=exclude_columns)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[:int(splits[0]*df.shape[0])])
    if scaler_labels is None:
        scaler_labels = MinMaxScaler() 
        scaler_labels.fit(target[:int(splits[0]*target.shape[0])])

    scaled_features = scaler.transform(df)
    scaled_labels = scaler_labels.transform(target)

    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
    scaled_df[target.columns] = scaled_labels

    X_data, y_data = SlidingWindow(
    window_len = n_past,
    horizon = n_future,
    get_y = target.columns.tolist(),
    stride = stride
    )(scaled_df)

    return [X_data.astype(np.float32), y_data.astype(np.float32), scaler, scaler_labels]
