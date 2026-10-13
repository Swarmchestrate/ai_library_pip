from math import sqrt
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.data.core import DataLoaders
from fastai.metrics import rmse
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.nn import MSELoss
from torch.utils.data import Dataset
from tsai.learner import Learner
from ..helpers.to_saved_files import atomic_save
from .itransformer_model import Model
import yaml
import os
from pathlib import Path

def _time_features(values):
    """Official iTransformer calendar features for minute-resolution data."""
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.hasnans:
        raise ValueError("time column contains missing timestamps")
    return np.column_stack(
        [
            index.minute / 59.0 - 0.5,
            index.hour / 23.0 - 0.5,
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5,
        ]
    ).astype(np.float32)


class _WindowDataset(Dataset):
    """Create forecasting windows on demand from one scaled 2D array."""

    def __init__(self, data, time_features, origins, n_past, n_future, n_targets):
        self.data = data
        self.time_features = time_features
        self.origins = np.asarray(origins, dtype=np.int64)
        self.n_past = n_past
        self.n_future = n_future
        self.n_targets = n_targets

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        past = slice(origin - self.n_past, origin)
        future = slice(origin, origin + self.n_future)

        x = torch.from_numpy(self.data[past].T)
        x_mark = torch.from_numpy(self.time_features[past])
        y = self.data[future, -self.n_targets :].T
        if self.n_targets == 1:
            y = y[0]
        return x, x_mark, torch.from_numpy(y)


class _ITAdapter(nn.Module):
    """Bridge channels-first batches and keep only the target forecasts."""

    def __init__(self, model, n_targets=1):
        super().__init__()
        self.model = model
        self.n_targets = n_targets

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)  # [B, n_vars, n_past] -> [B, n_past, n_vars]
        if x_mark is not None and x_mark.shape[-1] == 0:
            x_mark = None
        out = self.model(x, x_mark)[:, :, -self.n_targets :]
        if self.n_targets == 1:
            return out.reshape(out.shape[0], -1)
        return out.permute(0, 2, 1)


class iTransformer_pipeline:
    def __init__(
        self,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        dimensions = (d_model, n_heads, e_layers, d_ff)
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        
        self.model_config = {
            "d_model": d_model,
            "n_heads": n_heads,
            "e_layers": e_layers,
            "d_ff": d_ff,
            "dropout": dropout,
        }
        self.scaler_y = None
        self.scaler_x = None
        self.model = None
        self.pred_len = None
        self.n_past = None
        self.n_targets = None
        self.target_columns = None
        self.feature_columns = None
        self.time_column = None
        package_root = Path(__file__).resolve().parent.parent.parent
        with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.saved_files_dir = os.path.expanduser(config.get("saved_files_dir"))
            os.makedirs(self.saved_files_dir, exist_ok=True)


    def preprocess_splits(
        self,
        df,
        targets,
        splits=(0.7, 0.1, 0.2),
        n_future=96,
        n_past=288,
        stride=96,
        exclude_columns=None,
        bs=12,
        time_column="ts",
    ):
        """Build chronological, target-disjoint lazy train/val/test windows."""
        if len(splits) != 3 or any(value <= 0 for value in splits) or not np.isclose(sum(splits), 1):
            raise ValueError("splits must contain three positive values that sum to 1")
        if any(not isinstance(value, int) or value <= 0 for value in (n_future, n_past, stride, bs)):
            raise ValueError("n_future, n_past, stride, and bs must be positive integers")

        target_columns = list(targets) if isinstance(targets, (list, tuple)) else [targets]
        exclude_columns = list(exclude_columns or [])

        if targets is None:
            excluded_set = set(exclude_columns + ([time_column] if time_column else []))
            target_columns = [col for col in df.columns if col not in excluded_set]
        else:
            target_columns = list(targets) if isinstance(targets, (list, tuple)) else [targets]
        required = target_columns + exclude_columns + ([time_column] if time_column else [])
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")

        excluded = set(exclude_columns + target_columns + ([time_column] if time_column else []))
        feature_columns = [column for column in df.columns if column not in excluded]
        if not feature_columns:
            feature_columns = target_columns

        if time_column:
            timestamps = pd.to_datetime(df[time_column], errors="raise")
            if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
                raise ValueError("time column must contain valid chronological timestamps")
            marks = _time_features(timestamps)
        else:
            marks = np.empty((len(df), 0), dtype=np.float32)

        train_end = int(splits[0] * len(df))
        val_end = train_end + int(splits[1] * len(df))
        origins = (
            np.arange(n_past, train_end - n_future + 1, stride),
            np.arange(train_end, val_end - n_future + 1, stride),
            np.arange(val_end, len(df) - n_future + 1, stride),
        )
        if any(len(split_origins) == 0 for split_origins in origins):
            raise ValueError("each split must contain at least one complete forecast window")

        feature_df = df[feature_columns]
        target_df = df[target_columns]
        self.scaler_x = MinMaxScaler().fit(feature_df.iloc[:train_end])
        self.scaler_y = MinMaxScaler().fit(target_df.iloc[:train_end])
        data = np.concatenate(
            [self.scaler_x.transform(feature_df), self.scaler_y.transform(target_df)], axis=1
        ).astype(np.float32)
        if not np.isfinite(data).all():
            raise ValueError("features and targets must contain finite numeric values")

        self.pred_len = n_future
        self.n_past = n_past
        self.n_targets = len(target_columns)
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        self.time_column = time_column

        datasets = [
            _WindowDataset(data, marks, split_origins, n_past, n_future, self.n_targets)
            for split_origins in origins
        ]
        dls = DataLoaders.from_dsets(*datasets, bs=bs, num_workers=0)
        return dls, dls[2]

    def train(self, dls, test_dl, lr=1e-3, epochs=10, patience=10):
        package_root = Path(__file__).resolve().parent.parent.parent
        with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    self.params = config.get("itransformer", {}).get("model_params", {})

        epochs = self.params.get("epochs", epochs)
        lr = self.params.get("lr", lr)
        patience = self.params.get("patience", patience)
        if self.pred_len is None:
            raise RuntimeError("data was not preprocessed; run preprocess_splits first")

        xb, _, _ = dls.one_batch()
        model = _ITAdapter(
            Model(seq_len=xb.shape[-1], pred_len=self.pred_len, **self.model_config),
            n_targets=self.n_targets,
        )
        callbacks = [
            SaveModelCallback(monitor="valid_loss", fname="itransformer_best"),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience),
        ]

        with TemporaryDirectory(prefix="itransformer-") as checkpoint_dir:
            learn = Learner(
                dls,
                model,
                loss_func=MSELoss(),
                metrics=rmse,
                cbs=callbacks,
                path=checkpoint_dir,
                model_dir=".",
            )
            learn.fit_one_cycle(epochs, lr)
            preds, targets = learn.get_preds(dl=test_dl)
            preds_val, targets_val = learn.get_preds(dl=dls.valid)
            preds_val = self.scaler_y.inverse_transform(self._targets_2d(preds_val))
            targets_val = self.scaler_y.inverse_transform(self._targets_2d(targets_val))

        #perform conformal prediction interval calculation
        val_residuals = np.abs(preds_val - targets_val)
        alpha = 0.05  
        n_val = len(val_residuals)

        # 3. Finite-sample correction formula for Conformal Prediction
        q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val
        q_level = min(q_level, 1.0)  # Safety cap at 1.0

        # 4. Compute the quantile threshold
        # axis=0 calculates a distinct margin for each future timestep in your horizon
        self.conformal_q = np.quantile(val_residuals, q_level, axis=0)

        preds = self.scaler_y.inverse_transform(self._targets_2d(preds))
        targets = self.scaler_y.inverse_transform(self._targets_2d(targets))
        rmse_test = sqrt(mean_squared_error(preds, targets))

        self.model = model
        atomic_save(self.model, f"{self.saved_files_dir}/trained_model_itransformer.pth",use_pytorch=True)
        atomic_save(self.scaler_x, f"{self.saved_files_dir}/scaler_x_itransformer.pkl")
        atomic_save(self.scaler_y, f"{self.saved_files_dir}/scaler_y_itransformer.pkl")
        atomic_save(self.conformal_q, f"{self.saved_files_dir}/conformal_q_itransformer.pkl")
        return model, rmse_test, self.conformal_q

    def preprocess_inference(self, df, targets, n_past, exclude_columns=None):
        if self.scaler_x is None:
            raise RuntimeError("data was not preprocessed; run preprocess_splits first")

        target_columns = list(targets) if isinstance(targets, (list, tuple)) else [targets]
        if target_columns != self.target_columns:
            raise ValueError(f"targets must match training order: {self.target_columns}")
        if n_past != self.n_past:
            raise ValueError(f"n_past must match the trained value: {self.n_past}")
        if len(df) < n_past:
            raise ValueError(f"inference data must contain at least {n_past} rows")

        exclude_columns = list(exclude_columns or [])
        required = target_columns + exclude_columns + ([self.time_column] if self.time_column else [])
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")

        excluded = set(exclude_columns + target_columns + ([self.time_column] if self.time_column else []))
        feature_columns = [column for column in df.columns if column not in excluded]
        if feature_columns != self.feature_columns:
            raise ValueError(f"feature columns must match training order: {self.feature_columns}")

        if self.time_column:
            timestamps = pd.to_datetime(df[self.time_column], errors="raise")
            if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
                raise ValueError("time column must contain valid chronological timestamps")
            marks = _time_features(timestamps)
        else:
            marks = np.empty((len(df), 0), dtype=np.float32)

        data = np.concatenate(
            [
                self.scaler_x.transform(df[feature_columns]),
                self.scaler_y.transform(df[target_columns]),
            ],
            axis=1,
        ).astype(np.float32)
        if not np.isfinite(data).all():
            raise ValueError("features and targets must contain finite numeric values")

        window = data[-n_past:].T[None, ...]
        mark_window = marks[-n_past:][None, ...]
        return window, mark_window

    def infer(self, inputs, device="cpu"):
        if self.model is None:
            raise RuntimeError("model was not trained yet, please run the train method first")
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("inputs must be the result of preprocess_inference")

        window, marks = inputs
        window = torch.as_tensor(window, dtype=torch.float32, device=device)
        marks = torch.as_tensor(marks, dtype=torch.float32, device=device)
        model = self.model.to(device)
        model.eval()

        with torch.no_grad():
            preds = model(window, marks)

        preds = self._targets_2d(preds.cpu().numpy())
        return self.scaler_y.inverse_transform(preds)

    def _targets_2d(self, values):
        values = np.asarray(values)
        if self.n_targets == 1:
            return values.reshape(-1, 1)
        return np.transpose(values, (0, 2, 1)).reshape(-1, self.n_targets)
