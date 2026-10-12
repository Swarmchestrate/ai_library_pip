from math import sqrt
from sklearn.metrics import mean_squared_error
from ..helpers.sliding_window import apply_sliding_window
import numpy as np
import xgboost as xgb
import os
import yaml
from pathlib import Path
from ..helpers.to_saved_files import atomic_save

class XGBoost_pipeline:
    def __init__(self):
        self.scaler_y = None
        self.scaler_x = None
        self.clipping_min = None
        self.clipping_max = None
        self.model = None
        package_root = Path(__file__).resolve().parent.parent.parent
        with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.conformal_q_alpha = config.get("conf_q_alpha")
            self.params = config.get("xgb", {}).get("model_params", {})
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
        exclude_columns=None
        ):
        """
        Create all sliding windows first and then split
        the resulting samples into train/val/test.
        """

        if targets is None:
            exclude_cols = list(exclude_columns or [])
            targets = [c for c in df.columns if c not in exclude_columns]

        X, Y, scaler_x, scaler_y = apply_sliding_window(
            df,
            targets,
            n_future,
            n_past,
            stride,
            exclude_columns,
        )
        if Y.ndim == 2:
            Y = Y[:, None, :]

        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        n_samples = len(X)

        train_end = int(splits[0] * n_samples)
        val_end = int((splits[0] + splits[1]) * n_samples)

        x_train, y_train = X[:train_end], Y[:train_end]
        self.clipping_min = np.quantile(x_train, 0.01, axis=(0, 2), keepdims=True)
        self.clipping_max = np.quantile(x_train, 0.99, axis=(0, 2), keepdims=True)

        x_val, y_val = np.clip(X[train_end:val_end],self.clipping_min,self.clipping_max), Y[train_end:val_end]
        x_test, y_test = np.clip(X[val_end:],self.clipping_min,self.clipping_max), Y[val_end:]
        
        train_ds = (x_train, y_train)
        val_ds = (x_val, y_val)
        test_ds = (x_test, y_test)

        return train_ds, val_ds, test_ds
    
    def train(self, train_ds, val_ds, test_ds):
        x_train, y_train = train_ds
        x_val, y_val = val_ds
        x_test, y_test = test_ds

        # 1. Flatten 3D input features to 2D for XGBoost: (samples, channels, seq_len) -> (samples, channels * seq_len)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        package_root = Path(__file__).resolve().parent.parent.parent
        with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            n_future = config.get("horizon")

        if y_train.ndim == 3:
            n_samples, n_targets, n_future = y_train.shape
            
            # 1. Flatten for training model (bs, n_targets * n_future)
            y_train_xgb = y_train.transpose(0, 2, 1).reshape(n_samples, -1).astype(np.float32)
            y_val_xgb = y_val.transpose(0, 2, 1).reshape(y_val.shape[0], -1).astype(np.float32)
            
            # 2. Keep 2D flattened versions for inverse transformations and evaluation
            y_test_flat = y_test.transpose(0, 2, 1).reshape(-1, n_targets)
            y_val_flat = y_val.transpose(0, 2, 1).reshape(-1, n_targets)
        else:
            y_train_xgb = y_train.astype(np.float32)
            y_val_xgb = y_val.astype(np.float32)
            y_test_flat = y_test
            y_val_flat = y_val
            n_targets = y_train_xgb.shape[1] // n_future

        model = xgb.XGBRegressor(**self.params)

        model.fit(x_train, y_train_xgb)
        preds = model.predict(x_test)
        if preds.ndim == 2 and preds.shape[1] > n_targets:
            preds = preds.reshape(-1, n_future, n_targets).transpose(0, 2, 1).reshape(-1, n_targets)
        
        preds = self.scaler_y.inverse_transform(preds)
        targets = self.scaler_y.inverse_transform(y_test_flat)
        rmse = sqrt(mean_squared_error(preds, targets))

        self.model = model
        atomic_save(self.model, f"{self.saved_files_dir}/trained_model_xgb.pkl")
        atomic_save(self.scaler_x, f"{self.saved_files_dir}/scaler_x_xgb.pkl")
        atomic_save(self.scaler_y, f"{self.saved_files_dir}/scaler_y_xgb.pkl")
        atomic_save(self.clipping_min, f"{self.saved_files_dir}/clipping_min_xgb.pkl")
        atomic_save(self.clipping_max, f"{self.saved_files_dir}/clipping_max_xgb.pkl")

        # conformal prediction intervals
        val_preds = model.predict(x_val) # Note: x_val is already flattened in preprocess_splits
        if val_preds.ndim == 2 and val_preds.shape[1] > n_targets:
            val_preds = val_preds.reshape(-1, n_future, n_targets).transpose(0, 2, 1).reshape(-1, n_targets)
            
        val_preds = self.scaler_y.inverse_transform(val_preds)
        val_targets = self.scaler_y.inverse_transform(y_val_flat) 
        val_residuals = np.abs(val_targets - val_preds)
        
        alpha = self.conformal_q_alpha
        n_val = len(val_residuals)

        # Finite-sample correction formula for Conformal Prediction
        q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val
        q_level = min(q_level, 1.0)  # Safety cap at 1.0

        # Compute the quantile threshold
        self.conformal_q = np.quantile(val_residuals, q_level, axis=0)
        atomic_save(self.conformal_q, f"{self.saved_files_dir}/conformal_q_xgb.pkl")
        
        return model, rmse, self.conformal_q


    def preprocess_inference(self, df, targets, n_past, exclude_columns=None):
        exclude_columns = list(exclude_columns or [])

        # 1. Drop non-feature metadata columns (e.g., 'ts')
        df_full = df.drop(columns=exclude_columns, errors="ignore")

        # 2. Reindex columns to match the exact schema and sequence expected by scaler_x
        if hasattr(self.scaler_x, "feature_names_in_"):
            df_full = df_full.reindex(columns=self.scaler_x.feature_names_in_, fill_value=0.0)

        # 3. Scale all columns using scaler_x (since apply_sliding_window fits scaler_x on all columns)
        X = self.scaler_x.transform(df_full)

        # 4. Extract sliding window slice & shape to (1, channels, seq_len)
        window = X[-n_past:]
        window = window.T[None, ...]  # Shape: (1, channels, seq_len)

        # 5. Clip features using training set quantiles and flatten for XGBoost (1, channels * seq_len)
        window = np.clip(window, self.clipping_min, self.clipping_max).astype(np.float32)
        window = window.reshape(1, -1)

        return window

    def infer(self, window):
        package_root = Path(__file__).resolve().parent.parent.parent
        with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            n_future = config.get("horizon")
    
        if self.model is None:
            raise RuntimeError("model was not trained yet, please run the train method first.")
        
        # XGBoost predict directly on numpy array
        preds_xgb = self.model.predict(window) # Shape: (1, n_targets * n_future)
        
        # Infer shapes from model output dimensions or config if available; here we deduce n_future and n_targets
        c_out = preds_xgb.shape[1]
        n_targets = c_out // n_future

        preds_flat = preds_xgb.reshape(-1, n_future, n_targets).transpose(0, 2, 1).reshape(-1, n_targets)
        preds = self.scaler_y.inverse_transform(preds_flat)
        preds = preds.reshape(n_future, n_targets)
        #preds = np.clip(preds, a_min=0, a_max=None)

        return preds