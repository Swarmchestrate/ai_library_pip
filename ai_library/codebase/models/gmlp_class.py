import os
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.metrics import rmse
from sklearn.metrics import mean_squared_error
from torch.nn import HuberLoss
from tsai.data.core import TSDataLoader, TSDataLoaders, TSDatasets
from tsai.learner import Learner
from tsai.models.gMLP import gMLP

from ..helpers.sliding_window import apply_sliding_window
from ..helpers.to_saved_files import atomic_save

# 2. Wrapper class to reshape 2D output (bs, c_out) to 3D (bs, n_targets, n_future)
class ReshapedgMLP(torch.nn.Module):
    def __init__(self, model, n_targets, n_future):
        super().__init__()
        self.model = model
        self.n_targets = n_targets
        self.n_future = n_future

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, self.n_targets, self.n_future)

class gMLP_pipeline:

    def __init__(self):
        self.scaler_y = None
        self.scaler_x = None
        self.clipping_min = None
        self.clipping_max = None
        self.model = None
        package_root = Path(__file__).resolve().parent.parent.parent
        with (package_root / "config.yaml").open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.params = config.get("gmlp", {}).get("model_params", {})
            self.conformal_q_alpha = config.get("conf_q_alpha")
            self.saved_files_dir = os.path.expanduser(
                config.get("saved_files_dir")
            )
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
    ):
        if targets is None:
            exclude_cols = list(exclude_columns or [])
            targets = [c for c in df.columns if c not in exclude_cols]

        self.target_names = targets

        X, Y, scaler_x, scaler_y = apply_sliding_window(
            df,
            targets,
            n_future,
            n_past,
            stride,
            exclude_columns,
        )

        # Ensure Y is strictly 3D: (N, n_targets, n_future)
        if Y.ndim == 2:
            Y = Y[:, None, :]

        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        n_samples = len(X)
        train_end = int(splits[0] * n_samples)
        val_end = int((splits[0] + splits[1]) * n_samples)

        x_train, y_train = X[:train_end], Y[:train_end]
        self.clipping_min = np.quantile(
            x_train, 0.01, axis=(0, 2), keepdims=True
        )
        self.clipping_max = np.quantile(
            x_train, 0.99, axis=(0, 2), keepdims=True
        )

        x_val, y_val = np.clip(
            X[train_end:val_end], self.clipping_min, self.clipping_max
        ), Y[train_end:val_end]
        x_test, y_test = np.clip(
            X[val_end:], self.clipping_min, self.clipping_max
        ), Y[val_end:]

        train_ds = TSDatasets(x_train, y_train)
        val_ds = TSDatasets(x_val, y_val)
        test_ds = TSDatasets(x_test, y_test)

        dls = TSDataLoaders.from_dsets(train_ds, val_ds, bs=bs)
        test_dl = TSDataLoader(test_ds, bs=bs)

        return dls, test_dl
    
    def train(self, dls, test_dl):
        patience = self.params.get("patience")
        epochs = self.params.get("epochs")
        lr = self.params.get("lr")

        xb, yb = dls.one_batch()
        d_model = self.params.get("d_model")
        d_ffn = self.params.get("d_ffn")
        depth = self.params.get("depth")
        patch_size = 1

        n_targets = yb.shape[1]
        n_future = yb.shape[2]
        
        # 1. Total output size for all targets over all forecast steps
        c_out = n_targets * n_future

        base_model = gMLP(
            c_in=xb.shape[1],
            c_out=c_out,
            seq_len=xb.shape[2],
            d_model=d_model,
            d_ffn=d_ffn,
            depth=depth,
            patch_size=patch_size,
        )

        model = ReshapedgMLP(base_model, n_targets, n_future)

        learn = Learner(
            dls,
            model,
            loss_func=HuberLoss(delta=1.0),
            metrics=rmse,
            cbs=[
                EarlyStoppingCallback(monitor="valid_loss", patience=patience)
            ],
        )

        learn.fit_one_cycle(epochs, lr, wd=0.01)

        # 1. Predictions & Evaluation
        preds, targets = learn.get_preds(dl=test_dl)

        preds_flat = (
            preds.numpy().transpose(0, 2, 1).reshape(-1, n_targets)
        )
        targets_flat = (
            targets.numpy().transpose(0, 2, 1).reshape(-1, n_targets)
        )

        preds = self.scaler_y.inverse_transform(preds_flat)
        targets = self.scaler_y.inverse_transform(targets_flat)
        rmse_test = sqrt(mean_squared_error(preds, targets))

        # 2. Conformal prediction intervals
        val_preds, val_targets = learn.get_preds(dl=dls.valid)

        val_preds_flat = (
            val_preds.numpy().transpose(0, 2, 1).reshape(-1, n_targets)
        )
        val_targets_flat = (
            val_targets.numpy().transpose(0, 2, 1).reshape(-1, n_targets)
        )

        val_preds_orig = self.scaler_y.inverse_transform(val_preds_flat)
        val_targets_orig = self.scaler_y.inverse_transform(val_targets_flat)

        val_residuals = np.abs(val_targets_orig - val_preds_orig)

        alpha = self.conformal_q_alpha
        n_val = len(val_residuals)

        q_level = min(np.ceil((n_val + 1) * (1 - alpha)) / n_val, 1.0)
        self.conformal_q = np.quantile(val_residuals, q_level, axis=0)

        # 3. Save artifacts and set model attribute
        self.model = model
        atomic_save(
            self.scaler_x, f"{self.saved_files_dir}/scaler_x_gmlp.pkl"
        )
        atomic_save(
            self.scaler_y, f"{self.saved_files_dir}/scaler_y_gmlp.pkl"
        )
        atomic_save(
            self.model,
            f"{self.saved_files_dir}/trained_model_gmlp.pth",
            use_pytorch=True,
        )
        atomic_save(
            self.clipping_min, f"{self.saved_files_dir}/clipping_min_gmlp.pkl"
        )
        atomic_save(
            self.clipping_max, f"{self.saved_files_dir}/clipping_max_gmlp.pkl"
        )
        atomic_save(
            self.conformal_q, f"{self.saved_files_dir}/conformal_q_gmlp.pkl"
        )
        atomic_save(
            self.target_names, f"{self.saved_files_dir}/target_names_gmlp.pkl"
        )

        return model, rmse_test, self.conformal_q

    def preprocess_inference(
        self, df, targets=None, n_past=288, exclude_columns=None
        ):
        exclude_cols = list(exclude_columns or [])
        
        # 1. Drop excluded/metadata columns (e.g. 'ts')
        df_full = df.drop(columns=exclude_cols, errors="ignore")

        # 2. Align DataFrame columns with scaler_x.feature_names_in_ if available
        if hasattr(self.scaler_x, "feature_names_in_"):
            df_full = df_full.reindex(columns=self.scaler_x.feature_names_in_, fill_value=0.0)

        # 3. Transform the entire DataFrame using scaler_x (since it was fitted on all columns during training)
        X = self.scaler_x.transform(df_full)

        # 4. Extract sliding window slice & convert to (1, c_in, n_past) shape expected by TS DataLoader / PyTorch model
        window = X[-n_past:]
        window = window.T[None, ...]
        
        # 5. Clip features using training set quantiles
        window = np.clip(
            window, self.clipping_min, self.clipping_max
        ).astype(np.float32)
    
        return window

    def infer(self, window, device="cpu"):
        window = torch.tensor(window, dtype=torch.float32).to(device)

        if self.model is None:
            raise RuntimeError(
                "model was not trained yet, please run the train method first."
            )
        if not self.target_names:
            raise RuntimeError(
                "target_names not set, please train the model first."
            )
        model = self.model.to(device)
        model.eval()

        with torch.no_grad():
            preds = model(window)  # Shape: (1, n_targets, n_future)

        preds = preds.cpu().numpy()[0].T  # Shape: (n_future, n_targets)
        preds = self.scaler_y.inverse_transform(preds)
        preds_dict = {
            metric: preds[:, i] for i, metric in enumerate(self.target_names)
        }
        #preds = np.clip(preds, a_min=0, a_max=None)

        return preds_dict