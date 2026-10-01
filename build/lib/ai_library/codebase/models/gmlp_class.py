from tsai.data.core import TSDatasets, TSDataLoaders, TSDataLoader
from tsai.models.gMLP import gMLP
from tsai.learner import Learner
from fastai.callback.tracker import EarlyStoppingCallback
from torch.nn import MSELoss, HuberLoss
from fastai.metrics import rmse
from math import sqrt
from sklearn.metrics import mean_squared_error
from ..helpers.sliding_window import apply_sliding_window
import numpy as np
import torch
import yaml
from ..helpers.to_saved_files import atomic_save
import os
from pathlib import Path
import matplotlib.pyplot as plt

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
            ):
            """
            Create all sliding windows first and then split
            the resulting samples into train/val/test.
            """
            #print(df.head())
            #print(f"Targets: {targets}")
            X, Y, scaler_x, scaler_y = apply_sliding_window(
                df,
                targets,
                n_future,
                n_past,
                stride,
                exclude_columns,
            )
            #if Y.ndim == 2:
            #    Y = Y[:, None, :]

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

            train_ds = TSDatasets(x_train, y_train)
            val_ds = TSDatasets(x_val, y_val)
            test_ds = TSDatasets(x_test, y_test)

            dls = TSDataLoaders.from_dsets(
                train_ds,
                val_ds,
                bs=bs
            )

            test_dl = TSDataLoader(test_ds, bs=bs)

            return dls, test_dl
        '''
        def train(self, dls, test_dl):

            patience = self.params.get("patience")
            epochs = self.params.get("epochs")
            lr = self.params.get("lr")

            xb, yb = dls.one_batch()
            #print("X batch sample data:\n", xb[:2])
            #print("Y batch sample data:\n", yb[:2])
            d_model = self.params.get("d_model")
            d_ffn = self.params.get("d_ffn")
            depth = self.params.get("depth")
            patch_size = self.params.get("patch_size")
            model = gMLP(
                c_in=xb.shape[1],
                c_out=yb.shape[1],
                seq_len=xb.shape[2],
                d_model=d_model,
                d_ffn=d_ffn,
                depth=depth,
                patch_size=patch_size,
            )

            learn = Learner(
                    dls,
                    model,
                    loss_func=HuberLoss(delta=1.0),
                    metrics=rmse,
                    cbs=[EarlyStoppingCallback(monitor="valid_loss", patience=patience)],
                )

            learn.fit_one_cycle(epochs, lr, wd=0.01)
            preds, targets = learn.get_preds(dl=test_dl)
            #preds = self.scaler_y.inverse_transform(preds)
            #targets = self.scaler_y.inverse_transform(targets)
            # 1. Convert PyTorch Tensors to NumPy and flatten to (-1, 1)
            preds_flat = preds.numpy().reshape(-1, 1)
            targets_flat = targets.numpy().reshape(-1, 1)

            # 2. Inverse transform with 2D shape matching scaler expectations
            preds = self.scaler_y.inverse_transform(preds_flat)
            targets = self.scaler_y.inverse_transform(targets_flat)
            rmse_test = sqrt(mean_squared_error(preds, targets))

            #conformal prediction intervals
            val_preds, val_targets = learn.get_preds(dl=dls.valid)
            val_preds = self.scaler_y.inverse_transform(val_preds).reshape(-1, 1)
            val_targets = self.scaler_y.inverse_transform(val_targets).reshape(-1, 1)
            val_residuals = np.abs(val_targets - val_preds)
            alpha = 0.05  
            n_val = len(val_residuals)

            # 3. Finite-sample correction formula for Conformal Prediction
            q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val
            q_level = min(q_level, 1.0)  # Safety cap at 1.0

            # 4. Compute the quantile threshold
            # axis=0 calculates a distinct margin for each future timestep in your horizon
            self.conformal_q = np.quantile(val_residuals, q_level, axis=0)
            ####plotting (for debugging)
            
            plt.figure(figsize=(8, 3))
            plt.plot(targets.reshape(-1,1), label="target")
            plt.plot(preds.reshape(-1,1), label="preds", alpha=0.7)
            plt.legend()
            plt.xlabel("forecast step (aligned)")
            plt.ylabel("cpu_01_busy")
            plt.tight_layout()
            plt.savefig(f"{self.saved_files_dir}/predictions_gmlp.png")
            plt.close()

            self.model = model
            atomic_save(self.scaler_x, f"{self.saved_files_dir}/scaler_x_gmlp.pkl")  
            atomic_save(self.scaler_y, f"{self.saved_files_dir}/scaler_y_gmlp.pkl")
            atomic_save(self.model, f"{self.saved_files_dir}/trained_model_gmlp.pth", use_pytorch=True)
            atomic_save(self.clipping_min, f"{self.saved_files_dir}/clipping_min_gmlp.pkl")
            atomic_save(self.clipping_max, f"{self.saved_files_dir}/clipping_max_gmlp.pkl")
            atomic_save(self.conformal_q, f"{self.saved_files_dir}/conformal_q_gmlp.pkl")
            return model, rmse_test, self.conformal_q
        '''
        def train(self, dls, test_dl):

            patience = self.params.get("patience")
            epochs = self.params.get("epochs")
            lr = self.params.get("lr")

            xb, yb = dls.one_batch()
            d_model = self.params.get("d_model")
            d_ffn = self.params.get("d_ffn")
            depth = self.params.get("depth")
            
            # Force patch_size=1 to eliminate high-frequency aliasing noise
            patch_size = 1

            # 1. Instantiate base gMLP model
            # yb shape is (batch_size, n_future) -> c_out must equal n_future (e.g., 96)
            c_out = yb.shape[1] if yb.ndim == 2 else yb.shape[2]
            
            model = gMLP(
                c_in=xb.shape[1],
                c_out=c_out,
                seq_len=xb.shape[2],
                d_model=d_model,
                d_ffn=d_ffn,
                depth=depth,
                patch_size=patch_size,
            )

            # 2. Setup Learner with HuberLoss and Weight Decay
            learn = Learner(
                dls,
                model,
                loss_func=HuberLoss(delta=1.0),
                metrics=rmse,
                cbs=[EarlyStoppingCallback(monitor="valid_loss", patience=patience)],
            )

            # 3. Train with weight decay
            learn.fit_one_cycle(epochs, lr, wd=0.01)

            # 4. Predictions & Evaluation
            preds, targets = learn.get_preds(dl=test_dl)

            preds_flat = preds.numpy().reshape(-1, 1)
            targets_flat = targets.numpy().reshape(-1, 1)

            # Inverse transform
            preds = self.scaler_y.inverse_transform(preds_flat)
            targets = self.scaler_y.inverse_transform(targets_flat)

            # Clip negative predictions at 0
            preds = np.clip(preds, a_min=0, a_max=None)

            rmse_test = sqrt(mean_squared_error(preds, targets))

            # Conformal prediction intervals
            val_preds, val_targets = learn.get_preds(dl=dls.valid)
            val_preds = np.clip(self.scaler_y.inverse_transform(val_preds.numpy().reshape(-1, 1)), a_min=0, a_max=None)
            val_targets = self.scaler_y.inverse_transform(val_targets.numpy().reshape(-1, 1))
            val_residuals = np.abs(val_targets - val_preds)
            alpha = 0.05  
            n_val = len(val_residuals)

            q_level = min(np.ceil((n_val + 1) * (1 - alpha)) / n_val, 1.0)
            self.conformal_q = np.quantile(val_residuals, q_level, axis=0)

            # Plotting
            plt.figure(figsize=(8, 3))
            plt.plot(targets.reshape(-1,1), label="target")
            plt.plot(preds.reshape(-1,1), label="preds", alpha=0.7)
            plt.legend()
            plt.xlabel("forecast step (aligned)")
            plt.ylabel("cpu_01_busy")
            plt.tight_layout()
            plt.savefig(f"{self.saved_files_dir}/predictions_gmlp.png")
            plt.close()

            self.model = model
            atomic_save(self.scaler_x, f"{self.saved_files_dir}/scaler_x_gmlp.pkl")  
            atomic_save(self.scaler_y, f"{self.saved_files_dir}/scaler_y_gmlp.pkl")
            atomic_save(self.model, f"{self.saved_files_dir}/trained_model_gmlp.pth", use_pytorch=True)
            atomic_save(self.clipping_min, f"{self.saved_files_dir}/clipping_min_gmlp.pkl")
            atomic_save(self.clipping_max, f"{self.saved_files_dir}/clipping_max_gmlp.pkl")
            atomic_save(self.conformal_q, f"{self.saved_files_dir}/conformal_q_gmlp.pkl")

            return model, rmse_test, self.conformal_q
        
        def preprocess_inference(self, df, targets, n_past, exclude_columns=None):

            exclude_columns = list(exclude_columns or [])
            target_cols = targets if isinstance(targets, (list, tuple)) else [targets]

            # 1. keep all required columns (features + targets)
            df_full = df.drop(columns=exclude_columns)

            # 2. scale features ONLY
            feature_df = df_full.drop(columns=target_cols)
            X_feat = self.scaler_x.transform(feature_df)

            # 3. scale targets separately
            Y = self.scaler_y.transform(df_full[target_cols])

            # 4. recombine in correct column order (VERY important)
            X = np.concatenate([X_feat, Y], axis=1)

            # 6. window
            window = X[-n_past:]
            window = window.T[None, ...]
            window = np.clip(window, self.clipping_min, self.clipping_max).astype(np.float32)
            return window

        def infer(self, window, device="cpu"):

            window = torch.tensor(window, dtype=torch.float32).to(device)

            if self.model is None:
                raise RuntimeError("model was not trained yet, please run the train method first.")
            
            model = self.model.to(device)
            model.eval()

            with torch.no_grad():
                preds = model(window)

            preds = preds.cpu().numpy().reshape(-1,1)
            preds = self.scaler_y.inverse_transform(preds)

            return preds
