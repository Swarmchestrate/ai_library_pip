## Configuration reference (v0.1.3)

We will be updating this file according to the changes in the library.

This reference covers every active leaf setting in `ai_library/config.yaml` at the last commit. Values in the **Shipped value** column come from that file; they are not necessarily fallback values when a key is omitted. Dot notation identifies nested YAML keys; preserve the nesting in `config.yaml`.

**Value guidance:** Type/validity constraints are distinguished from illustrative tuning ranges. These ranges are engineering starting points, not benchmark results or promises of accuracy. Tune against the application's sampling cadence, history, forecast duration, and resource budget.

**SAT recommendation:** This is a proposed integration policy. **Yes** means application information should be supplied or mapped from SAT; **Optional** means a default should work without a SAT field; **derive** means calculate the config value from higher-level SAT information; **No** means keep the setting under library or deployment control. “No” does not mean remove it from `config.yaml`.

### Application, data, and runtime settings

| Variable | Shipped value | Meaning / actual behavior | Expected values / rough range | Include in SAT? |
| --- | --- | --- | --- | --- |
| `pipeline_type` | `xgb` | Forecasting pipeline to train and load for inference. | String: `xgb`, `gmlp`, or `itransformer`. | Optional — expert override; library should select a default. |
| `window` | `288` | Number of past resampled time steps supplied to the model. | Positive integer; illustrative starting range 48–672 steps. At 30s, 288 steps represent 144 minutes. | Optional/derive — SAT may specify a lookback duration; convert to steps. |
| `horizon` | `96` | Number of future resampled time steps predicted per call. | Positive integer; illustrative starting range 1–288 steps. At 30s, 96 steps represent 48 minutes. | Yes/derive — application forecast duration is a requirement; convert to steps. |
| `stride` | `96` | Step between successive training-window origins; does not schedule inference. | Positive integer; commonly 1–horizon. Smaller values create more, overlapping samples; values above horizon are allowed. | No — training/data-efficiency setting. |
| `prediction_target` | `cpu_util_instance` | Metric column(s) to forecast after pivoting and preprocessing. | Metric-name string or nonempty list of strings; names must exist and be numeric. Prefer explicit names; see null limitation below. | Yes — identifies the application quantities to forecast. |
| `STANDARD_METRICS` | `['cpu_util_instance', 'mean_cpu_util_prct', 'cpu_slo_upper', 'cpu_slo_lower']` | Metrics subscribed to by the standard recorder and selected when reading recorded data. | List of valid monitoring metric names; normally nonempty. Include targets and useful input metrics. | Yes/derive — map SAT monitoring declarations to this list. |
| `cols_to_drop` | `['ts']` | Columns excluded from model inputs, such as timestamp metadata. | List of existing column names; usually `["ts"]`. Do not include prediction targets. Keep numeric feature columns. | No — derive from the data schema; expert override only. |
| `splits` | `[0.7, 0.1, 0.2]` | Chronological train/validation/test proportions. Validation also supplies residuals for interval calibration. | Three numbers strictly between 0 and 1; use sum = 1. Typical train 0.6–0.8 and validation/test 0.1–0.2 each, jointly summing to 1. | No — model evaluation policy. |
| `retrain_frequency` | `30m` | Schedule used when installing the training cron job; not an inference interval. | For current cron code use minute strings, preferably `1m`, `2m`, `3m`, `5m`, `10m`, `15m`, `20m`, or `30m`; see scheduler limitations. | Optional — application adaptation requirement, subject to platform training budget. |
| `data_frequency` | `30s` | Time-bin width used to average/resample recorded metrics before forecasting. | Positive duration string with integer and `s` or `m`, e.g. `30s`, `1m`, `5m`; illustrative range 5s–15m. | Yes — controls forecast resolution and conversion of durations to steps. |
| `parquet_path` | `~/.local/share/ai_library/data.parquet` | Intermediate consolidated Parquet file written by the reader and read by training/inference. | Writable file path with a parent directory; use an application-specific path. `~` is used in the shipped config. | No — resolve from deployment storage configuration. |
| `data_dir` | `~/.local/share/ai_library/collector` | Directory containing the recorder’s partitioned Parquet dataset. | Writable directory path; reader and recorder must use the same directory. | No — deployment storage setting. |
| `saved_files_dir` | `~/.local/share/ai_library/saved_models` | Directory for trained models, scalers, clipping thresholds, and calibration margins. | Writable directory path; isolate by application/model configuration to prevent overwrites. | No — deployment storage setting. |
| `poll_interval_seconds` | `5.0` | Delay between recorder polls of subscribed metrics. | Positive number in seconds; illustrative 1–60s. Usually no greater than data_frequency in seconds; align with producer cadence. | Optional/derive — use SAT monitoring cadence, with a platform default. |
| `flush_max_rows` | `5000` | Buffered recorder row threshold that triggers a disk flush. Counts metric records, not resampled time steps. | Positive integer; illustrative 1,000–50,000. Flush occurs when either row or elapsed-time threshold is met. | No — memory/I/O tuning. |
| `flush_max_seconds` | `60.0` | Elapsed-time threshold for flushing a nonempty recorder buffer. | Positive number in seconds; illustrative 5–300s. Checked during polling, so not an exact deadline. | No — operational tuning; derive from freshness requirements if needed. |
| `parquet_train_size` | `100000` | Maximum number of latest resampled rows retained for a training run; does not limit recorder disk retention. | Positive integer or `null` for all rows. Illustrative 10,000–100,000+, subject to memory and enough windows in each split. | Optional/derive — expose training-history duration if application-specific; platform caps resource use. |
| `pivot_df` | `true` | Convert long metric/value records into a wide table with one column per metric. | YAML boolean `true` or `false`. Use `true` with the bundled recorder; `false` only for suitable already-wide data. | No — internal data-format setting. |
| `conf_q_alpha` | `0.05` | Nominal miscoverage used to compute an absolute-residual calibration margin; smaller values generally give wider intervals. | Number strictly between 0 and 1; illustrative 0.01–0.20. 0.05 corresponds to nominal 95% coverage. Ignored by iTransformer in this release. | Optional — application uncertainty requirement; fix iTransformer wiring before exposing consistently. |

### gMLP settings

| Variable | Shipped value | Meaning / actual behavior | Expected values / rough range | Include in SAT? |
| --- | --- | --- | --- | --- |
| `gmlp.model_params.d_model` | `64` | gMLP hidden embedding width. | Positive integer; illustrative 32–256. Larger widths require more resources. | No — ML tuning. |
| `gmlp.model_params.d_ffn` | `128` | gMLP feed-forward width. | Positive integer; illustrative 64–1024, often 2–4 times d_model; prefer even values. | No — ML tuning. |
| `gmlp.model_params.depth` | `2` | Number of gMLP blocks. | Positive integer; illustrative 1–6. | No — ML tuning. |
| `gmlp.model_params.patch_size` | `1` | Intended temporal patch length, but the training code hardcodes 1. | Validator accepts positive integers; effective value is always 1 in v0.1.3. | No — currently ineffective; fix implementation before offering overrides. |
| `gmlp.model_params.epochs` | `1000` | Maximum gMLP training epochs, subject to early stopping. | Positive integer; illustrative 50–1000. | No — training budget/tuning. |
| `gmlp.model_params.lr` | `0.001` | Maximum learning rate passed to one-cycle gMLP training. | Positive number; illustrative 0.0001–0.01. | No — ML tuning. |
| `gmlp.model_params.patience` | `200` | Epochs without validation-loss improvement before early stopping. | Use a positive integer; illustrative 10–200, normally below epochs. Validator also accepts positive non-integers. | No — ML tuning. |

### XGBoost settings

| Variable | Shipped value | Meaning / actual behavior | Expected values / rough range | Include in SAT? |
| --- | --- | --- | --- | --- |
| `xgb.model_params.n_estimators` | `100` | Number of boosting rounds. No early stopping is configured by this wrapper. | Positive integer; illustrative 100–1000. | No — ML tuning. |
| `xgb.model_params.max_depth` | `4` | Maximum tree depth. | Library requires a positive integer; illustrative 2–10. | No — ML tuning. |
| `xgb.model_params.learning_rate` | `0.01` | Contribution of each boosting round. | Use a number > 0 and ≤ 1; illustrative 0.01–0.3. | No — ML tuning. |
| `xgb.model_params.subsample` | `0.8` | Fraction of training samples sampled per boosting round. | Number > 0 and ≤ 1; illustrative 0.5–1. | No — ML tuning. |
| `xgb.model_params.colsample_bytree` | `0.8` | Fraction of input columns sampled for each tree. | Number > 0 and ≤ 1; illustrative 0.5–1. | No — ML tuning. |
| `xgb.model_params.min_child_weight` | `3` | Minimum summed Hessian weight required in a child node; constrains small leaves. | Library requires a positive integer; illustrative 1–10. | No — ML tuning. |
| `xgb.model_params.gamma` | `0.1` | Minimum loss reduction required to split a leaf. | Nonnegative number; illustrative 0–5. | No — ML tuning. |
| `xgb.model_params.reg_lambda` | `1.0` | L2 penalty on leaf weights. | Nonnegative number; illustrative 0–10. | No — ML tuning. |
| `xgb.model_params.reg_alpha` | `0.0` | L1 penalty on leaf weights. | Nonnegative number; illustrative 0–10. | No — ML tuning. |
| `xgb.model_params.tree_method` | `hist` | Algorithm used to construct trees. | Keep `hist` as baseline; XGBoost also documents `auto`, `approx`, and `exact`, subject to model compatibility. | No — implementation/resource choice. |
| `xgb.model_params.predictor` | `cpu_predictor` | Legacy XGBoost prediction-backend setting passed through by the wrapper. | Shipped value `cpu_predictor`; do not rely on it for device selection with pinned XGBoost 3.2.0. Modern device control uses `device`. | No — deployment concern; review/remove legacy parameter. |
| `xgb.model_params.random_state` | `42` | Random seed for reproducibility. | Nonnegative integer; e.g. 0–2,147,483,647. Keep fixed for comparisons. | No — reproducibility policy. |
| `xgb.model_params.n_jobs` | `-1` | XGBoost worker-thread setting. | Integer; commonly -1 for all available threads or 1 up to the allocated CPU count. | No/derive — platform should map CPU allocation to an explicit limit. |
| `xgb.model_params.objective` | `reg:squarederror` | Training loss/objective passed to XGBoost. | Keep `reg:squarederror` for this continuous, multi-output forecasting pipeline; other objectives require compatibility checks. | No — ML implementation choice. |

### iTransformer settings

| Variable | Shipped value | Meaning / actual behavior | Expected values / rough range | Include in SAT? |
| --- | --- | --- | --- | --- |
| `itransformer.model_params.epochs` | `300` | Maximum iTransformer training epochs, subject to early stopping. | Positive integer; illustrative 50–500. | No — training budget/tuning. |
| `itransformer.model_params.lr` | `0.001` | Maximum learning rate passed to one-cycle iTransformer training. | Use a positive number; illustrative 0.0001–0.01. Current validator fails to check this key correctly. | No — ML tuning. |
| `itransformer.model_params.patience` | `200` | Epochs without validation-loss improvement before early stopping. | Positive integer; illustrative 10–200, normally below epochs. | No — ML tuning. |

### SAT mapping and integration notes

A compact SAT contract should describe the forecast target(s), available monitoring metrics, sampling/resampling interval, and required forecast duration. Optionally describe lookback duration, retained training-history duration, retraining cadence, and nominal coverage. I think we should avoid making application authors choose dozens of ML hyperparameters.

For an interval of `Δ` seconds, map `horizon = forecast_duration_seconds / Δ` and, when supplied, `window = lookback_duration_seconds / Δ`. Require exact positive integer ratios or document an explicit rounding rule. Map requested coverage `c` to `conf_q_alpha = 1 - c`. For example, with `Δ = 30`, a 48-minute forecast produces `horizon = 96`, and a 144-minute lookback produces `window = 288`. These are durations represented by the steps; the timestamp span from the first to the last of H predictions is `(H - 1) × Δ`.

The existing `update_config` accepts dotted keys but only updates keys that already exist. `train()` and `infer()` load the package config rather than accepting an arbitrary config path. A SAT adapter and per-application configuration isolation therefore need explicit integration work. Configure before starting processes: recorder/reader settings are cached at module import. Changing model shape, target names, or input schema requires compatible retraining before inference.
