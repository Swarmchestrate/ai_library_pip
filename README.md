# ai_library

`ai_library` is a Python package that exposes model training, inference, metrics recording, cron scheduling, and package-based configuration management.

## Configuration

See the [Configuration reference](docs/configuration.md) for descriptions
of all configuration variables, expected values, defaults, and SAT
integration recommendations.

## Package Overview

### Available Top-Level APIs

- `ai_library.validate_config(config_path=None)`
  - Loads and validates the package default `ai_library/config.yaml` if no path is provided.
  - Returns the parsed configuration dictionary.

- `ai_library.update_config(config_path=None, updates=...)`
  - Updates the package config file by default.
  - Supports a dictionary, a `[key, value]` pair, or a list of `[key, value]` pairs.

- `ai_library.show_config(config_path=None)`
  - Prints the current configuration to stdout.

- `ai_library.train()`
  - Loads configuration from the package config.
  - Reads data, builds the selected pipeline, and trains the model.

- `ai_library.infer()`
  - Loads package configuration.
  - Loads a saved model and performs inference.
  - returns predictions, first_predicted, last_predicted, data_frequency, conformal_q 
  - conformal_q is the safety margin in order to be 1-alpha accurate for the predicted time window

- `ai_library.add_to_cron()` / `ai_library.remove_from_cron()`
  - Manage cron scheduling for recurring training runs.
  - Adds or removes a cron job that runs `python3 -m ai_library.codebase.setup.train`.


- `ai_library.ingest()`
  - Ingest query data into the dateframe path which is defined in config

## How to Use

### Install
```bash
pip install ai-library-swch
```
A safe Python version is 3.12.3

### Configuration
This is an example of how config can be printed and updated.
```python
from ai_library import validate_config, update_config, show_config

update_config(None, updates=["parquet_train_size", 100000])
update_config(None, {
    "gmlp.model_params.epochs": 500,        
    "itransformer.model_params.epochs": 500,
    "xgb.model_params.n_estimators": 700,
    "xgb.model_params.max_depth": 10
})
show_config()
validate_config()
```

### Train and Infer
Import the functions:
```python
from ai_library import train, infer
```

# Manually use train (no cron)
The standard way is to use the cron functions to train in intervals. This might still be useful, for example when switching models.
```python
train()
```

### Cron Scheduling
Add_to_cron creates a cron job that runs the train function in intervals.
```python
import ai_library.codebase.setup.cron_manager as cron_manager

cron_manager.add_to_cron()
cron_manager.remove_from_cron()
```

# Inference
After a trained model is accessible via the path, that is given in the config file, inference can be applied:
```python
predictions = infer()
```
predictions shape:
```
{node_id_1: {metric_A: {predicted_values:[REAL VALUES],
            conformal_q: CONFORMAL_Q_VALUE}
            metric_B: {predicted_values:[REAL VALUES],
            conformal_q: CONFORMAL_Q_VALUE},
            ...
            },
{node_id_2: ...}, 
* goes on like this* 
first_predicted: FIRST_TIMESTAMP
last_predicted: LAST_TIMESTAMP
data_frequency: DATA_FREQUENCY OF PREDICTED HORIZON
            }
```

### Example data ingestion to build the train parquet file using monitoring query functions (here on simulated sine data)
```
import random
from ai_library import ingest

import math

def get_sine_value(timestamp, period=300.0, amplitude=50.0, offset=50.0):
    """Calculates a smooth sine wave value from a timestamp."""
    return round(amplitude * math.sin(2 * math.pi * timestamp / period) + offset, 2)

def generate_synchronized_data(ip_addresses, num_points=5, start_timestamp=1716712345.12, interval=15.0):
    """Generates raw and composite data using a timestamp-based sine wave."""
    timestamps = [round(start_timestamp + (i * interval), 2) for i in range(num_points)]
    
    # Generate raw data for each IP with slight phase shifts per IP
    raw_data = {}
    for idx, ip in enumerate(ip_addresses):
        phase_shift = idx * 10.0
        raw_data[ip] = [
            {"timestamp": ts, "value": get_sine_value(ts + phase_shift, period=1800, amplitude=40.0, offset=50.0)}
            for ts in timestamps
        ]
        
    # Generate composite data synchronized to the same timestamps
    comp_data = [
        {"timestamp": ts, "value": get_sine_value(ts, period=1800, amplitude=25.0, offset=75.0)}
        for ts in timestamps
    ]
    
    return raw_data, comp_data

# Simulation setup
ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
start_timestamp = 1716712345.12

# Run a controlled simulation batch (e.g., 5 iterations for testing)
for i in range(1000):
    num_points = random.randint(3, 8)
    
    # Generate synchronized data
    raw_data, comp_data = generate_synchronized_data(
        ip_addresses=ips, 
        num_points=num_points, 
        start_timestamp=start_timestamp, 
        interval=15.0
    )
    # Ingest metrics
    ingest("cpu_usage", raw_data, mode="raw")
    ingest("cpu_usage", comp_data, mode="composite")
    
    # Update start timestamp for the next batch cleanly
    start_timestamp = raw_data[ips[0]][-1]["timestamp"] + 15.0
```
### Example Pipeline
```
from ai_library import train,infer, show_config, update_config

update_config(None, {
    "gmlp.model_params.epochs": 500,          # Sets prediction_target to None properly
    "itransformer.model_params.epochs": 500,
    "xgb.model_params.n_estimators": 700,
    "xgb.model_params.max_depth": 10
})
update_config(None, {
    "pipeline_type": "itransformer"
})
show_config()
train()

preds_dict_itransformer = infer()

update_config(None, {
    "pipeline_type": "gmlp"
})
train()
preds_dict_gmlp = infer()

update_config(None, {
    "pipeline_type": "xgb"
})
train()
preds_dict_xgb = infer()

print(preds_dict_itransformer)

from ai_library import add_to_cron,remove_from_cron

add_to_cron() #add train to cron with retrain_frequency frequency
remove_from_cron() #remove train from crontab
```
# git link:
https://github.com/maxsotto1/ai_library_pip
