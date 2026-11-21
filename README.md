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
  - writes inferences to the inference dataframe in config

- `ai_library.ingest(metric_name, data, mode("raw" or "composite"))`
  - ingest query data into the training dataframe

- `ai_library.apply_adwin()`
  - checks for recent model drift using adaptive windowing
  - inference must have been run before (builds windows based on prediction history)
  - does **not**
    - benchmark challenger model & champion model
    - retrain (this should ideally be done after more time has passed to monitor the predictions for a while)
    - have a cooldown
  - smaller adwin_delta in config -> less sensitve to drift
  


- `ai_library.train_to_cron()` / `ai_library.remove_train_from_cron()`
  - Manage cron scheduling for recurring training runs.
  - Adds or removes a cron job that runs `python3 -m ai_library.codebase.setup.train`.

  - `ai_library.adwin_to_cron()` / `ai_library.remove_adwin_from_cron()`
  - Manage cron scheduling for recurring training runs.
  - Adds or removes a cron job that runs the apply_adwin  

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

# Adwin (model drift detector)
```python
from ai_libray import apply_adwin
apply_adwin()
```
- returns true, if model drift was found since last call
- returns false else

### Cron Scheduling
Add_to_cron creates a cron job that runs the train function in intervals.
```python
import ai_library.codebase.setup.cron_manager as cron_manager

cron_manager.train_to_cron()
cron_manager.remove_train_from_cron()
cron_manager.adwin_to_cron()
cron_manager.remove_adwin_from_cron()
```
- both functions run with config.train_frequeny as of now.

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

train_to_cron() #add train to cron with retrain_frequency frequency
remove_train_from_cron() #remove train from crontab
adwin_to_cron() #add adwin drift detector to cron
remove_adwin_from_cron() #remove adwin from con
```

### Example pipeline to test adwin functionality
```
import random
from ai_library import ingest
from ai_library import train,infer, show_config, update_config
import numpy as np

paths = ["~/.local/share/ai_library/data_synth.parquet", "~/.local/share/ai_library/inference_data_synth.parquet"] #update these to your config paths
import os
for path in paths:
    file_path = os.path.expanduser(path)
    if os.path.exists(file_path):
        os.remove(file_path)

update_config(None, {
    "itransformer.model_params.epochs": 500,
})
update_config(None, {
    "pipeline_type": "itransformer"
})
show_config()

import math

def get_sine_value(timestamp, period=300.0, amplitude=50.0, offset=50.0):
    """Calculates a smooth sine wave value from a timestamp."""
    return round(amplitude * math.sin(2 * math.pi * timestamp / period) + offset, 2)

def generate_synchronized_data(ip_addresses, num_points=5, start_timestamp=1716712345.12, interval=15.0, period=1800, amplitude=40.0, offset=50.0):
    """Generates raw and composite data using a timestamp-based sine wave."""
    timestamps = [round(start_timestamp + (i * interval), 2) for i in range(num_points)]
    
    # Generate raw data for each IP with slight phase shifts per IP
    raw_data = {}
    for idx, ip in enumerate(ip_addresses):
        phase_shift = idx * 10.0
        raw_data[ip] = [
            {"timestamp": ts, "value": get_sine_value(ts + phase_shift, period=period, amplitude=amplitude, offset=offset)}
            for ts in timestamps
        ]
        
    # Generate composite data synchronized to the same timestamps
    comp_data = [
        {"timestamp": ts, "value": get_sine_value(ts, period=period, amplitude=amplitude, offset=offset)}
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

from ai_library import apply_adwin
train()
predictions = infer()

for i in range(200):
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

predictions = infer()
print(apply_adwin()) #should be false, no data drift added

for i in range(200):
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

predictions = infer()
print(apply_adwin()) #should still be false, no data drift added


for i in range(200):
    num_points = random.randint(3, 8)
    
    # Generate synchronized data
    raw_data, comp_data = generate_synchronized_data(
        ip_addresses=ips, 
        num_points=num_points, 
        start_timestamp=start_timestamp, 
        interval=15.0,
        offset=40.0,
        amplitude=20.0,
        period=1700.0
    )
    # Ingest metrics
    ingest("cpu_usage", raw_data, mode="raw")
    ingest("cpu_usage", comp_data, mode="composite")
    
    # Update start timestamp for the next batch cleanly
    start_timestamp = raw_data[ips[0]][-1]["timestamp"] + 15.0

predictions = infer() #now it should become true
print(apply_adwin())


for i in range(200):
    num_points = random.randint(3, 8)
    
    # Generate synchronized data
    raw_data, comp_data = generate_synchronized_data(
        ip_addresses=ips, 
        num_points=num_points, 
        start_timestamp=start_timestamp, 
        interval=12.0,
        offset=30.0,
        amplitude=20.0,
        period=1500.0
    )
    # Ingest metrics
    ingest("cpu_usage", raw_data, mode="raw")
    ingest("cpu_usage", comp_data, mode="composite")
    
    # Update start timestamp for the next batch cleanly
    start_timestamp = raw_data[ips[0]][-1]["timestamp"] + 15.0

predictions = infer()
print(apply_adwin())


for i in range(200):
    num_points = random.randint(3, 8)
    
    # Generate synchronized data
    raw_data, comp_data = generate_synchronized_data(
        ip_addresses=ips, 
        num_points=num_points, 
        start_timestamp=start_timestamp, 
        interval=12.0,
        offset=20.0,
        amplitude=80.0,
        period=1000.0
    )
    # Ingest metrics
    ingest("cpu_usage", raw_data, mode="raw")
    ingest("cpu_usage", comp_data, mode="composite")
    
    # Update start timestamp for the next batch cleanly
    start_timestamp = raw_data[ips[0]][-1]["timestamp"] + 15.0

predictions = infer()
print(apply_adwin())

```
# git link:
https://github.com/maxsotto1/ai_library_pip
