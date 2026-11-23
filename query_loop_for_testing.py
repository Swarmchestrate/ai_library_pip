import random
from ai_library import ingest
from ai_library import train,infer, show_config, update_config
import numpy as np

paths = ["~/.local/share/ai_library/data_synth.parquet", "~/.local/share/ai_library/inference_data_synth.parquet"]
import os
for path in paths:
    file_path = os.path.expanduser(path)
    if os.path.exists(file_path):
        os.remove(file_path)

update_config(None, {
    "gmlp.model_params.epochs": 500,          # Sets prediction_target to None properly
    "itransformer.model_params.epochs": 500,
    "xgb.model_params.n_estimators": 200,
    "xgb.model_params.max_depth": 10
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
print(apply_adwin())

for i in range(30):
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
print(apply_adwin())


for i in range(30):
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

predictions = infer()
print(apply_adwin())


for i in range(30):
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


for i in range(30):
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
