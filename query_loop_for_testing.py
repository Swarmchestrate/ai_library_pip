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