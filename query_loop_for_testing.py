import random
from ai_library import ingest

def generate_comp_data(num_points=5, start_timestamp=1716712345.12, interval=15.0):
    """
    Generiert eine Liste von Datenpunkten mit 15-Sekunden-Intervallen 
    und zufälligen Werten.
    """
    data = []
    current_time = start_timestamp
    
    for _ in range(num_points):
        data.append({
            "timestamp": round(current_time, 2),
            "value": round(random.uniform(0.0, 100.0), 2)  # Zufälliger Wert zwischen 0 und 100
        })
        current_time += interval
        
    return data

def generate_raw_data(ip_addresses, num_points=5, start_timestamp=1716712345.12, interval=15.0):
    """
    Generates a dictionary mapping IP addresses to a list of data points 
    with timestamps and random values.
    """
    raw_data = {}
    
    for ip in ip_addresses:
        data = []
        current_time = start_timestamp
        
        for _ in range(num_points):
            data.append({
                "timestamp": round(current_time, 2),
                "value": round(random.uniform(0.0, 100.0), 2)  # Random value between 0 and 100
            })
            current_time += interval
            
        raw_data[ip] = data
        
    return raw_data
'''
ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
start_timestamp = 1716712345.12  # Startzeitpunkt in Sekunden seit der Epoche
while True:
    num_points = random.randint(1, 10)  # Zufällige Anzahl von Datenpunkten zwischen 1 und 10
    comp_data = generate_comp_data(num_points=num_points, start_timestamp=start_timestamp, interval=15.0)
    raw_data = generate_raw_data(
        ip_addresses=ips, 
        num_points=num_points, 
        start_timestamp=start_timestamp, 
        interval=15.0
    )
    start_timestamp = max([raw_data[ip][-1]["timestamp"] for ip in ips]) + 15.0  # Aktualisiere den Startzeitpunkt für die nächste Iteration
    injest_metrics("cpu_usage", raw_data, mode="raw")
    injest_metrics("cpu_usage", comp_data, mode="composite")
'''

def generate_synchronized_data(ip_addresses, num_points=5, start_timestamp=1716712345.12, interval=15.0):
    """
    Generates raw and composite data using the exact same timestamp sequence 
    to eliminate drift.
    """
    timestamps = [round(start_timestamp + (i * interval), 2) for i in range(num_points)]
    
    # Generate raw data for each IP
    raw_data = {}
    for ip in ip_addresses:
        raw_data[ip] = [
            {"timestamp": ts, "value": round(random.uniform(0.0, 100.0), 2)}
            for ts in timestamps
        ]
        
    # Generate composite data synchronized to the same timestamps
    comp_data = [
        {"timestamp": ts, "value": round(random.uniform(50.0, 100.0), 2)}
        for ts in timestamps
    ]
    
    return raw_data, comp_data

# Simulation setup
ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
start_timestamp = 1716712345.12

# Run a controlled simulation batch (e.g., 5 iterations for testing)
for i in range(500):
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