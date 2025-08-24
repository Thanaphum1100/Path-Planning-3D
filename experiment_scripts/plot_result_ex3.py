import os
import json
import pandas as pd

# Constants
METHOD_FOLDERS = ["ours", "catnip"]
METHOD_NAMES = ["This work", "CATNIP"]
EXP_NAMES = ["stonehenge", "statues", "flightroom"]
COMPUTATION_TIME_KEYS = ["avg_catnips_time", "avg_astar_time", "avg_post_time", "avg_smoothing_time"]
PATH_KEYS = ["avg_traj", "avg_traj_ori"]

# Initialize storage for results
table_data = {}

# Iterate over each computation time key
for key in COMPUTATION_TIME_KEYS:
    row = []
    for method in METHOD_FOLDERS:
        for exp in EXP_NAMES:
            file_path = f"./experiment/experiment3/{method}/{exp}/data_average.json"
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    value = data.get(key, 0)
            except FileNotFoundError:
                value = 0
            row.append(value)
    table_data[key] = row

# Iterate over each path key
for key in PATH_KEYS:
    row = []
    for method in METHOD_FOLDERS:
        for exp in EXP_NAMES:
            file_path = f"./experiment/experiment3/{method}/{exp}/data_average.json"
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    values = data.get(key, [])
                    if isinstance(values, list) and len(values) > 0:
                        avg_value = sum(values) / len(values)
                    else:
                        avg_value = 0
            except FileNotFoundError:
                avg_value = 0
            row.append(avg_value)
    table_data[key] = row

# Create column headers
columns = [f"{method} - {exp}" for method in METHOD_NAMES for exp in EXP_NAMES]

# Create DataFrame
df = pd.DataFrame(table_data, index=columns).T

# Display the table
print(df)

# Optional: Save to CSV
df.to_csv("experiment3_table.csv")
