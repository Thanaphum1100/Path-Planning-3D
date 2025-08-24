import json
import os
import math

# Ours
TIME_KEYS = [
    "catnips_time", "total_astar_time", "total_post_time", "total_smoothing_time"
]

AVG_TIME_KEYS = [
    "avg_catnips_time", "avg_astar_time", "avg_post_time", "avg_smoothing_time"
]

# # Catnip
# TIME_KEYS = [
#     "catnips_time", "total_astar_time", "total_smoothing_time"
# ]

# AVG_TIME_KEYS = [
#     "avg_catnips_time", "avg_astar_time", "avg_smoothing_time"
# ]

PATH_KEYS = [
    "traj", "traj_ori"
]

AVG_PATH_KEYS = [
    "avg_traj", "avg_traj_ori"
]


def calculate_path_length(path):
    """Calculates the total length of a path given a list of points."""
    total_length = 0
    for i in range(1, len(path)):
        x1, y1, z1 = path[i-1]
        x2, y2, z2 = path[i]
        # Calculate Euclidean distance between consecutive points
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        total_length += distance
    return total_length


def read_experiment_data(data_path):
    """Reads experiment data from JSON files in the given folder."""

    with open(data_path, 'r') as f:
        meta = json.load(f)
    
    # Ensure 'number_path' is not zero to avoid division by zero errors
    if meta.get("number_path", 0) == 0:
        raise ValueError(f"Invalid 'number_path' in {data_path}")

    # Calculate average times and return them
    avg_data_dict = {avg_key: meta[key] / meta["number_path"] for key, avg_key in zip(TIME_KEYS, AVG_TIME_KEYS)}

    # Calculate path lengths and turning points
    for path_key, avg_length_key in zip(PATH_KEYS, AVG_PATH_KEYS):
        paths = meta[path_key]
        avg_data_dict[avg_length_key] = [calculate_path_length(path) for path in paths]

    return avg_data_dict


def calculate_average_data(folder_path, exp_name, experiment_number=10):
    """Calculates the average times and path lengths across multiple experiments."""

    # Initialize the dictionary for averaging 
    avg_data_dict = {key: 0 for key in AVG_TIME_KEYS}
    avg_data_dict.update({key: [] for key in AVG_PATH_KEYS})  # For path lengths

    for i in range(experiment_number):
        data_path = os.path.join(folder_path, f"{exp_name}/data_seed1_exnum{i+1}.json")
        cur_data_dict = read_experiment_data(data_path)

        # Accumulate time values
        for key in AVG_TIME_KEYS:
            avg_data_dict[key] += cur_data_dict.get(key, 0)

        # Accumulate path lengths element-wise
        for key in AVG_PATH_KEYS:
            path_lengths = cur_data_dict[key]
            
            # If first experiment, initialize list with correct size
            if not avg_data_dict[key]:  
                avg_data_dict[key] = [0] * len(path_lengths)

            # Sum corresponding path lengths
            for j in range(len(path_lengths)):
                avg_data_dict[key][j] += path_lengths[j]

    # Compute the final average values
    for key in AVG_TIME_KEYS:
        avg_data_dict[key] /= experiment_number

    for key in AVG_PATH_KEYS:
        avg_data_dict[key] = [val / experiment_number for val in avg_data_dict[key]]

    return avg_data_dict


if __name__ == "__main__":
    folder_path = "./experiment/experiment3/ours"  # Path to your data folder
    # folder_path = "./experiment/experiment3/catnip"  # Path to your data folder

    for exp_name in ["statues" ,"flightroom", "stonehenge"]:
        avg_data_dict = calculate_average_data(folder_path, exp_name)

        fp = f'{folder_path}/{exp_name}/data_average.json'
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as outfile:
            json.dump(avg_data_dict, outfile, indent=4)
        print(f"Finish save file at {fp}")

    # print("Average Time Metrics:")
    # for key, value in avg_data_dict.items():
    #     print(f"{key}: {value}") 