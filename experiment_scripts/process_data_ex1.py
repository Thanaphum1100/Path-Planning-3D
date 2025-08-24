import json
import os
import math


TIME_KEYS = [
    "total_astar_time", "total_post_reorder_time", "total_post_reconst_time", "total_post_reorder_reconst_time", 
    "total_post_los_time", "total_post_pnp_time", "total_post_remove_reconst_pnp", "total_post_time"
]

AVG_TIME_KEYS = [
    "avg_astar_time", "avg_post_reorder_time", "avg_post_reconst_time", "avg_post_reorder_reconst_time", 
    "avg_post_los_time", "avg_post_pnp_time", "avg_post_remove_reconst_pnp", "avg_post_time"
]

PATH_KEYS = [
    "init_path", "path_reorder", "path_reconst", "path_reorder_reconst",
    "path_los", "path_pnp", "path_remove_reconst_pnp", "path_post"

]

AVG_PATH_KEYS = [
    "avg_init_path", "avg_path_reorder", "avg_path_reconst", "avg_path_reorder_reconst",
    "avg_path_los", "avg_path_pnp", "avg_path_remove_reconst_pnp", "avg_path_post"
]

AVG_ANGLE_KEYS = [
    "avg_angle_init_path", "avg_angle_path_reorder", "avg_angle_path_reconst", "avg_angle_path_reorder_reconst",
    "avg_angle_path_los", "avg_angle_path_pnp", "avg_angle_path_remove_reconst_pnp", "avg_angle_path_post"
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


def count_turning_points(path):
    """Counts the number of turning points in a 3D path."""
    def is_collinear(v1, v2):
        """Check if two 3D vectors are collinear."""
        return (v1[0] * v2[1] == v1[1] * v2[0]) and \
            (v1[0] * v2[2] == v1[2] * v2[0]) and \
            (v1[1] * v2[2] == v1[2] * v2[1])

    if len(path) < 3:
        return 0  # No turns possible with fewer than 3 points

    turning_points = 0
    for i in range(2, len(path)):
        v1 = (path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1], path[i-1][2] - path[i-2][2])
        v2 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1], path[i][2] - path[i-1][2])

        if not is_collinear(v1, v2):  # Turning point detected
            turning_points += 1

    return turning_points


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
    for path_key, avg_length_key, avg_angle_key in zip(PATH_KEYS, AVG_PATH_KEYS, AVG_ANGLE_KEYS):
        paths = meta[path_key]
        avg_data_dict[avg_length_key] = [calculate_path_length(path) for path in paths]
        avg_data_dict[avg_angle_key] = [count_turning_points(path) for path in paths]

    # for key, value in avg_data_dict.items():
    #     print(f"{key}: {value}") 

    return avg_data_dict


def calculate_average_data(folder_path, exp_name, experiment_number=10):
    """Calculates the average times and path lengths across multiple experiments."""

    # Initialize the dictionary for averaging 
    avg_data_dict = {key: 0 for key in AVG_TIME_KEYS}
    avg_data_dict.update({key: [] for key in AVG_PATH_KEYS})  # For path lengths
    avg_data_dict.update({key: [] for key in AVG_ANGLE_KEYS})  # For turning points

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

        # Accumulate turning points element-wise
        for key in AVG_ANGLE_KEYS:
            turning_points = cur_data_dict[key]

            # If first experiment, initialize list with correct size
            if not avg_data_dict[key]:
                avg_data_dict[key] = [0] * len(turning_points)

            # Sum corresponding turning points
            for j in range(len(turning_points)):
                avg_data_dict[key][j] += turning_points[j]

    # Compute the final average values
    for key in AVG_TIME_KEYS:
        avg_data_dict[key] /= experiment_number

    for key in AVG_PATH_KEYS:
        avg_data_dict[key] = [val / experiment_number for val in avg_data_dict[key]]

    for key in AVG_ANGLE_KEYS:
        avg_data_dict[key] = [val / experiment_number for val in avg_data_dict[key]]

    return avg_data_dict


if __name__ == "__main__":
    folder_path = "./experiment/experiment1"  # Path to your data folder

    exp_name = "statues"  # Name of your experiment
    # exp_name = "flightroom"
    # exp_name = "stonehenge"
    
    avg_data_dict = calculate_average_data(folder_path, exp_name)

    fp = f'./experiment/experiment1/{exp_name}/data_average.json'
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as outfile:
        json.dump(avg_data_dict, outfile, indent=4)
    print(f"Finish save file at {fp}")

    # print("Average Time Metrics:")
    # for key, value in avg_data_dict.items():
    #     print(f"{key}: {value}") 


