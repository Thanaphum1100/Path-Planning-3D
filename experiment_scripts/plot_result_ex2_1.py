import json
import matplotlib.pyplot as plt

# Load data
data_path = "./experiment/experiment2/stonehenge/data_exnum1.json"
with open(data_path, 'r') as f:
    meta = json.load(f)

# Define categories and weights
weights = [10, 100, 1000]
# categories = ["traj", "traj_v", "traj_a", "traj_j", "traj_s"]
categories = ["traj", "traj_v", "traj_a"]
METRICES_NAME = ["Position", "Velocity", "Acceleration"]

PATH_KEYS = {cat: [f"{cat}_w{w}" for w in weights] for cat in categories}

index_to_plot = 3  # Select specific index


axes_labels = ['X', 'Y', 'Z']
TITLE_NAME = ["X-axis", "Y-axis", "Z-axis"]


# Dynamically set number of rows and columns
num_rows = len(categories)
num_cols = len(axes_labels)

# Create a figure with dynamic subplots
plt.rcParams["font.family"] = "Times New Roman"

fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))

# Loop through categories and axes
for row_idx, category in enumerate(categories):  # Row index: 0 = traj, 1 = traj_v
    for col_idx, axis_label in enumerate(axes_labels):  # Column index: 0 = X, 1 = Y, 2 = Z
        ax = axes[row_idx, col_idx]  # Select the subplot

        for w in weights:
            key = f"{category}_w{w}"
            if key in meta:  # Ensure the key exists
                points = meta[key][index_to_plot]  # Get x, y, z list
                time_stamp = meta["time_value"][index_to_plot]  # Get corresponding time

                # Extract the axis component (X, Y, or Z)
                values = [point[col_idx] for point in points]

                # Plot each weight as a separate line
                ax.plot(time_stamp[:len(values)], values, label=f"$\lambda$={w}", linewidth=2)

        # Plot original path if it exists
        ori_key = f"{category}_ori"
        if ori_key in meta:
            ori_points = meta[ori_key][index_to_plot]
            ori_time_stamp = meta["time_value"][index_to_plot]
            ori_values = [point[col_idx] for point in ori_points]
            ax.plot(ori_time_stamp[:len(ori_values)], ori_values,
                    label="Original", linestyle="--", color="black", linewidth=2)


        if row_idx == 0:
            axes[0, col_idx].set_title(TITLE_NAME[col_idx], fontsize=14)

        axes[row_idx, col_idx].tick_params(axis='x', labelsize=10)  # Set x-axis tick font size
        axes[row_idx, col_idx].tick_params(axis='y', labelsize=10)  # Set y-axis tick font

        ax.set_xlabel("Time Stamp (scaled time)", fontsize=12)
        ax.set_ylabel(f"{METRICES_NAME[row_idx]} {axis_label}", fontsize=12)
        # ax.set_title(f"{category.upper()} - {axis_label} Axis")
        ax.legend(fontsize=10)
        ax.grid()

# Adjust layout and show the entire figure
plt.tight_layout()
plt.show()


