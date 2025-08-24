import json
import matplotlib.pyplot as plt
import numpy as np

# Experiment names
EXP_NAMES = ["stonehenge", "statues", "flightroom"]
TITLE_NAME = ["Stonehenge", "Statues", "Flightroom"]

## with LoS
# # Keys for different metrics
# AVG_TIME_KEYS = [
#     "avg_astar_time", "avg_post_los_time", "avg_post_pnp_time", "avg_post_reorder_reconst_time", "avg_post_remove_reconst_pnp", "avg_post_time"
# ]
# AVG_PATH_KEYS = [
#     "avg_init_path", "avg_path_los", "avg_path_pnp", "avg_path_reorder_reconst", "avg_path_remove_reconst_pnp", "avg_path_post"
# ]
# AVG_ANGLE_KEYS = [
#     "avg_angle_init_path", "avg_angle_path_los", "avg_angle_path_pnp", "avg_angle_path_reorder_reconst", "avg_angle_path_remove_reconst_pnp", "avg_angle_path_post"
# ]

## without LoS
AVG_TIME_KEYS = [
    "avg_astar_time", "avg_post_pnp_time", "avg_post_reorder_reconst_time", "avg_post_remove_reconst_pnp", "avg_post_time"
]
AVG_PATH_KEYS = [
    "avg_init_path", "avg_path_pnp", "avg_path_reorder_reconst", "avg_path_remove_reconst_pnp", "avg_path_post"
]
AVG_ANGLE_KEYS = [
    "avg_angle_init_path", "avg_angle_path_pnp", "avg_angle_path_reorder_reconst", "avg_angle_path_remove_reconst_pnp", "avg_angle_path_post"
]

# Define colors
## with LoS
# LABEL_COLORS = ['Astar', 'LOS', 'PNP', 'Reorder+Shortcut', 'Remove+Shortcut+PNP', 'Full Reconstructing']
# TIME_COLORS = ['lightskyblue', 'lightcoral', 'palegreen', 'gold', 'orange', 'plum']
# PATH_COLORS = ['lightskyblue', 'lightcoral', 'palegreen', 'gold', 'orange', 'plum']
# ANGLE_COLORS = ['lightskyblue', 'lightcoral', 'palegreen', 'gold', 'orange', 'plum']

# with LoS
LABEL_COLORS = ['Astar', 'PNP', 'Reorder+Shortcut', 'Remove+Shortcut+PNP', 'Full Reconstruction']
TIME_COLORS = ['lightskyblue', 'lightcoral', 'gold', 'orange', 'plum']
PATH_COLORS = ['lightskyblue', 'lightcoral', 'gold', 'orange', 'plum']
ANGLE_COLORS = ['lightskyblue', 'lightcoral', 'gold', 'orange', 'plum']

# TIME_COLORS = ['lightskyblue', 'salmon', 'limegreen', 'gold', 'plum']
# PATH_COLORS = ['lightskyblue', 'salmon', 'limegreen', 'gold', 'plum']
# ANGLE_COLORS = ['lightskyblue', 'salmon', 'limegreen', 'gold', 'plum']

# Create figure with 3 rows and len(EXP_NAMES) columns
plt.rcParams["font.family"] = "Times New Roman"

fig, axes = plt.subplots(3, len(EXP_NAMES), figsize=(15, 12))
fig.subplots_adjust(hspace=0.4)

# Check if axes is 1D (only one experiment) and adjust indexing accordingly
if len(EXP_NAMES) == 1:
    axes = np.expand_dims(axes, axis=1)  # Convert 1D to 2D for consistent indexing

# Stacked Bar Chart for Time Metrics (for multiple experiments)
for col, exp_name in enumerate(EXP_NAMES):
    data_path = f"./experiment/experiment1/{exp_name}/data_average.json"
    
    # Load data
    with open(data_path, 'r') as f:
        meta = json.load(f)
    
    # Extract values
    astar_time = meta["avg_astar_time"]
    other_times = [meta[key] for key in AVG_TIME_KEYS[1:]]  # Exclude A* time
    
    astar_color = TIME_COLORS[0]  # A* time color
    other_colors = TIME_COLORS[1:]  # Other times colors
    
    # Plot A* time under every method (for each experiment)
    for key in AVG_TIME_KEYS:  
        axes[0, col].bar(key, astar_time, color=astar_color, label=LABEL_COLORS[0] if key == AVG_TIME_KEYS[0] else "")
        if key == AVG_TIME_KEYS[0]:
            axes[0, col].text(key, astar_time + 0.01*astar_time, f"{astar_time:.5f}", ha='center', va='bottom', fontsize=10)
        else:
            axes[0, col].text(key, astar_time * 0.85, f"{astar_time:.5f}", ha='center', va='bottom', fontsize=10)
        # if key == AVG_TIME_KEYS[0]:
        #     axes[0, col].text(key, astar_time + 0.01, f"{astar_time:.5f}", ha='center', va='bottom', fontsize=10)

    # Plot each method’s time on top of A* time (for each experiment)
    bottom_values = astar_time
    for i, key in enumerate(AVG_TIME_KEYS[1:]):
        axes[0, col].bar(key, meta[key], bottom=bottom_values, color=other_colors[i], label=LABEL_COLORS[i+1])
        axes[0, col].text(key, bottom_values + meta[key] * 1.15, f"{meta[key]:.5f}", ha='center', va='bottom', fontsize=10)
        # axes[0, col].text(key, bottom_values + meta[key] + 0.01, f"{meta[key]:.5f}", ha='center', va='bottom', fontsize=10)
    
    # Set titles and labels for each experiment column
    axes[0, col].set_title(TITLE_NAME[col], fontsize=14)  # Set experiment name on top
    axes[0, col].set_xlabel("Computation Time", fontsize=12)
    axes[0, col].set_ylabel("Total Time (seconds)", fontsize=12)

    axes[0, col].set_ylim(0, bottom_values * 1.5)

    axes[0, col].set_xticklabels([])  # Hides the tick labels
    axes[0, col].set_xticks([])

    axes[0, col].tick_params(axis='x', labelsize=10)  # Set x-axis tick font size
    axes[0, col].tick_params(axis='y', labelsize=10)  # Set y-axis tick font
    
    # # Add legend only for the first column (col == 0)
    if col == 0:
        axes[0, col].legend(loc='lower center', bbox_to_anchor=(1.65, 1.2), ncol=5, fontsize=14)


    # ---- 2️⃣ Path Length Whisker Plot (Second Row) ----
    path_data = [meta[key] for key in AVG_PATH_KEYS]
    box_plot = axes[1, col].boxplot(path_data, labels=AVG_PATH_KEYS, patch_artist=True, showfliers=True)
    
    for patch, color in zip(box_plot['boxes'], PATH_COLORS):
        patch.set_facecolor(color)
    
    for median in box_plot['medians']:
        median.set_color('black')

    for i, data in enumerate(path_data):
        mean_value = np.mean(data)
        axes[1, col].scatter(i + 1, mean_value, color='black', marker='x', s=80, zorder=3, label="Mean" if i == 0 else "")

        min_val = np.min(data)
        max_val = np.max(data)

        # Show text for min, max, and mean
        axes[1, col].text(i + 1, min_val * 0.995, f"{min_val:.2f}", ha='center', va='top', fontsize=10, color='black')
        axes[1, col].text(i + 1 + 0.15, mean_value * 1.015, f"{mean_value:.2f}", ha='left', va='bottom', fontsize=10, color='black')
        axes[1, col].text(i + 1, max_val * 1.005, f"{max_val:.2f}", ha='center', va='bottom', fontsize=10, color='black')

    axes[1, col].set_xlabel("Algorithms", fontsize=12)
    axes[1, col].set_ylabel("Path Length (movement cost)", fontsize=12)
    axes[1, col].set_xticklabels([])  # Hides the tick labels
    axes[1, col].set_xticks([])

    axes[1, col].tick_params(axis='x', labelsize=10)  # Set x-axis tick font size
    axes[1, col].tick_params(axis='y', labelsize=10)  # Set y-axis tick font

    all_min_val = np.min([np.min(d) for d in path_data])
    all_max_val = np.max([np.max(d) for d in path_data])
    axes[1, col].set_ylim(all_min_val * 0.95, all_max_val * 1.05)

    if col == 0:
        axes[1, col].legend()

    # ---- 3️⃣ Angle Whisker Plot (Third Row) ----
    angle_data = [meta[key] for key in AVG_ANGLE_KEYS]
    box_plot = axes[2, col].boxplot(angle_data, labels=AVG_ANGLE_KEYS, patch_artist=True, showfliers=True)

    for patch, color in zip(box_plot['boxes'], ANGLE_COLORS):
        patch.set_facecolor(color)

    for median in box_plot['medians']:
        median.set_color('black')

    for i, data in enumerate(angle_data):
        mean_value = np.mean(data)
        axes[2, col].scatter(i + 1, mean_value, color='black', marker='x', s=80, zorder=3, label="Mean" if i == 0 else "")

        min_val = np.min(data)
        max_val = np.max(data)

        # Show text for min, max, and mean
        axes[2, col].text(i + 1, min_val * 0.9, f"{min_val:.2f}", ha='center', va='top', fontsize=10, color='black')
        axes[2, col].text(i + 1 + 0.15, mean_value * 1.3, f"{mean_value:.2f}", ha='left', va='bottom', fontsize=10, color='black')
        axes[2, col].text(i + 1, max_val * 1.05, f"{max_val:.2f}", ha='center', va='bottom', fontsize=10, color='black')

    axes[2, col].set_xlabel("Algorithms", fontsize=12)
    axes[2, col].set_ylabel("Number of turning points", fontsize=12)
    axes[2, col].set_xticklabels([])  # Hides the tick labels
    axes[2, col].set_xticks([])

    axes[2, col].tick_params(axis='x', labelsize=10)  # Set x-axis tick font size
    axes[2, col].tick_params(axis='y', labelsize=10)  # Set y-axis tick font

    all_min_val = np.min([np.min(d) for d in angle_data])
    all_max_val = np.max([np.max(d) for d in angle_data])
    axes[2, col].set_ylim(-2, all_max_val * 1.2)

    if col == 0:
        axes[2, col].legend()
# Show final figure
plt.show()
