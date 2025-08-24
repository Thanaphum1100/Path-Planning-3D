import subprocess

# Path to Blender file
# blend_file = "statues.blend"  
blend_file = "stonehenge.blend" 

# # Path to Python script 
script_path = "experiment_scripts/blender_traj.py" 

# Run Blender with the script and parameters
subprocess.run(['blender', blend_file, '-P', script_path])