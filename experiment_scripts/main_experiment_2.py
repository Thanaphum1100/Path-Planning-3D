#%% 
import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R
import time
import os

#Import utilies
from nerf.nerf import NeRFWrapper
from purr.purr import Catnips
from corridor.init_path import PathInit
from corridor.bounds import BoxCorridor
from planner.spline_planner import SplinePlanner
# from planner.mpc import MPC 

from pipeline_scripts.Global_planner import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

###NOTE: Point your path to the correct NeRF model

# Stonehenge
nerfwrapper = NeRFWrapper("./outputs/stonehenge/nerfacto/2025-02-06_020342") #blender data 
exp_name = 'stonehenge'

# # Statues
# nerfwrapper = NeRFWrapper("./outputs/statues/nerfacto/2024-07-02_135009")
# exp_name = 'statues'

# # Flightroom
# nerfwrapper = NeRFWrapper("./outputs/flightroom/nerfacto/2025-02-05_031542")
# exp_name = 'flightroom'

# World frame will convert the path back to the world frame from the Nerfstudio frame
world_frame = False

#%%
### Catnips configs

# Grid is the bounding box of the scene in which we sample from. Note that this grid is typically in the Nerfstudio frame.
# Stonehenge
grid = np.array([
    [-1.4, 1.1],
    [-1.4, 1.1],
    [-0.1, 0.5]
    ])   


# # Statues
# grid = np.array([
#     [-1., 1.],
#     [-1, 1.],
#     [-.5, .5]
#     ])   

# # Flightroom
# grid = np.array([
#     [-1., 1.],
#     [-1., 1.],
#     [-0.5, 0.5]
#     ])   


# Stonehenge
r = 1.12
dz = 0.05
center = np.array([-0.21, -0.132, 0.16])

# # Statues
# r = 0.475       # radius of circle
# dz = 0.05       # randomness in height of circle
# center = np.array([-0.064, -0.0064, -0.025])

# # Flightroom
# r = 2.6
# dz = 0.2
# center = np.array([0.10, 0.057, 0.585])


# Create robot body. This is also in the Nerfstudio frame/scale.
# agent_body = .03*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]])
agent_body = .01*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]]) # Stonehenge


# #Configs
sigma = 0.99    # Chance of being below interpenetration volume
discretization = 150    # Number of partitions per side of voxel grid

catnips_configs = {
    'grid': grid,               # Bounding box of scene
    'agent_body': agent_body,   # Bounding box of agent in body frame
    'sigma': sigma,             # chance of being below interpenetration vol.
    'Aaux': 1e-8,               # area of auxiliary/reference particle
    'dt': 1e-2,                 # depth of aux/ref particle
    'Vmax': 5e-6,               # interpenetration volume
    'gamma': 1.,                # occlusion threshold
    'discretization': discretization,   # number of partitions per side of voxel grid
    'density_factor': 1,        # scaling factor to density
    'get_density': nerfwrapper.get_density  # queries NeRF to get density
}

catnips = Catnips(catnips_configs)      # Instantiate class
catnips.load_purr()                     # MUST load details about the PURR
catnips.create_purr()                 # Generates PURR voxel grid

### If you need to run this in real-time, don't save the mesh.
catnips.save_purr(f'./catnips_data/{exp_name}/purr/',save_property=True) #, transform=nerfwrapper.transform.cpu().numpy(), scale=nerfwrapper.scale, save_property=True)


#%%

SEED = 1
if SEED is not None:
    np.random.seed(SEED)

planner = GlobalPlanner(~catnips.purr, catnips.conv_centers)

N_test = 10      # Number of test trajectories
t = np.linspace(0, np.pi, N_test)

x0 = np.stack([r*np.cos(t), r*np.sin(t), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)     # starting positions
xf = np.stack([r*np.cos(t + np.pi), r*np.sin(t + np.pi), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)     # goal positions

x0 = x0 + center        # Shift circle
xf = xf + center

for num in range(1):
    # for experiment only
    minsnap_weight = [10, 100, 1000]

    list_plan_points = []
    time_value_list = []
    ori_traj_list = []
    path_dict = {}

    path_dict["traj_ori"] = []
    path_dict["traj_v_ori"] = []
    path_dict["traj_a_ori"] = []
    path_dict["traj_j_ori"] = []
    path_dict["traj_s_ori"] = []


    for weight in minsnap_weight:
        path_dict[f'time_w{weight}'] = []
        path_dict[f'traj_w{weight}'] = []
        path_dict[f'traj_v_w{weight}'] = []
        path_dict[f'traj_a_w{weight}'] = []
        path_dict[f'traj_j_w{weight}'] = []
        path_dict[f'traj_s_w{weight}'] = []
        

    number_path = 0
    total_ns_time = 0
    total_astar_time = 0
    total_post_time = 0
    total_boundary_time = 0
    total_smoothing_time = 0
    total_time = time.perf_counter()

    for it, (start, end) in enumerate(zip(x0, xf)):
        print(f"Index {it}", end="\r")

        # converts the start and end locations to the Nerfstudio frame
        if world_frame:
            x0_ns = nerfwrapper.data_frame_to_ns_frame(torch.from_numpy(start).to(device, dtype=torch.float32)).squeeze().cpu().numpy()
            xf_ns = nerfwrapper.data_frame_to_ns_frame(torch.from_numpy(end).to(device, dtype=torch.float32)).squeeze().cpu().numpy()

        else:
            x0_ns = start
            xf_ns = end

        try:
            start_time = time.perf_counter()
            init_path = planner.astar(x0_ns, xf_ns) # Astar
            current_astar_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            post_path = planner.post_processing(init_path)
            current_post_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            path_points, upper, lower = planner.create_boundary(post_path, interpolate_path=True)
            current_boundary_time = time.perf_counter() - start_time

            path_points_dis = planner.path_index_to_path_point(post_path)

            temp_path_dict = {}
            for weight in minsnap_weight:
                start_time = time.perf_counter()
                traj = planner.path_smothing(path_points, upper, lower, weight=weight)
                current_smoothing_time = time.perf_counter() - start_time
            
                traj_v, traj_a, traj_j, traj_s = planner.get_differential()

                temp_path_dict[f'time_w{weight}'] = current_smoothing_time
                temp_path_dict[f'traj_w{weight}'] = traj.tolist()
                temp_path_dict[f'traj_v_w{weight}'] = traj_v.tolist()
                temp_path_dict[f'traj_a_w{weight}'] = traj_a.tolist()
                temp_path_dict[f'traj_j_w{weight}'] = traj_j.tolist()
                temp_path_dict[f'traj_s_w{weight}'] = traj_s.tolist()

            time_value = planner.get_time_stamps()
            ori_traj = planner.get_original_traj()
            ori_traj_v, ori_traj_a, ori_traj_j, ori_traj_s = planner.get_original_differential()

            path_dict["traj_ori"].append(ori_traj.tolist())
            path_dict["traj_v_ori"].append(ori_traj_v.tolist())
            path_dict["traj_a_ori"].append(ori_traj_a.tolist())
            path_dict["traj_j_ori"].append(ori_traj_j.tolist())
            path_dict["traj_s_ori"].append(ori_traj_s.tolist())

            if world_frame:
                traj = nerfwrapper.ns_frame_to_data_frame(torch.from_numpy(traj[..., :3]).to(device, dtype=torch.float32)).squeeze().cpu().numpy()

            # for experiment
            list_plan_points.append(path_points_dis)
            time_value_list.append(time_value)

            for weight in minsnap_weight:
                path_dict[f'time_w{weight}'].append(temp_path_dict[f'time_w{weight}'])
                path_dict[f'traj_w{weight}'].append(temp_path_dict[f'traj_w{weight}'])
                path_dict[f'traj_v_w{weight}'].append(temp_path_dict[f'traj_v_w{weight}'])
                path_dict[f'traj_a_w{weight}'].append(temp_path_dict[f'traj_a_w{weight}'])
                path_dict[f'traj_j_w{weight}'].append(temp_path_dict[f'traj_j_w{weight}'])
                path_dict[f'traj_s_w{weight}'].append(temp_path_dict[f'traj_s_w{weight}'])

            # add time
            total_astar_time += current_astar_time
            total_post_time += current_post_time
            total_boundary_time += current_boundary_time
            number_path += 1

        except:
            print('Error in cannot find a path or unable to optimize.')


    print(f"total time = {time.perf_counter() - total_time}")

    print("\nTotal execution times:")
    print(f"NS Frame Conversion: {total_ns_time:.4f}s")
    print(f"A* Search: {total_astar_time:.4f}s")
    print(f"Post-processing: {total_post_time:.4f}s")
    print(f"Boundary Creation: {total_boundary_time:.4f}s")
    print(f"Path Smoothing: {total_smoothing_time:.4f}s")


    def convert_to_native_float(path_list):
        converted_paths = []
        for path in path_list:
            new_path = []
            for point in path:
                if isinstance(point, (tuple, list, np.ndarray)) and len(point) == 3:
                    x, y, z = point
                    new_path.append((float(x), float(y), float(z)))
                else:
                    print(f"Unexpected format in path: {point}")
            converted_paths.append(new_path)
        return converted_paths

    data = {
        'number_path': number_path,
        'total_astar_time': total_astar_time,
        'total_post_time': total_post_time,
        'total_boundary_time': total_boundary_time,
        'traj_point': convert_to_native_float(list_plan_points),
        'time_value': [time_value.tolist() for time_value in time_value_list]
    }
    data.update(path_dict)  # Add path data

    fp = f'result/experiment/experiment2/{exp_name}/data_seed{SEED}_exnum{num+1}.json'
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)
# %%
