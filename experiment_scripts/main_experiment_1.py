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

# # # Flightroom
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

N_test = 100     # Number of test trajectories
t = np.linspace(0, np.pi, N_test)

x0 = np.stack([r*np.cos(t), r*np.sin(t), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)     # starting positions
xf = np.stack([r*np.cos(t + np.pi), r*np.sin(t + np.pi), dz * 2*(np.random.rand(N_test)-0.5)], axis=-1)     # goal positions

x0 = x0 + center        # Shift circle
xf = xf + center


for num in range(10):

    list_plan = []

    # for experiment only
    init_path_list = []
    path_reorder_list = []
    path_reconst_list = []
    path_reorder_reconst_list = []
    path_los_list = []
    path_pnp_list = []
    path_remove_reconst_pnp_list = []
    path_post_list = []

    number_path = 0
    total_astar_time = 0
    total_post_reorder_time = 0 # for experiment only
    total_post_reconst_time = 0 # for experiment only
    total_post_reorder_reconst_time = 0 # for experiment only
    total_post_los_time = 0 # for experiment only
    total_post_pnp_time = 0 # for experiment only
    total_post_remove_reconst_pnp = 0 # for experiment only
    total_post_time = 0
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
            # init_path = planner.jps(x0_ns, xf_ns) # JPS
            current_astar_time = time.perf_counter() - start_time

            # for experiment (reorder)
            start_time = time.perf_counter()
            post_path_reorder = planner.post_processing_reorder(init_path)
            current_post_reorder_time = time.perf_counter() - start_time

            # for experiment (reconst)
            start_time = time.perf_counter()
            post_path_reconst = planner.post_processing_reconst(init_path)
            current_post_reconst_time = time.perf_counter() - start_time

            # for experiment (reorder+reconst)
            start_time = time.perf_counter()
            post_path_reorder_reconst = planner.post_processing_reorder_reconst(init_path)
            current_post_reorder_reconst_time = time.perf_counter() - start_time

            # for experiment (LOS)
            start_time = time.perf_counter()
            post_path_los = planner.post_processing_los(init_path)
            current_post_los_time = time.perf_counter() - start_time

            # for experiment (PNP)
            start_time = time.perf_counter()
            post_path_pnp = planner.post_processing_pnp(init_path)
            current_post_pnp_time = time.perf_counter() - start_time

            # for experiment (remove node+reconst+PNP)
            start_time = time.perf_counter()
            post_path_remove_reconst_pnp = planner.post_processing_remove_reconst_pnp(init_path)
            current_post_remove_reconst_pnp = time.perf_counter() - start_time

            start_time = time.perf_counter()
            post_path = planner.post_processing(init_path)
            current_post_time = time.perf_counter() - start_time

            # add to list
            # for experiment only
            init_path_list.append(init_path)
            path_reorder_list.append(post_path_reorder)
            path_reconst_list.append(post_path_reconst)
            path_reorder_reconst_list.append(post_path_reorder_reconst)
            path_los_list.append(post_path_los)
            path_pnp_list.append(post_path_pnp)
            path_remove_reconst_pnp_list.append(post_path_remove_reconst_pnp)
            path_post_list.append(post_path)
            # list_plan.append(traj)

            # add time
            total_astar_time += current_astar_time
            total_post_reorder_time += current_post_reorder_time
            total_post_reconst_time += current_post_reconst_time
            total_post_reorder_reconst_time += current_post_reorder_reconst_time
            total_post_los_time += current_post_los_time
            total_post_pnp_time += current_post_pnp_time
            total_post_remove_reconst_pnp += current_post_remove_reconst_pnp
            total_post_time += current_post_time

            number_path += 1

        except:
            print('Error in cannot find a path or unable to optimize.')

    print(f"total time = {time.perf_counter() - total_time}")
    print("\nTotal execution times:")
    print(f"A* Search: {total_astar_time:.4f}s")
    print(f"Post-processing: {total_post_time:.4f}s")

    # Convert all paths to native Python integers
    def convert_to_native_int(path_list):
        converted_paths = []
        for path in path_list:
            new_path = []
            for point in path:
                # Check if point is a tuple or a NumPy array
                if isinstance(point, (tuple, list, np.ndarray)) and len(point) == 3:
                    x, y, z = point
                    new_path.append((int(x), int(y), int(z)))
                else:
                    print(f"Unexpected format in path: {point}")
            converted_paths.append(new_path)
        return converted_paths

    # Apply conversion to all paths
    init_path_list = convert_to_native_int(init_path_list)
    path_reorder_list = convert_to_native_int(path_reorder_list)
    path_reconst_list = convert_to_native_int(path_reconst_list)
    path_reorder_reconst_list = convert_to_native_int(path_reorder_reconst_list)
    path_los_list = convert_to_native_int(path_los_list)
    path_pnp_list = convert_to_native_int(path_pnp_list)
    path_remove_reconst_pnp_list = convert_to_native_int(path_remove_reconst_pnp_list)
    path_post_list = convert_to_native_int(path_post_list)

    data = {
        'number_path': number_path,
        'total_astar_time': total_astar_time,
        'total_post_reorder_time': total_post_reorder_time,
        'total_post_reconst_time': total_post_reconst_time,
        'total_post_reorder_reconst_time': total_post_reorder_reconst_time,
        'total_post_los_time': total_post_los_time,
        'total_post_pnp_time': total_post_pnp_time,
        'total_post_remove_reconst_pnp': total_post_remove_reconst_pnp,
        'total_post_time': total_post_time,
        'init_path': init_path_list,
        'path_reorder': path_reorder_list,
        'path_reconst': path_reconst_list,
        'path_reorder_reconst': path_reorder_reconst_list,
        'path_los': path_los_list,
        'path_pnp': path_pnp_list,
        'path_remove_reconst_pnp': path_remove_reconst_pnp_list,
        'path_post': path_post_list,
    }


    fp = f'result/experiment/experiment1/{exp_name}/data_seed{SEED}_exnum{num+1}.json'
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)
# %%
