import numpy as np
import torch
import scipy
# from scipy.spatial import KDTree


class Boundary:
    def __init__(self, grid_binary_occupied, grid_pts_occupied, r=0.1):
        self.r = r
        self.occupied = grid_binary_occupied
        self.lx, self.ly, self.lz = grid_pts_occupied[1, 1, 1] - grid_pts_occupied[0, 0, 0]

        eroded_occupy = scipy.ndimage.binary_erosion(self.occupied)
        surface_voxels = np.logical_and(self.occupied, ~eroded_occupy)
        self.centers = grid_pts_occupied[surface_voxels]

        self.grid_pts_kd = scipy.spatial.KDTree(self.centers)

    def check_collision(self, bounds_pts, obs_pts):
        n = obs_pts.shape[0]
        bounds_expanded = bounds_pts[None, :].expand(n, 6)
        coll_per_pt = torch.all(bounds_expanded >= obs_pts, dim=-1)
        return torch.any(coll_per_pt)

    def grow_box(self, occ_pts, minimal, maximal, delta): 
        occupied_pts = torch.cat([occ_pts, -occ_pts], dim=-1)

        bounds = minimal.T.reshape(-1)
        bounds = bounds*torch.tensor([1., 1., 1., -1., -1., -1.]).cuda()

        maximal_bounds = maximal.T.reshape(-1)
        maximal_bounds = maximal_bounds * torch.tensor([1., 1., 1., -1., -1., -1.]).cuda()

        side_ind = np.arange(6)

        # Stop loop if no more possible degrees of freedom
        while len(side_ind) > 0:

            next_side_ind = []
            for side in side_ind:
                # Check to see if adding to a particular side will cause collision
                delta_side = delta[side]
                bounds_copy = torch.clone(bounds)
                bounds_copy[side] += delta_side

                is_collide = self.check_collision(bounds_copy, occupied_pts)
    
                # If occupied pts are in the bounds when we grow it, do
                # not use that dilation and remove it from future growths
                # in that direction.
                if not is_collide and (bounds_copy[side] <= maximal_bounds[side]):
                    next_side_ind.append(side)
                    bounds[side] += delta_side
            side_ind = next_side_ind

        bounds = bounds*torch.tensor([1., 1., 1., -1., -1., -1.]).cuda()
        return bounds.reshape(-1, 3).T.cpu().numpy() #TODO: fix? (maybe not)

    def create_corridor(self, traj):

        bounds = []
        delta = torch.tensor([self.lx ,self.ly, self.lz, self.lx, self.ly, self.lz]).cuda()

        min_delta = torch.tensor([self.lx / 2, self.ly / 2, self.lz / 2 ]).cuda()

        traj_points = traj[1:-1] # exclude start and goal

        for i, point in enumerate(traj_points, start=1):

            dist_prev = np.linalg.norm(traj[i] - traj[i-1])  # Distance from previous point
            dist_next = np.linalg.norm(traj[i+1] - traj[i])  # Distance from next point
            dist = max(dist_prev, dist_next)

            if dist > self.r:
                r = dist
            else:
                r = self.r

            offset = np.array([r/(2*.866), r/(2*.866), r/(2*.866)])

            neighbors = self.grid_pts_kd.query_ball_point(point, r)
            near_pts = torch.from_numpy(self.grid_pts_kd.data[neighbors]).cuda()
        
            xbound_min = [point[0] + min_delta[0], point[0] - min_delta[0]]
            ybound_min = [point[1] + min_delta[1], point[1] - min_delta[1]]
            zbound_min = [point[2] + min_delta[2], point[2] - min_delta[2]]
            minimal = torch.tensor([xbound_min, ybound_min, zbound_min]).cuda()
            maximal = torch.tensor(np.array([point + offset, point - offset])).T.cuda()

            bound = self.grow_box(near_pts, minimal, maximal, delta)
            bounds.append(bound)

        return bounds

    def get_upper_lower_bounds(self, traj):
        bounds = self.create_corridor(traj)

        upper = [(b[0, 0], b[1, 0], b[2, 0]) for b in bounds]
        lower = [(b[0, 1], b[1, 1], b[2, 1]) for b in bounds]

        return upper, lower