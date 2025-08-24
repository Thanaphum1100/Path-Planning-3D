import numpy as np

from pipeline_scripts.AStar import AStar
from pipeline_scripts.post_processing import PostProcessing
from pipeline_scripts.post_processing_baseline import PostProcessingBaseline


from pipeline_scripts.boundary import *
from pipeline_scripts.minsnap import *



class GlobalPlanner:
    def __init__(self, grid_occupied, grid_points) -> None:
        self.grid_occupied = grid_occupied 
        self.grid_points = grid_points 
        self.cell_sizes = self.grid_points[1, 1, 1] - self.grid_points[0, 0, 0]
        self.boundary = Boundary(grid_occupied, grid_points)

    def get_indices(self, point):
        min_bound = self.grid_points[0, 0, 0] - self.cell_sizes/2

        transformed_pt = point - min_bound

        indices = np.round(transformed_pt / self.cell_sizes).astype(np.uint32)

        return_indices = indices.copy()
        # If querying points outside of the bounds, project to the nearest side
        for i, ind in enumerate(indices):
            if ind < 0.:
                return_indices[i] = 0

                print('Point is outside of minimum bounds. Projecting to nearest side. This may cause unintended behavior.')

            elif ind > self.grid_occupied.shape[i]:
                return_indices[i] = self.grid_occupied.shape[i]

                print('Point is outside of maximum bounds. Projecting to nearest side. This may cause unintended behavior.')

        return return_indices

    def astar(self, x0, xf):
        start = tuple(map(np.int32, self.get_indices(x0))) # Find nearest grid point and find its index
        goal = tuple(map(np.int32, self.get_indices(xf)))

        start_occupied = self.grid_occupied[start[0], start[1], start[2]]
        goal_occupied = self.grid_occupied[goal[0], goal[1], goal[2]]

        if goal_occupied:
            raise ValueError('Goal is in occupied voxel. Please choose another end point.')

        if start_occupied:
            raise ValueError('Start is in occupied voxel. Please choose another starting point.')

        astar = AStar(start, goal, self.grid_occupied)
        path_astar, closed_list_astar, opened_list_astar = astar.run()

        if path_astar == None:
            raise ValueError("A* cannot find the initial path. Path does not exist.")
        else:
            return path_astar

    def post_processing_reorder(self, path): # for experiment (reorder)
        post_processing_reorder = PostProcessing(path, self.grid_occupied)
        path_reordering = post_processing_reorder.path_reordering(path)

        return path_reordering

    def post_processing_reconst(self, path): # for experiment (reconst)
        post_processing_reconst = PostProcessing(path, self.grid_occupied)
        path_reconst = post_processing_reconst.path_reconstructing(path)
        
        return path_reconst

    def post_processing_reorder_reconst(self, path): # for experiment (reorder+reconst)
        post_processing_reorder_reconst = PostProcessing(path, self.grid_occupied)
        path_reordering = post_processing_reorder_reconst.path_reordering(path)
        path_reorder_reconst = post_processing_reorder_reconst.path_reconstructing(path_reordering)

        return path_reorder_reconst

    def post_processing_los(self, path): # for experiment (LOS)
        post_processing_los = PostProcessingBaseline(path, self.grid_occupied)
        path_los = post_processing_los.path_reconstructing(path)

        return path_los

    def post_processing_pnp(self, path): # for experiment (PNP)
        post_processing_pnp = PostProcessingBaseline(path, self.grid_occupied)
        path_pnp = post_processing_pnp.parent_node_passing(path)

        return path_pnp

    def post_processing_remove_reconst_pnp(self, path):
        post_processing_remove_reconst = PostProcessing(path, self.grid_occupied)
        post_processing_pnp = PostProcessingBaseline(path, self.grid_occupied)

        path_remove = post_processing_remove_reconst.remove_redundant_nodes(path)
        path_remove_reconst = post_processing_remove_reconst.path_reconstructing(path_remove)
        path_remove_reconst_pnp = post_processing_pnp.parent_node_passing(path_remove_reconst)

        return path_remove_reconst_pnp

    def post_processing(self, path): 
        post_processing = PostProcessing(path, self.grid_occupied)
        path_reordering = post_processing.path_reordering(path)
        path_reorder_reconst = post_processing.path_reconstructing(path_reordering)

        post_processing_pnp = PostProcessingBaseline(path_reorder_reconst, self.grid_occupied)
        path_reorder_reconst_pnp = post_processing_pnp.parent_node_passing(path_reorder_reconst)

        return path_reorder_reconst_pnp

    def path_index_to_path_point(self, path):
        path_points = []
        for point in path:
            path_points.append(self.grid_points[point[0], point[1], point[2]])
        return path_points

    def path_interpolation_grid(self, path_indices):

        def bresenham_line(start, end):
            """Perform Bresenham's line to get all the grid points along the line from start to end."""
            x0, y0, z0 = start
            x1, y1, z1 = end
            points = []

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            dz = abs(z1 - z0)
            sx = 1 if x1 > x0 else -1
            sy = 1 if y1 > y0 else -1
            sz = 1 if z1 > z0 else -1

            if dx >= dy and dx >= dz:
                yd = 2 * dy - dx
                zd = 2 * dz - dx
                while x0 != x1:
                    points.append((x0, y0, z0))
                    if yd >= 0:
                        y0 += sy
                        yd -= 2 * dx
                    if zd >= 0:
                        z0 += sz
                        zd -= 2 * dx
                    yd += 2 * dy
                    zd += 2 * dz
                    x0 += sx
            elif dy >= dx and dy >= dz:
                xd = 2 * dx - dy
                zd = 2 * dz - dy
                while y0 != y1:
                    points.append((x0, y0, z0))
                    if xd >= 0:
                        x0 += sx
                        xd -= 2 * dy
                    if zd >= 0:
                        z0 += sz
                        zd -= 2 * dy
                    xd += 2 * dx
                    zd += 2 * dz
                    y0 += sy
            else:
                xd = 2 * dx - dz
                yd = 2 * dy - dz
                while z0 != z1:
                    points.append((x0, y0, z0))
                    if xd >= 0:
                        x0 += sx
                        xd -= 2 * dz
                    if yd >= 0:
                        y0 += sy
                        yd -= 2 * dz
                    xd += 2 * dx
                    yd += 2 * dy
                    z0 += sz

            points.append((x1, y1, z1))
            return points


        interpolated_path = []

        for i in range(1, len(path_indices)):
            p1 = path_indices[i - 1]
            p2 = path_indices[i]
            segment = bresenham_line(p1, p2)

            # Avoid duplicating points (except for first one)
            if i > 1:
                segment = segment[1:]

            interpolated_path.extend(segment)

        return interpolated_path


    def create_boundary(self, path_indices, interpolate_path=True):
        if interpolate_path:
            path_indices = self.path_interpolation_grid(path_indices)  # Interpolate in grid first
        path_points = self.path_index_to_path_point(path_indices)  # Convert after interpolation

        upper, lower = self.boundary.get_upper_lower_bounds(path_points)
        return path_points, upper, lower

    def path_smothing(self, path, upper, lower, weight=None):
        if weight is None:  # Use 'is' instead of '=' for None check
            self.minsnap = min_snap(path, upper, lower)
        else:
            self.minsnap = min_snap(path, upper, lower, weight=weight)

        p_x, p_y, p_z = self.minsnap.solve()

        # result = Result()
        # result.plot_path_with_smoothing(path, self.grid_occupied, p_x, p_y, p_z, self.minsnap.time_stamps, self.minsnap.k, self.minsnap.n)
        # result.plot_path_with_map_smoothing(path, self.grid_occupied, p_x, p_y, p_z, self.minsnap.time_stamps, self.minsnap.k, self.minsnap.n)

        return self.minsnap.get_waypoints()

    def get_differential(self):
        if hasattr(self, 'minsnap'):
            velocity = self.minsnap.get_velocity()
            acceleration = self.minsnap.get_acceleration()
            jerk = self.minsnap.get_jerk()
            snap = self.minsnap.get_snap()
            return velocity, acceleration, jerk, snap
        else:
            raise ValueError("Call path_smothing first to initialize minsnap.")

    def get_original_traj(self):
        if hasattr(self, 'minsnap'):
            return self.minsnap.get_reference_waypoints()
        else:
            raise ValueError("Call path_smothing first to initialize minsnap.")

    def get_original_differential(self):
        if hasattr(self, 'minsnap'):
            velocity = self.minsnap.get_reference_velocity()
            acceleration = self.minsnap.get_reference_acceleration()
            jerk = self.minsnap.get_reference_jerk()
            snap = self.minsnap.get_reference_snap()
            return velocity, acceleration, jerk, snap
        else:
            raise ValueError("Call path_smothing first to initialize minsnap.")

    def get_time_stamps(self):
        if hasattr(self, 'minsnap'):
            return self.minsnap.get_time_values()
        else:
            raise ValueError("Call path_smothing first to initialize minsnap.")


    # def path_interpolation(self, path, max_segment_percent=0.1):
    #     """
    #     Smooth the path using interpolation.
        
    #     Args:
    #         path: List of waypoints [(x, y, z), ...].
    #         max_segment_percent: Maximum allowed segment length as a percentage of the total path length.
        
    #     Returns:
    #         Smoothed and grid-aligned path with intermediate points.
    #     """
        
    #     def calculate_distance(p1, p2):
    #         return np.linalg.norm(np.array(p2) - np.array(p1))
        
    #     def interpolate_segment(p1, p2, num_segments):
    #         x_vals = np.linspace(p1[0], p2[0], num_segments + 1)
    #         y_vals = np.linspace(p1[1], p2[1], num_segments + 1)
    #         z_vals = np.linspace(p1[2], p2[2], num_segments + 1)
    #         return list(zip(x_vals, y_vals, z_vals))
    
    #     total_length = sum(calculate_distance(path[i], path[i+1]) for i in range(len(path)-1))
    #     max_segment_length = total_length * max_segment_percent
        
    #     interpolated_path = [path[0]]  
        
    #     for i in range(1, len(path)):
    #         p1 = path[i - 1]
    #         p2 = path[i]
            
    #         distance = calculate_distance(p1, p2)
            
    #         # If the segment is too long, interpolate and add points
    #         if distance > max_segment_length:
    #             num_segments = int(np.ceil(distance / max_segment_length))
    #             interpolated = interpolate_segment(p1, p2, num_segments)
    #             indices_points = [self.get_indices(p) for p in interpolated]
    #             if np.any([self.grid_occupied[tuple(i)] for i in indices_points]):
    #                 print(f"Trigger")
    #                 raise ValueError("Some interpolated points are inside obstacles!")
    #             interpolated_points = self.path_index_to_path_point(indices_points)
    #             interpolated_path.extend(interpolated_points[1:-1] + [p2])

    #         else:
    #             # Otherwise, just add the original point
    #             interpolated_path.append(p2)
        
    #     return interpolated_path

    # def create_boundary(self, path_indices, interpolate_path = True):
    #     path_points = self.path_index_to_path_point(path_indices)
    #     if interpolate_path:
    #         path_points = self.path_interpolation(path_points)
    #     upper, lower = self.boundary.get_upper_lower_bounds(path_points) 
    #     return path_points, upper, lower
