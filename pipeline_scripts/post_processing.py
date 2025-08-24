"""
Disclaimer:
- This implementation may not be fully optimized and is primarily intended for prototyping purposes.
"""

import math
import numpy as np

class PostProcessing:
    def __init__(self, path, grid):
        self.ori_path = path
        self.grid = grid

    def is_in_bounds(self, x, y, z):
        """Check if the coordinates are within 3D grid bounds."""
        return (
            0 <= x < self.grid.shape[0]
            and 0 <= y < self.grid.shape[1]
            and 0 <= z < self.grid.shape[2]
        )

    def is_obstacle(self, x, y, z):
        """Check if the cell is an obstacle."""
        return self.grid[x, y, z] != 0

    def is_passable(self, x, y, z):
        """Check if the cell is within 3D bounds and not an obstacle."""
        return self.is_in_bounds(x, y, z) and not self.is_obstacle(x, y, z)

    def is_corner_cutting(self, node, direction):
        """
        Check for diagonal corner moves in 2D or 3D space.
        Parameters:
            - node: The current position (before the move)
            - direction: The direction of the next move
        """
        x, y, z = node
        dx, dy, dz = direction
        non_zero_components = sum(1 for component in direction if component != 0)
        adjacent_neighbors = []
        
        if non_zero_components == 2:  # 2D diagonal
            if dz == 0:  # Movement in the XY plane
                adjacent_neighbors = [(x + dx, y, z), (x, y + dy, z)]
            
            elif dx == 0:  # Movement in the YZ plane
                adjacent_neighbors = [(x, y + dy, z), (x, y, z + dz)]
            
            elif dy == 0:  # Movement in the XZ plane
                adjacent_neighbors = [(x + dx, y, z), (x, y, z + dz)]

        elif non_zero_components == 3:  # 3D diagonal
            adjacent_neighbors = [
            (x + dx, y, z),        # Neighbor in the X direction
            (x, y + dy, z),        # Neighbor in the Y direction
            (x + dx, y + dy, z),   # Neighbor in the XY plane
            (x + dx, y, z + dz),   # Neighbor in the XZ plane
            (x, y + dy, z + dz)    # Neighbor in the YZ plane
            ]

        for (ax, ay, az) in adjacent_neighbors:
            if not self.is_passable(ax, ay, az):
                return True

        return False

    def get_path(self, p1, p2):
        """Returns the direction (dx, dy, dz) from p1 to p2 in 3D."""
        return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    def sign_direction(self, v):
        """Returns the sign direction vector with components as -1, 0, or 1."""
        return (
            1 if v[0] > 0 else -1 if v[0] < 0 else 0,
            1 if v[1] > 0 else -1 if v[1] < 0 else 0,
            1 if v[2] > 0 else -1 if v[2] < 0 else 0
        )

    def is_collinear(self, v1, v2):
        """Check if two 3D vectors are collinear."""
        return (v1[0] * v2[1] == v1[1] * v2[0]) and \
            (v1[0] * v2[2] == v1[2] * v2[0]) and \
            (v1[1] * v2[2] == v1[2] * v2[1])

    def bresenham_line(self, start, end):
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

    def is_path_passable(self, start_node, end_node):
        """Check if the straight line from start_node to end_node passes through any obstacles or cuts corners."""
        line_points = self.bresenham_line(start_node, end_node)

        # Check each point along the line
        for i in range(len(line_points)):
            current_point = line_points[i]
            
            # Check if the current point is passable
            if not self.is_passable(*current_point):
                return False, line_points  # Hit an obstacle

            # Check for corner cutting only for the first n-1 points
            if i < len(line_points) - 1:
                next_point = line_points[i + 1]
                direction = self.sign_direction(self.get_path(current_point, next_point))
                if self.is_corner_cutting(current_point, direction):
                    return False, line_points  # Detected corner cutting

            # Check for another corner cutting only for the first n-2 points
            if i < len(line_points) - 2:
                next_next_point = line_points[i + 2]
                direction = self.sign_direction(self.get_path(current_point, next_next_point))
                if self.is_corner_cutting(current_point, direction):
                    return False, line_points  # Detected corner cutting

        return True, line_points  # All points are passable and no corner cutting detected

    def find_turning_angle(self, direction_1, direction_2):
        """Returns the turning angle in radians between two 3D directions."""
        # Convert sign directions to numpy arrays
        v1 = np.array(direction_1)
        v2 = np.array(direction_2)
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0
        
        # Calculate angle in radians and convert to degrees
        angle_rad = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
        # angle_deg = np.degrees(angle_rad) # convert to degrees
        
        return round(angle_rad, 5)

    def path_reordering(self, path=None):
        if path is None:
            reordered_path = self.ori_path[:]  # Copy of the original path
        else:
            reordered_path = path[:] # Copy of the input path

        # Initial direction calculations
        n = 0

        less_than_3_paths = False

        while n + 3 < len(reordered_path):

            current_path = self.get_path(reordered_path[n], reordered_path[n+1])
            next_path = self.get_path(reordered_path[n+1], reordered_path[n+2])
            future_path = self.get_path(reordered_path[n+2], reordered_path[n+3])

            # remove redundant node
            while self.is_collinear(current_path, next_path) or self.is_collinear(next_path, future_path):
                if self.is_collinear(next_path, future_path):
                    removed_node = reordered_path.pop(n+2)
                if self.is_collinear(current_path, next_path):
                    removed_node = reordered_path.pop(n+1)
                    
                if n + 3 >= len(reordered_path):
                    less_than_3_paths = True
                    break
                    
                # re-calculate the paths
                current_path = self.get_path(reordered_path[n], reordered_path[n+1])
                next_path = self.get_path(reordered_path[n+1], reordered_path[n+2])
                future_path = self.get_path(reordered_path[n+2], reordered_path[n+3])

            if less_than_3_paths:
                break
                        
            switch_flag = False

            # check is it better to switch the paths 
            if n == 0: ## special case (the first path direction) 
                ## switch curerent with next direction
                if self.is_collinear(current_path, future_path) or self.find_turning_angle(current_path, future_path) < self.find_turning_angle(next_path, future_path):
                    new_next_node_sp = (reordered_path[n][0] + next_path[0], reordered_path[n][1] + next_path[1], reordered_path[n][2] + next_path[2])
                    new_future_node_sp = (new_next_node_sp[0] + current_path[0], new_next_node_sp[1] + current_path[1], new_next_node_sp[2] + current_path[2])

                    # check is it possible to switch the paths or not
                    if self.is_path_passable(reordered_path[n], new_next_node_sp)[0] and self.is_path_passable(new_next_node_sp, new_future_node_sp)[0]:
                        switch_flag = True
                        reordered_path[n+1] = new_next_node_sp
                        reordered_path[n+2] = new_future_node_sp

            if not switch_flag: 
                ## switch next direction with future direction
                if self.is_collinear(current_path, future_path) or self.find_turning_angle(current_path, future_path) < self.find_turning_angle(current_path, next_path):
                    new_next_node = (reordered_path[n+1][0] + future_path[0], reordered_path[n+1][1] + future_path[1], reordered_path[n+1][2] + future_path[2])
                    new_future_node = (new_next_node[0] + next_path[0], new_next_node[1] + next_path[1], new_next_node[2] + next_path[2])

                    # check is it possible to switch the paths or not
                    if self.is_path_passable(reordered_path[n+1], new_next_node)[0] and self.is_path_passable(new_next_node, new_future_node)[0]:
                        switch_flag = True
                        reordered_path[n+2] = new_next_node
                        reordered_path[n+3] = new_future_node

            if not switch_flag:
                n += 1

        if n +2 < len(reordered_path): # handle last 2 paths in case there is redundant node
            current_path = self.get_path(reordered_path[n], reordered_path[n+1])
            next_path = self.get_path(reordered_path[n+1], reordered_path[n+2])

            if self.is_collinear(current_path, next_path):
                removed_node = reordered_path.pop(n+1)

        return reordered_path

    def euclidean_distance(self, node1, node2):
        """Euclidean distance heuristic for diagonal movement."""
        dx = abs(node1[0] - node2[0])
        dy = abs(node1[1] - node2[1])
        dz = abs(node1[2] - node2[2])
        return round(math.sqrt(dx**2 + dy**2 + dz**2), 5)

    def find_intersect_point(self, line_1, line_2):
        set_1 = set(line_1)
        set_2 = set(line_2)

        intersection = set_1.intersection(set_2)

        # Convert the result back to a list
        intersection_points = list(intersection)

        # Check and print the intersection points
        if intersection_points:
            return intersection_points
        else:
            return None

    def find_symmetric_path(self, node_start, node_middle, node_end):
        # Get direction vectors from the nodes
        path_middle_to_end = self.get_path(node_middle, node_end)

        # Calculate the new middle node by reversing the order of movement
        new_x_middle = node_start[0] + path_middle_to_end[0]
        new_y_middle = node_start[1] + path_middle_to_end[1]
        new_z_middle = node_start[2] + path_middle_to_end[2]
        new_node_middle = (new_x_middle, new_y_middle, new_z_middle)

        if self.is_path_passable(node_start, new_node_middle)[0] and self.is_path_passable(new_node_middle, node_end)[0]:
            return new_node_middle
        else:
            return None

    def create_pairs(self, list1, list2):
        # Exclude the first and last elements
        list1 = list1[:-1]
        list2 = list2[:-1]
   
        pairs = []
        
        len1, len2 = len(list1), len(list2)
        # Determine the shorter and longer list, but always keep list1 in front
        if len1 <= len2:
            short_list, long_list = list1, list2
        else:
            short_list, long_list = list2, list1

        long_idx = 0
        short_index_float = 0  # Floating-point index for the shorter list
        increment = len(short_list) / len(long_list)  # Increment for the shorter list
        
        while long_idx < len(long_list):
            # Floor the floating-point index to favor earlier elements
            short_idx = int(short_index_float)
            # Ensure the short index does not exceed bounds
            short_idx = min(short_idx, len(short_list) - 1)
            
            # Add the pair (preserve the order: list1 before list2)
            if len1 <= len2:
                pairs.append((short_list[short_idx], long_list[long_idx]))
            else:
                pairs.append((long_list[long_idx], short_list[short_idx]))
            
            # Increment indices
            short_index_float += increment
            long_idx += 1
        
        return pairs

    def find_first_shortcut_line(self, first_side, second_side): 
        """
        Attempt to connect a member of `first_side` with a member of `second_side`
        using exponential search followed by binary search.
        """
        # Make copies of start_side and end_side to avoid modifying the original lists
        first_side = first_side[:]
        second_side = second_side[:] 

        evaluate_pairs = self.create_pairs(first_side, second_side)

        #TODO: change to expo then binary (now it's linear)
        for pair in evaluate_pairs:
            if (passable := (result := self.is_path_passable(pair[0], pair[1]))[0]):
                return list(pair)

        return None

    def find_second_shortcut_line(self, first_side, second_side):
        """
        Attempt to connect the last member of `second_side` with a member of `first_side` 
        that satisfies the condition and is closest to the first member of `first_side`.
        """
        # Make copies of start_side and end_side to avoid modifying the original lists
        first_side = first_side[:]
        connect_point = second_side[-1]

        shortcut_node = None
        shortcut_path = []

        for i in range(len(first_side) - 1): #TODO: change to exponential search?
            eval_node = first_side[i]

            if (passable := (result := self.is_path_passable(eval_node, connect_point))[0]):
                # If path is passable return the shortcut node and shortcut path 
                shortcut_node = eval_node
                shortcut_path = result[1]
                return shortcut_node, shortcut_path

        return shortcut_node, shortcut_path

    def find_shortcut_node(self, node_start, ori_first_shortcut_node, ori_second_shortcut_node, node_end, shortcut_path, shortcut_path_length):
        """ 
        Attemp to find a shortcut node that is the intersect(if possible) point of 2 shortcut line
        if not compare which of the 2 shortcut is provide more shorther length
        """
        if node_start == ori_first_shortcut_node:
            evaluate_points_start_side = [node_start]
        else:
            evaluate_points_start_side = self.bresenham_line(node_start, ori_first_shortcut_node)

        if node_end == ori_second_shortcut_node:
            evaluate_points_end_side = [node_end]
        else:
            evaluate_points_end_side = self.bresenham_line(node_end, ori_second_shortcut_node)

        intermediate_shortcut_node = None
        shortcut_path = shortcut_path[:]
        shortcut_length = shortcut_path_length

        new_first_shortcut_node, new_first_shortcut_path = self.find_second_shortcut_line(evaluate_points_start_side, evaluate_points_end_side)
        new_second_shortcut_node, new_second_shortcut_path = self.find_second_shortcut_line(evaluate_points_end_side, evaluate_points_start_side)

        if new_first_shortcut_node is not None and new_second_shortcut_node is not None:
            new_intermediate_shortcut_node = self.find_intersect_point(new_first_shortcut_path, new_second_shortcut_path)

            if new_intermediate_shortcut_node is not None: # Found intersect point(s)
                new_intermediate_shortcut_node_list = new_intermediate_shortcut_node[:]

                for new_intermediate_shortcut_node in new_intermediate_shortcut_node_list: # for-loop in case of multiple intersect points 
                    shortcut_length_new = (
                        self.euclidean_distance(node_start, new_first_shortcut_node) + 
                        self.euclidean_distance(new_first_shortcut_node, new_intermediate_shortcut_node) + 
                        self.euclidean_distance(new_intermediate_shortcut_node, new_second_shortcut_node) + 
                        self.euclidean_distance(new_second_shortcut_node, node_end)
                    )

                    if shortcut_length_new < shortcut_length:
                        shortcut_length = shortcut_length_new
                        intermediate_shortcut_node = new_intermediate_shortcut_node
                        shortcut_path[0] = new_first_shortcut_node
                        shortcut_path[1] = new_second_shortcut_node

                return intermediate_shortcut_node, shortcut_path, shortcut_length

   
        if new_first_shortcut_node is not None: # Only 1 shortcut line or Not found intersect point  
            shortcut_length_new = (
                self.euclidean_distance(node_start, new_first_shortcut_node) + 
                self.euclidean_distance(new_first_shortcut_node, ori_second_shortcut_node) +
                self.euclidean_distance(ori_second_shortcut_node, node_end)
            )

            if shortcut_length_new < shortcut_length:
                shortcut_length = shortcut_length_new
                shortcut_path[0] = new_first_shortcut_node


        if new_second_shortcut_node is not None: # Only 1 shortcut line or Not found intersect point  
            shortcut_length_new = (
                self.euclidean_distance(node_start, ori_first_shortcut_node) + 
                self.euclidean_distance(ori_first_shortcut_node, new_second_shortcut_node) +
                self.euclidean_distance(new_second_shortcut_node, node_end)
            )
            if shortcut_length_new < shortcut_length:
                shortcut_length = shortcut_length_new
                shortcut_path[1] = new_second_shortcut_node


        return intermediate_shortcut_node, shortcut_path, shortcut_length

    def find_shortcut(self, node_start, node_middle, node_end):
        """
        Attempts to find the most efficient shortcut for the given path. 
        It evaluates potential shortcut paths and intermediate nodes to identify a better route.

        Process:
        1. Generate evaluation points for the default path and, if valid, the symmetric path.
        - The symmetric path is evaluated only when a valid symmetric middle node exists.
        2. For each set of evaluation points (default and symmetric, if applicable):
        - Identify the first valid shortcut line connecting the start and end sections.
        - Calculate the total length of the shortcut line and update the best path if this is the shortest so far.
        3. For each shortcut line, further attempt to find a valid shortcut node:
        - Calculate the total length of the shortcut node path and update the best length
        if it provides a shorter route than the current best shortcut node.
        4. Compare the results of the best shortcut line and the best shortcut node:
        - If the node-based shortcut is shorter, return the intermediate node.
        - Otherwise, return the shortcut line.
        - Return `None` if no valid shortcuts are found.

        Parameters:
        - node_start: The starting node of the path.
        - node_middle: The current middle node of the path.
        - node_end: The ending node of the path.

        Returns:
        - A list of nodes representing the shortcut path or a single intermediate node, or `None` if no shortcut exists.
        """
        evaluate_points_start_side = self.bresenham_line(node_start, node_middle)
        evaluate_points_end_side = self.bresenham_line(node_end, node_middle)

        symmetric_node_middle = self.find_symmetric_path(node_start, node_middle, node_end)

        if symmetric_node_middle is not None: # symmetric path is valid
            evaluate_points_start_side_symmetric = self.bresenham_line(node_start, symmetric_node_middle)
            evaluate_points_end_side_symmetric = self.bresenham_line(node_end, symmetric_node_middle)     
            
            evaluate_pairs = [
            [evaluate_points_start_side, evaluate_points_end_side],
            [evaluate_points_start_side_symmetric, evaluate_points_end_side_symmetric]
            ]

        else: # symmetric path is not valid
            evaluate_pairs = [
                [evaluate_points_start_side, evaluate_points_end_side]
            ]

        shortcut_path = []
        shortcut_node = []
        shortcut_node_path = [] # for a pair of node that connect to shortcut_node
        shortcut_path_length = float('inf')
        shortcut_node_length = float('inf')


        # attemp to find first shortcut line
        for i in range(len(evaluate_pairs)):
            if (current_shortcut_path := self.find_first_shortcut_line(evaluate_pairs[i][0], evaluate_pairs[i][1])) is not None:

                current_shortcut_path_length = (
                    self.euclidean_distance(node_start, current_shortcut_path[0]) +
                    self.euclidean_distance(current_shortcut_path[0], current_shortcut_path[1]) + 
                    self.euclidean_distance(current_shortcut_path[1], node_end) 
                    )

                if current_shortcut_path_length < shortcut_path_length:
                    shortcut_path_length = current_shortcut_path_length
                    shortcut_path = current_shortcut_path

                    # Attemp to find intermidiste node or better shortcut line
                    current_shortcut_node, current_shortcut_path, current_shortcut_length = self.find_shortcut_node(node_start, shortcut_path[0], shortcut_path[1], node_end, shortcut_path, shortcut_path_length)
                    if current_shortcut_node is not None: # found shortcut node
                        if current_shortcut_length < shortcut_node_length:
                            shortcut_node_length = current_shortcut_length
                            shortcut_node = current_shortcut_node
                            shortcut_node_path = current_shortcut_path

                    elif current_shortcut_length < shortcut_node_length: # found new shortcut path
                        shortcut_node_length = current_shortcut_length
                        shortcut_node_path = current_shortcut_path
    
        if shortcut_node:
            # print(f"return shortcut_node: {[shortcut_node_path[0], shortcut_node, shortcut_node_path[1]]} with length {shortcut_node_length}\n")
            return [shortcut_node_path[0], shortcut_node, shortcut_node_path[1]]

        elif shortcut_node_path:
            # print(f"return shortcut_node_path: {shortcut_node_path} with length {shortcut_node_length} \n")
            return shortcut_node_path

        elif shortcut_path:
            # print(f"return shortcut_path: {shortcut_path} with length {shortcut_path_length}\n")
            return shortcut_path

        else: # Not found shortcut
            return None

    def path_reconstructing(self, path=None):
        if path is None:
            reconstructed_path = self.ori_path[:]  # Copy of the original path
        else:
            reconstructed_path = path[:] # Copy of the input path

        if len(reconstructed_path) < 3:
            return reconstructed_path

        n = 0
        while n < len(reconstructed_path) - 2:
            node_start = reconstructed_path[n]
            node_middle = reconstructed_path[n + 1]
            node_end = reconstructed_path[n + 2]

            # Check if direct connection is possible
            if self.is_path_passable(node_start, node_end)[0]:
                reconstructed_path.pop(n + 1)  # Remove the middle node
                continue  # Skip to the next iteration with the updated path

            else: # Attempt to find a shortcut
                if (shortcut := self.find_shortcut(node_start, node_middle, node_end)) is not None:
                    new_shortcut = [node for node in shortcut if node != reconstructed_path[n] and node != reconstructed_path[n + 2]]
                    reconstructed_path = reconstructed_path[:n+1] + list(new_shortcut) + reconstructed_path[n+2:]

            n += 1

        return reconstructed_path

    def remove_redundant_nodes(self, path=None):
        if path is None:
            remove_redundant_path = self.ori_path[:]  # Copy of the original path
        else:
            remove_redundant_path = path[:]  # Copy of the input path

        i = 0
        while i < len(remove_redundant_path) - 2:
            v1, v2, v3 = remove_redundant_path[i], remove_redundant_path[i + 1], remove_redundant_path[i + 2]

            # Compute direction vectors
            d1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
            d2 = (v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2])

            # Check if the direction vectors are collinear
            if self.is_collinear(d1, d2):
                remove_redundant_path.pop(i + 1)  # Remove the middle point if collinear
            else:
                i += 1  # Move to the next point only if no removal

        return remove_redundant_path  # Return the modified path
        
    def path_smoothing(self):
        pass

    def run(self):
        pass