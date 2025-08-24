"""
Disclaimer:
- This implementation may not be fully optimized and is primarily intended for prototyping purposes.
"""

import math

from collections import deque

class PostProcessingBaseline:
    def __init__(self, path, grid):
        self.ori_path = path 
        self.grid = grid    
        self.g_score = {}
        self.start = self.ori_path[0]
        self.goal = self.ori_path[-1]


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

    def euclidean_distance(self, node1, node2):
        """Euclidean distance heuristic for diagonal movement."""
        dx = abs(node1[0] - node2[0])
        dy = abs(node1[1] - node2[1])
        dz = abs(node1[2] - node2[2])
        return round(math.sqrt(dx**2 + dy**2 + dz**2), 5)

    def calculate_g_score(self, current, neighbor):
        """Calculate the g_score from the current node to  a neighbor in a 3D grid."""
        
        # Extract coordinates for clarity
        x1, y1, z1 = current
        x2, y2, z2 = neighbor
        
        # Check for orthogonal moves along any axis
        if x1 == x2 and y1 == y2:  # Movement along the z-axis
            return round(self.g_score[current] + abs(z2 - z1), 5)
        elif x1 == x2 and z1 == z2:  # Movement along the y-axis
            return round(self.g_score[current] + abs(y2 - y1), 5)
        elif y1 == y2 and z1 == z2:  # Movement along the x-axis
            return round(self.g_score[current] + abs(x2 - x1), 5)
        
        # Check for diagonal moves
        # Diagonal in two dimensions
        elif x1 == x2 or y1 == y2 or z1 == z2:
            return round(self.g_score[current] + (math.sqrt(2) * max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1))), 5)
        
        # Diagonal in three dimensions
        else:
            return round(self.g_score[current] + (math.sqrt(3) * max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1))), 5)


    def initialize_g_scores(self, pathlist):
        """Initialize and calculate the g_score for every node in the pathlist."""
                
        # Initialize the g_score dictionary
        self.g_score = {}
        self.g_score[pathlist[0]] = 0  # Starting node has a g_score of 0
        
        # Iterate through the pathlist starting from the second element
        for i in range(len(pathlist)-1):
            current = pathlist[i]
            neighbor = pathlist[i+1]
            
            # Calculate the g_score for the neighbor
            self.g_score[neighbor] = self.calculate_g_score(current, neighbor)

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


    def reconstruct_path(self, came_from):
        """Reconstruct the path from start to goal."""
        path = deque()
        current = self.goal
        while current in came_from:
            path.appendleft(current)
            current = came_from[current]
        path.appendleft(self.start)
        return list(path)

    def parent_node_passing(self, path=None):
        if path is None:
            pathlist = self.ori_path[:]  # Copy of the original path
        else:
            pathlist = path[:]
            self.start = pathlist[0]
            self.goal = pathlist[-1]

        self.initialize_g_scores(pathlist) # initialize the g_score

        if len(pathlist) > 2:
            closelist = [] #TODO change to set()
            closelist_set = set()
            parentnode = pathlist[0]
            parentnode_index = 0

            came_from = {pathlist[i]: pathlist[0] for i in range(1, len(pathlist))} # all node have direct visibility with the initial node by default

            for i in range(1, len(pathlist)):
                xi = pathlist[i]

                if self.is_path_passable(parentnode, xi)[0]: # Direct connect from parent node to xi

                    # Change parent node of xi to be parent node
                    came_from[xi] = parentnode
                    
                    # add nodes between them to closelist
                    for node in pathlist[(parentnode_index+1):i]:
                        if node not in closelist_set:  # Check if node is already in the set
                            closelist.append(node)     # Add to the list if not already in closelist
                            closelist_set.add(node)    # Add the node to the set to track it

                    # update g-score
                    self.g_score[xi] = self.g_score[parentnode] + self.euclidean_distance(parentnode, xi)

                else: 
                    flag = False
    
                    for j in range(len(closelist)):
                        cj = closelist[j]
 
                        if (self.is_path_passable(cj, xi)[0] and
                                (tentative_g_score := (self.euclidean_distance(cj, xi) + self.g_score[cj])) < self.g_score[xi]):
                            # Change parent node of xi to be cj
                            came_from[xi] = cj

                            # add nodes between them to closelist
                            cj_index = pathlist.index(cj)
                            for node in pathlist[cj_index:i]:
                                if node not in closelist_set:  # Check if node is already in the set
                                    closelist.append(node)     # Add to the list if not already in closelist
                                    closelist_set.add(node)    # Add the node to the set to track it

                            # update g-score
                            self.g_score[xi] = tentative_g_score

                            flag = True

                    if not flag:
                        # Set the parent of xi to be xi-1
                        came_from[xi] = pathlist[i - 1]
                
                parentnode = came_from[xi]
        else:
            return pathlist

        return self.reconstruct_path(came_from)

    def path_reconstructing(self, path=None):
        if path is None:
            reconstructed_path = self.ori_path[:]  # Copy of the original path
        else:
            reconstructed_path = path[:] # Copy of the input path

        if len(reconstructed_path) < 3:
            return reconstructed_path

        self.initialize_g_scores(reconstructed_path) # initialize the g_score

        i = 0

        while i < len(reconstructed_path) - 2:

            current_node = reconstructed_path[i]
            old_candidate = reconstructed_path[i+1]
            self.g_score[old_candidate] = self.g_score[current_node] + self.euclidean_distance(current_node, old_candidate)
            temp_reconstructed_path = reconstructed_path[:]

            # Candidate sub-paths from the current voxel to future voxels
            candidates = reconstructed_path[(i+2):]
            candidate_index = i+2

            # Evaluate candidates based on collision-free
            for candidate in candidates:
                # current_h_score = self.euclidean_distance(candidate, reconstructed_path[-1]) 
                current_cost = self.g_score[old_candidate] + self.euclidean_distance(old_candidate, candidate) 

                if self.is_path_passable(current_node, candidate)[0]:
                    shortcut_cost = self.g_score[current_node] + self.euclidean_distance(current_node, candidate)

                    if shortcut_cost <= current_cost:
                        temp_reconstructed_path = reconstructed_path[:(i+1)] + reconstructed_path[(candidate_index):]
                        current_cost = shortcut_cost

                if self.g_score[candidate] != current_cost:
                    self.g_score[candidate] = current_cost

                old_cost = current_cost
                candidate_index += 1

            reconstructed_path = temp_reconstructed_path
            
            i += 1

        return reconstructed_path