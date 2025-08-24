"""
Disclaimer:
- This implementation may not be fully optimized and is primarily intended for prototyping purposes.
"""


import heapq
import math
import numpy as np

from collections import deque

import time

class AStar:
    def __init__(self, start, goal, grid):
        self.SQRT_3_MINUS_SQRT_2 = math.sqrt(3) - math.sqrt(2)
        self.SQRT_2_MINUS_1 = math.sqrt(2) - 1
        self.start = start
        self.goal = goal
        self.grid = grid
        self.open_list = []
        self.closed_list = set()
        self.came_from = {}

        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}

        heapq.heappush(self.open_list, (self.f_score[start], start))

        self.neighbor_offsets = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

    def heuristic(self, node, goal):
        """Heuristic function using weighted distance for diagonal movement."""
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        dz = abs(node[2] - goal[2])

        # Determine d_max, d_min, and d_mid
        d_max = max(dx, dy, dz)
        d_min = min(dx, dy, dz)
        d_mid = dx + dy + dz - d_max - d_min  # Sum minus max and min gives the middle value

        # Calculate heuristic using the given formula
        return round(self.SQRT_3_MINUS_SQRT_2 * d_min + self.SQRT_2_MINUS_1 * d_mid + d_max, 5)

    def calculate_g_score(self, current, neighbor):
        """Calculate the g_score from the current node to a neighbor in a 3D grid."""
        
        # Extract coordinates for clarity
        x1, y1, z1 = current
        x2, y2, z2 = neighbor
        
        # Check for orthogonal moves along any axis
        if (x1 == x2 and y1 == y2) or (x1 == x2 and z1 == z2) or (y1 == y2 and z1 == z2):  # Movement along the z-axis, y-axis, x-axis
            return self.g_score[current] + 1

        # Diagonal in two dimensions
        elif x1 == x2 or y1 == y2 or z1 == z2:
            return round(self.g_score[current] + math.sqrt(2), 5)
        
        # Diagonal in three dimensions
        else:
            return round(self.g_score[current] + math.sqrt(3), 5)


    def update_scores(self, current, neighbor, tentative_g_score):
        """Update the g_score and f_score for a neighbor if a better path is found."""
        if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
            self.came_from[neighbor] = current
            self.g_score[neighbor] = tentative_g_score
            f_score = tentative_g_score + self.heuristic(neighbor, self.goal)
            self.f_score[neighbor] = f_score
            return f_score
        return None

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
        """Check if the cell is within bounds and not an obstacle."""
        return self.is_in_bounds(x, y, z) and not self.is_obstacle(x, y, z)

    def is_corner_cutting(self, node, direction): 
        """Check for diagonal corner moves in 2D or 3D space."""
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

    def get_neighbors(self, node):
        """Get the 26 neighbors of a node in 3D."""
        x, y, z = node
        valid_neighbors = []

        # Filter out neighbors that would result in corner-cutting and is not passable
        for dx, dy, dz in self.neighbor_offsets:
            neighbor = (x + dx, y + dy, z + dz)
            direction = (dx, dy, dz)
            if self.is_passable(*neighbor) and not self.is_corner_cutting(node, direction):
                valid_neighbors.append(neighbor)
        
        return valid_neighbors


    def reconstruct_path(self):
        """Reconstruct the path from start to goal."""
        path = deque()
        current = self.goal
        while current in self.came_from:
            path.appendleft(current)
            current = self.came_from[current]
        path.appendleft(self.start)
        return list(path)

    def run(self):
        """Perform the A* search algorithm."""
        open_list_nodes = set()  # Track nodes in open_list to avoid repeated list comprehension
        while self.open_list:
            f_score, current = heapq.heappop(self.open_list)

            if current == self.goal:
                return self.reconstruct_path(), list(self.closed_list), list(open_list_nodes)

            self.closed_list.add(current)
            open_list_nodes.discard(current)

            for neighbor in self.get_neighbors(current):

                if neighbor in self.closed_list:
                    continue

                # Calculate the cost (g_score)
                tentative_g_score = self.calculate_g_score(current, neighbor)

                # Update score and add to open list if necessary
                f_score = self.update_scores(current, neighbor, tentative_g_score)
                if f_score is not None and neighbor not in open_list_nodes:
                    heapq.heappush(self.open_list, (f_score, neighbor))
                    open_list_nodes.add(neighbor)

        return [], list(self.closed_list), list(open_list_nodes) # No path found
