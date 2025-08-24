"""
This code is a Python adaptation of the minimum snap trajectory generation originally provided in MATLAB by 
https://github.com/symao/minimum_snap_trajectory_generation.
It also incorporates concepts and structure from `fixed_time_stamps_minsnap.py` available at:
https://github.com/IllierZer/Trajectory_Generation/blob/main/fixed_time_stamps_minsnap.py.


Disclaimer:
- This implementation may not be fully optimized and is primarily intended for prototyping purposes.
"""

import numpy as np
import matplotlib.pyplot as plt

from math import factorial, degrees, atan2 
from qpsolvers import solve_qp
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

from scipy.interpolate import interp1d

# Class definition for solving the minimum snap trajectory optimization problem
class min_snap:
    def __init__(self, parameters, upper_bound=None, lower_bound=None, total_time=10, weight=100):
        """
        Initialize the min_snap class with velocity and waypoints.
        Parameters:
        - velocity: The constant velocity to be followed.
        - parameters: List of waypoints in the form [(x1, y1, z1), (x2, y2, z2), ...].
        """
        self.solver = 'osqp'  # Solver to be used for quadratic programming
        self.num_points = len(parameters)  # Number of waypoints
        self.k = self.num_points - 1  # Number of trajectory segments
        self.n = 7  # Polynomial degree (7th order polynomial for smooth trajectory)
        self.lamda = weight
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.time = total_time
        
        # Extract x, y, z coordinates from waypoints
        self.get_points(parameters)

        # Extract boundary of Intermediate waypoints
        if self.upper_bound is not None and self.lower_bound is not None:
            self.get_bounds(self.upper_bound, self.lower_bound)

        self.parameters = parameters
        self.time_stamps = self.get_time_stamps(self.time)  # Compute time stamps for each segment

        # Cost function
        self.Q = self.get_Q()  # Quadratic cost matrix for minimum snap
        self.q = np.zeros(((self.n + 1) * self.k, ))  # Linear term in the cost function
        if self.upper_bound is not None and self.lower_bound is not None:
            self.Q_x, self.q_x = self.add_guiding_pos(self.Q, self.q, self.x)
            self.Q_y, self.q_y = self.add_guiding_pos(self.Q, self.q, self.y)
            self.Q_z, self.q_z = self.add_guiding_pos(self.Q, self.q, self.z)

        # Equality constraints
        self.A = self.get_A()  # Equality constraint matrix
        self.b_x, self.b_y, self.b_z = self.get_b()  # Equality constraint vectors for x, y, z

        # Inequality constraints
        if self.upper_bound is None and self.lower_bound is None:
            self.G = np.zeros((4 * self.k + 2, (self.n + 1) * self.k))  # Inequality constraint matrix
            self.h = np.zeros((4 * self.k + 2, ))  # Inequality constraint vector
        else:
            self.G = self.get_G()
            self.h_x, self.h_y, self.h_z = self.get_h()

        if self.upper_bound is not None and self.lower_bound is not None: # remove position constraints
            self.A = np.vstack((self.A[:3], self.A[self.k+2:]))
            self.b_x = np.concatenate((self.b_x[:3], self.b_x[self.k+2:]))
            self.b_y = np.concatenate((self.b_y[:3], self.b_y[self.k+2:]))
            self.b_z = np.concatenate((self.b_z[:3], self.b_z[self.k+2:]))


    def get_points(self, parameters):
        """
        Extract x, y, z coordinates from the list of waypoints.
        """
        x, y, z = [], [], []
        for point in parameters:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        self.x, self.y, self.z = x, y, z

    def get_bounds(self, upper_bound, lower_bound):
        """
        Extract upper and lower x, y, z bounds from the provided inputs.
        """
        upper_x, upper_y, upper_z = [], [], []
        lower_x, lower_y, lower_z = [], [], []

        for u_point, l_point in zip(upper_bound, lower_bound):
            upper_x.append(u_point[0])
            upper_y.append(u_point[1])
            upper_z.append(u_point[2])
            
            lower_x.append(l_point[0])
            lower_y.append(l_point[1])
            lower_z.append(l_point[2])
        
        self.upper_x, self.upper_y, self.upper_z = upper_x, upper_y, upper_z
        self.lower_x, self.lower_y, self.lower_z = lower_x, lower_y, lower_z

    def get_time_stamps(self, total_time):
        distances = [
            np.linalg.norm(np.array(self.parameters[i+1]) - np.array(self.parameters[i]))
            for i in range(self.k)
        ]

        scaling_factor = total_time / sum(distances)
        scaled_times = [d * scaling_factor for d in distances]

        timestamps = [0.01]
        for t in scaled_times:
            timestamps.append(timestamps[-1] + t)

        return timestamps

    def computeQ(self, n, r, t1, t2):
        """
        Compute the quadratic cost matrix Q for a given polynomial order n and derivative order r.
        
        Parameters:
        - n: int
            Polynomial order. Determines the degree of the polynomial.
        - r: int
            Derivative order for optimization. 
        - t1: float
            Start timestamp for the polynomial.
        - t2: float
            End timestamp for the polynomial.
        
        Returns:
        - Q: np.ndarray
            Quadratic cost matrix for the polynomial coefficients.
        """
        T = np.zeros((n - r) * 2 + 1)
        for i in range((n - r) * 2 + 1):
            T[i] = t2**(i + 1) - t1**(i + 1) 

        Q = np.zeros((n + 1, n + 1)) 
        for i in range(r, n + 1): 
            for j in range(i, n + 1):
                k1 = i - r
                k2 = j - r
                k = k1 + k2 + 1
                Q[i, j] = (
                    np.prod(np.arange(k1 + 1, k1 + r + 1)) *
                    np.prod(np.arange(k2 + 1, k2 + r + 1)) /
                    k * T[k - 1]
                )
                Q[j, i] = Q[i, j]  
        return Q

    def get_Q(self):
        """
        Construct the quadratic cost matrix Q for the minimum snap objective.
        """
        Q_all = []
        n = self.n  # Polynomial order
        t = self.time_stamps

        for l in range(self.k):
            Q_all.append(self.computeQ(n, 3, t[l], t[l + 1]))  # Compute Q for each segment

        # Combine all segment matrices into a block diagonal matrix
        Q = block_diag(*Q_all)
        Q += 1e-12 * np.identity(Q.shape[0])  # Add a small value for positive definiteness
        return Q

    def add_guiding_pos(self, Q, q, waypoints):
        H_all = []
        q_all = []
        t = self.time_stamps
        n = self.n # Polynomial order

        for l in range(self.k):
            t1 = t[l]
            t2 = t[l+1]
            p1 = waypoints[l]
            p2 = waypoints[l+1]
            a1 = (p2 - p1) / (t2 - t1)
            a0 = p1 - a1 * t1
            ci = np.zeros((n + 1, ))
            ci[0:2] = [a0, a1]
            Qi = self.computeQ(n, 0, t1, t2)
            qi = -Qi.T @ ci
            q_all.extend(qi)
            H_all.append(Qi)

        H = block_diag(*H_all)
        H += 1e-12 * np.identity(H.shape[0])
        Q = self.Q + self.lamda * H
        q = self.q + self.lamda * np.array(q_all)

        return Q, q

    def get_A(self):
        """
        Construct the equality constraint matrix A for the trajectory.
        """
        k = self.k
        n = self.n
        A = np.zeros((4 * k + 2, k * (n + 1)))
        t = self.time_stamps
        
        # Start point constraints
        for j in range(n + 1):
            A[0][j] = t[0]**j # position constraint of start
            A[1][j] = j * t[0]**(j - 1) # velocity constraint of start
            A[2][j] = j * (j - 1) * t[0]**(j - 2) # acceleration constraint of start

        # Intermediate waypoints
        for i in range(1, k):
            for l in range(n + 1):
                A[i + 2][(i - 1) * (n + 1) + l] = t[i]**l

        # End point constraints
        for j in range((k - 1) * (n + 1), k * (n + 1)):
            r = j - (k - 1) * (n + 1)
            A[k + 2][j] = t[k]**r # position constraint of end
            A[k + 3][j] = r * t[k]**(r - 1) # velocity constraint of end
            A[k + 4][j] = r * (r - 1) * t[k]**(r - 2) # acceleration constraint of end

        # Continuity constraints (position, velocity, acceleration)
        for i in range(k - 1):
            for l in range(2 * n + 2):
                if l < (n + 1):
                    A[k + 5 + 3 * i][(n + 1) * i + l] = t[i + 1]**l
                    A[k + 6 + 3 * i][(n + 1) * i + l] = l * t[i + 1]**(l - 1)
                    A[k + 7 + 3 * i][(n + 1) * i + l] = l * (l - 1) * t[i + 1]**(l - 2)
                else:
                    A[k + 5 + 3 * i][(n + 1) * i + l] = -t[i + 1]**(l - (n + 1))
                    A[k + 6 + 3 * i][(n + 1) * i + l] = -(l - (n + 1)) * t[i + 1]**((l - (n + 1)) - 1)
                    A[k + 7 + 3 * i][(n + 1) * i + l] = -(l - (n + 1)) * ((l - (n + 1)) - 1) * t[i + 1]**((l - (n + 1)) - 2)
        return A

    def get_b(self):
        """
        Construct the equality constraint vectors b_x, b_y, b_z for x, y, z coordinates.
        """
        points = [self.x, self.y, self.z]
        b = []
        for i in range(3):
            bi = np.array([points[i][0], 0.0, 0.0])
            last_point = np.array([points[i][self.num_points - 1], 0.0, 0.0])
            bi = np.append(bi, points[i][1:(self.num_points - 1)])
            bi = np.append(bi, last_point)
            bi = np.append(bi, np.zeros((3 * (self.k - 1))))
            b.append(bi)
        return b

    def get_G(self):
        k = self.k
        n = self.n
        t = self.time_stamps
        G = np.zeros((2 * (k - 1) , (n + 1) * k))
        for i in range(1, k):
            for l in range(n + 1):
                G[2 * (i - 1)][(i - 1) * (n + 1) + l] = t[i]**l        # for upper bound
                G[2 * (i - 1) + 1][(i - 1) * (n + 1) + l] = -t[i]**l   # for lower bound
        return G
        
    def get_h(self):
        k = self.k
        upper_bound = [self.upper_x, self.upper_y, self.upper_z]
        lower_bound = [self.lower_x, self.lower_y, self.lower_z] 
        h = []
        for i in range(3):
            hi = []
            for j in range(k - 1):
                hi.extend([upper_bound[i][j], -1*lower_bound[i][j]])
            h.append(np.array(hi))
        return h


    def solve(self):
        """
        Solve the quadratic programming problem for x, y, and z coordinates.
        """  
        if self.upper_bound is None and self.lower_bound is None:
            # Convert matrices to sparse format if they are dense and have many zeros
            self.Q = csc_matrix(self.Q)  # Sparse CSC format for matrix Q
            self.G = csc_matrix(self.G)  # Sparse CSC format for matrix G
            self.A = csc_matrix(self.A)  # Sparse CSC format for matrix A

            self.p_x = solve_qp(self.Q, self.q, self.G, self.h, self.A, self.b_x, solver=self.solver)
            self.p_y = solve_qp(self.Q, self.q, self.G, self.h, self.A, self.b_y, solver=self.solver)
            self.p_z = solve_qp(self.Q, self.q, self.G, self.h, self.A, self.b_z, solver=self.solver)
        else:
            # Convert matrices to sparse format if they are dense and have many zeros
            self.Q_x = csc_matrix(self.Q_x)  # Sparse CSC format for matrix Q
            self.Q_y = csc_matrix(self.Q_y)  # Sparse CSC format for matrix Q
            self.Q_z = csc_matrix(self.Q_z)  # Sparse CSC format for matrix Q
            self.G = csc_matrix(self.G)  # Sparse CSC format for matrix G
            self.A = csc_matrix(self.A)  # Sparse CSC format for matrix A

            self.p_x = solve_qp(self.Q_x, self.q_x, self.G, self.h_x, self.A, self.b_x, solver=self.solver)
            self.p_y = solve_qp(self.Q_y, self.q_y, self.G, self.h_y, self.A, self.b_y, solver=self.solver)
            self.p_z = solve_qp(self.Q_z, self.q_z, self.G, self.h_z, self.A, self.b_z, solver=self.solver)

        return self.p_x, self.p_y, self.p_z

    def get_waypoints(self, time_resolution=100):

        waypoints = [] 
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z

        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            for j in range(time_resolution):
                x_term, y_term, z_term = 0, 0, 0
                for l in range((self.n + 1) * i, (self.n + 1) * (i + 1)):
                    x_term += p_x[l] * (t[j]**(l - (self.n + 1) * i))
                    y_term += p_y[l] * (t[j]**(l - (self.n + 1) * i))
                    z_term += p_z[l] * (t[j]**(l - (self.n + 1) * i))
                
                waypoint = [x_term , y_term, z_term] 
                waypoints.append(waypoint)  

        return np.array(waypoints)

    def get_velocity(self, time_resolution=100):
        velocities = []  
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z

        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            for j in range(time_resolution):
                vx_term, vy_term, vz_term = 0, 0, 0
                for l in range((self.n + 1) * i, (self.n + 1) * (i + 1)):
                    power = l - (self.n + 1) * i
                    if power > 0:  # Derivative of t^power is power * t^(power-1)
                        vx_term += p_x[l] * power * (t[j]**(power - 1))
                        vy_term += p_y[l] * power * (t[j]**(power - 1))
                        vz_term += p_z[l] * power * (t[j]**(power - 1))
                
                velocity = [vx_term, vy_term, vz_term]
                velocities.append(velocity)

        return np.array(velocities)

    def get_acceleration(self, time_resolution=100):
        accelerations = []
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z

        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            for j in range(time_resolution):
                ax_term, ay_term, az_term = 0, 0, 0
                for l in range((self.n + 1) * i, (self.n + 1) * (i + 1)):
                    power = l - (self.n + 1) * i
                    if power > 1:  # Second derivative
                        ax_term += p_x[l] * power * (power - 1) * (t[j]**(power - 2))
                        ay_term += p_y[l] * power * (power - 1) * (t[j]**(power - 2))
                        az_term += p_z[l] * power * (power - 1) * (t[j]**(power - 2))

                acceleration = [ax_term, ay_term, az_term]
                accelerations.append(acceleration)

        return np.array(accelerations)

    def get_jerk(self, time_resolution=100):
        jerks = []
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z

        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            for j in range(time_resolution):
                jx_term, jy_term, jz_term = 0, 0, 0
                for l in range((self.n + 1) * i, (self.n + 1) * (i + 1)):
                    power = l - (self.n + 1) * i
                    if power > 2:  # Third derivative
                        jx_term += p_x[l] * power * (power - 1) * (power - 2) * (t[j]**(power - 3))
                        jy_term += p_y[l] * power * (power - 1) * (power - 2) * (t[j]**(power - 3))
                        jz_term += p_z[l] * power * (power - 1) * (power - 2) * (t[j]**(power - 3))

                jerk = [jx_term, jy_term, jz_term]
                jerks.append(jerk)

        return np.array(jerks)

    def get_snap(self, time_resolution=100):
        snaps = []
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z

        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            for j in range(time_resolution):
                sx_term, sy_term, sz_term = 0, 0, 0
                for l in range((self.n + 1) * i, (self.n + 1) * (i + 1)):
                    power = l - (self.n + 1) * i
                    if power > 3:  # Fourth derivative
                        sx_term += p_x[l] * power * (power - 1) * (power - 2) * (power - 3) * (t[j]**(power - 4))
                        sy_term += p_y[l] * power * (power - 1) * (power - 2) * (power - 3) * (t[j]**(power - 4))
                        sz_term += p_z[l] * power * (power - 1) * (power - 2) * (power - 3) * (t[j]**(power - 4))

                snap = [sx_term, sy_term, sz_term]
                snaps.append(snap)

        return np.array(snaps)

    def get_time_values(self, time_resolution=100):
        time_values = []
        
        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            time_values.extend(t)  # Collect all t[j] values

        return np.array(time_values)



    def get_reference_waypoints(self, time_resolution=100, kind='linear'):
        """
        Return interpolated reference waypoints (based on original input waypoints),
        matching the shape and time resolution of get_waypoints().
        """
        # Interpolation functions for x, y, z
        fx = interp1d(self.time_stamps, self.x, kind=kind, fill_value="extrapolate")
        fy = interp1d(self.time_stamps, self.y, kind=kind, fill_value="extrapolate")
        fz = interp1d(self.time_stamps, self.z, kind=kind, fill_value="extrapolate")

        reference_waypoints = []

        for i in range(self.k):
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            x_vals = fx(t)
            y_vals = fy(t)
            z_vals = fz(t)
            for x, y, z in zip(x_vals, y_vals, z_vals):
                reference_waypoints.append([x, y, z])

        return np.array(reference_waypoints)

    def get_reference_velocity(self, time_resolution=100, kind='linear'):
        """
        Compute velocity for the interpolated reference waypoints.
        """
        fx = interp1d(self.time_stamps, self.x, kind=kind, fill_value="extrapolate")
        fy = interp1d(self.time_stamps, self.y, kind=kind, fill_value="extrapolate")
        fz = interp1d(self.time_stamps, self.z, kind=kind, fill_value="extrapolate")

        t_values = np.linspace(self.time_stamps[0], self.time_stamps[-1], self.k * time_resolution)
        
        # Compute first derivatives using finite differences
        vx = np.gradient(fx(t_values), t_values)
        vy = np.gradient(fy(t_values), t_values)
        vz = np.gradient(fz(t_values), t_values)

        return np.column_stack((vx, vy, vz))

    def get_reference_acceleration(self, time_resolution=100, kind='linear'):
        """
        Compute acceleration for the interpolated reference waypoints.
        """
        fx = interp1d(self.time_stamps, self.x, kind=kind, fill_value="extrapolate")
        fy = interp1d(self.time_stamps, self.y, kind=kind, fill_value="extrapolate")
        fz = interp1d(self.time_stamps, self.z, kind=kind, fill_value="extrapolate")

        t_values = np.linspace(self.time_stamps[0], self.time_stamps[-1], self.k * time_resolution)
        
        # Compute second derivatives using finite differences
        ax = np.gradient(np.gradient(fx(t_values), t_values), t_values)
        ay = np.gradient(np.gradient(fy(t_values), t_values), t_values)
        az = np.gradient(np.gradient(fz(t_values), t_values), t_values)

        return np.column_stack((ax, ay, az))

    def get_reference_jerk(self, time_resolution=100, kind='linear'):
        """
        Compute jerk (3rd derivative) for interpolated reference waypoints.
        """
        fx = interp1d(self.time_stamps, self.x, kind=kind, fill_value="extrapolate")
        fy = interp1d(self.time_stamps, self.y, kind=kind, fill_value="extrapolate")
        fz = interp1d(self.time_stamps, self.z, kind=kind, fill_value="extrapolate")

        t_values = np.linspace(self.time_stamps[0], self.time_stamps[-1], self.k * time_resolution)

        jx = np.gradient(np.gradient(np.gradient(fx(t_values), t_values), t_values), t_values)
        jy = np.gradient(np.gradient(np.gradient(fy(t_values), t_values), t_values), t_values)
        jz = np.gradient(np.gradient(np.gradient(fz(t_values), t_values), t_values), t_values)

        return np.column_stack((jx, jy, jz))

    def get_reference_snap(self, time_resolution=100, kind='linear'):
        """
        Compute snap (4th derivative) for interpolated reference waypoints.
        """
        fx = interp1d(self.time_stamps, self.x, kind=kind, fill_value="extrapolate")
        fy = interp1d(self.time_stamps, self.y, kind=kind, fill_value="extrapolate")
        fz = interp1d(self.time_stamps, self.z, kind=kind, fill_value="extrapolate")

        t_values = np.linspace(self.time_stamps[0], self.time_stamps[-1], self.k * time_resolution)

        sx = np.gradient(np.gradient(np.gradient(np.gradient(fx(t_values), t_values), t_values), t_values), t_values)
        sy = np.gradient(np.gradient(np.gradient(np.gradient(fy(t_values), t_values), t_values), t_values), t_values)
        sz = np.gradient(np.gradient(np.gradient(np.gradient(fz(t_values), t_values), t_values), t_values), t_values)

        return np.column_stack((sx, sy, sz))



    def plot(self, time_resolution):
        """
        Plot the trajectory in 3D space with specified time resolution for smoothness.
        """
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(self.x, self.y, self.z, 'b', marker='o')  # Plot waypoints
        self.solve()  # Solve the optimization problem
        p_x, p_y, p_z = self.p_x, self.p_y, self.p_z
        for i in range(self.k):
            x_segment, y_segment, z_segment = [], [], []
            t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], time_resolution)
            for j in range(time_resolution):
                x_term, y_term, z_term = 0, 0, 0
                for l in range((self.n + 1) * i, (self.n + 1) * (i + 1)):
                    x_term += p_x[l] * (t[j]**(l - (self.n + 1) * i))
                    y_term += p_y[l] * (t[j]**(l - (self.n + 1) * i))
                    z_term += p_z[l] * (t[j]**(l - (self.n + 1) * i))
                x_segment.append(x_term)
                y_segment.append(y_term)
                z_segment.append(z_term)
            ax.plot3D(x_segment, y_segment, z_segment, 'r')  # Plot trajectory
        plt.show()


def get_upper_lower(points, upper_bound=1, lower_bound=1):
    """
    Modify the points by adding a certain offset for the upper and lower bounds.
    Excludes the first and last points.
    
    upper_bound: The value to add to the current point to create the upper bound.
    lower_bound: The value to subtract from the current point to create the lower bound.
    """
    upper = []
    lower = []

    for i in range(1, len(points) - 1):  # Excluding first and last points
        # Extract current point
        x, y, z = points[i]
        
        # Add offset for upper bound and lower bound
        upper_point = (x + upper_bound, y + upper_bound, z + upper_bound)
        lower_point = (x - lower_bound, y - lower_bound, z - lower_bound)
        
        # Append the new points to respective lists
        upper.append(upper_point)
        lower.append(lower_point)

    return upper, lower
