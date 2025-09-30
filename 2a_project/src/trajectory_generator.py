import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt
import math
from environment import Environment3D

class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using splines
    Complete implementation with velocity and acceleration profiles
    """
    
    def __init__(self, waypoints, environment=None):
        self.waypoints = np.array(waypoints)
        self.trajectory_duration = None  # seconds
        self.max_velocity = None  # m/s
        self.max_acceleration = None  # m/s^2
        self.environment = environment  # Environment3D object for collision checking

    def _check_trajectory_collisions(self, trajectory_points, check_every_n=5):
        """
        Check if trajectory collides with obstacles using environment's methods
        
        Parameters:
        - trajectory_points: numpy array of shape (N, 3)
        - check_every_n: check every nth point (for efficiency)
        
        Returns: True if collision-free, False if collision detected
        """
        if self.environment is None:
            print("  No environment provided - skipping collision check")
            return True
        
        print("  Checking trajectory for collisions...")
        
        collision_count = 0
        total_checks = 0
        
        # Check points along trajectory
        for i in range(0, len(trajectory_points), check_every_n):
            point = trajectory_points[i]
            total_checks += 1
            
            if not self.environment.is_point_in_free_space(point):
                collision_count += 1
                if collision_count == 1:  # Print first collision
                    print(f"    COLLISION at point {i}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")
            
            # Also check line segments between consecutive checked points
            if i > 0 and i >= check_every_n:
                prev_point = trajectory_points[i - check_every_n]
                if not self.environment.is_line_collision_free(prev_point, point):
                    collision_count += 1
                    if collision_count == 1:
                        print(f"    COLLISION on segment from {prev_point} to {point}")
        
        if collision_count > 0:
            print(f"  Found {collision_count} collision(s) out of {total_checks} checks")
            return False
        
        print(f"  Trajectory is collision-free! ({total_checks} checks passed)")
        return True

    def generate_bspline_trajectory(self, num_points=None):
        """
        Generate minimum snap trajectory (B-spline-like smooth trajectory)
        
        Uses 7th order polynomials and global optimization to minimize snap
        across all segments while ensuring continuity
        
        Parameters:
        - num_points: number of points per segment
        
        Returns:
        - trajectory_points: numpy array of shape (total_points, 3)
        - time_points: numpy array of time stamps
        - velocities: numpy array of velocities (total_points, 3)
        - accelerations: numpy array of accelerations (total_points, 3)
        """
        print("Generating minimum snap trajectory...")
        
        n = len(self.waypoints)
        n_segments = n - 1
        
        if num_points is None:
            num_points = 50  # Default points per segment
        
        # Use 7th order polynomials (8 coefficients per segment)
        poly_order = 7
        n_coeffs = poly_order + 1  # 8 coefficients
        
        # STEP 1: Allocate time for each segment
        segment_times = self._allocate_segment_times()
        
        print(f"Solving minimum snap optimization for {n_segments} segments...")
        
        # STEP 2: Solve for each dimension (x, y, z) separately
        all_trajectory_points = []
        all_velocities = []
        all_accelerations = []
        
        for dim in range(3):
            print(f"  Solving dimension {['X', 'Y', 'Z'][dim]}...")
            
            # Solve the QP problem for this dimension
            coeffs = self._solve_minimum_snap_qp(
                dimension=dim,
                segment_times=segment_times,
                poly_order=poly_order
            )
            
            # Evaluate the trajectory
            traj, vel, acc = self._evaluate_polynomial_trajectory(
                coeffs, segment_times, poly_order, num_points
            )
            
            all_trajectory_points.append(traj)
            all_velocities.append(vel)
            all_accelerations.append(acc)
        
        # STEP 3: Stack dimensions and create time array
        final_trajectory = np.stack(all_trajectory_points, axis=-1)
        final_velocities = np.stack(all_velocities, axis=-1)
        final_accelerations = np.stack(all_accelerations, axis=-1)
        
        # STEP 4: COLLISION CHECK - uses your environment's methods!
        is_safe = self._check_trajectory_collisions(final_trajectory, check_every_n=10)
        
        if not is_safe:
            print("\n" + "="*60)
            print("WARNING: Trajectory collides with obstacles!")
            print("Suggestions:")
            print("  1. Use more waypoints (less simplification in RRT*)")
            print("  2. Increase segment times (slower trajectory)")
            print("  3. Check if waypoints are too close to obstacles")
            print("  4. Increase safety margin in environment")
            print("="*60 + "\n")
        
        # Create cumulative time points
        final_time_points = []
        current_time = 0.0
        for seg_time in segment_times:
            seg_times = np.linspace(0, seg_time, num_points)
            final_time_points.append(seg_times + current_time)
            current_time += seg_time
        final_time_points = np.concatenate(final_time_points)
        
        self.trajectory_duration = final_time_points[-1]
        
        print(f"\nGenerated minimum snap trajectory:")
        print(f"  Total points: {len(final_trajectory)}")
        print(f"  Duration: {self.trajectory_duration:.2f} seconds")
        print(f"  Max velocity: {np.max(np.linalg.norm(final_velocities, axis=1)):.2f} m/s")
        print(f"  Max acceleration: {np.max(np.linalg.norm(final_accelerations, axis=1)):.2f} m/s²")
        print(f"  Collision-free: {'YES ✓' if is_safe else 'NO ✗'}")
        
        # Visualize
        self.visualize_trajectory(final_trajectory, final_velocities, final_accelerations)
        
        return final_trajectory, final_time_points, final_velocities, final_accelerations
    def _allocate_segment_times(self):
        """
        Allocate time for each segment based on distance
        """
        n_segments = len(self.waypoints) - 1
        segment_times = []
        
        for i in range(n_segments):
            distance = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            # Time proportional to distance (average speed ~2 m/s)
            time = max(1.0, distance / 2.0)
            segment_times.append(time)
        
        return np.array(segment_times)

    def _solve_minimum_snap_qp(self, dimension, segment_times, poly_order):
        """
        Solve the minimum snap QP problem for one dimension
        
        Minimizes: ∫(snap²)dt subject to waypoint and continuity constraints
        """
        n_segments = len(segment_times)
        n_coeffs = poly_order + 1
        
        # Build cost matrix Q (minimizes integral of snap squared)
        Q = self._build_snap_cost_matrix(segment_times, poly_order)
        
        # Build constraint matrices
        A_eq, b_eq = self._build_equality_constraints(
            dimension, segment_times, poly_order
        )
        
        # Solve QP using KKT system: minimize (1/2)x^T Q x subject to A_eq x = b_eq
        n_vars = n_segments * n_coeffs
        n_constraints = A_eq.shape[0]
        
        # Build KKT matrix: [Q    A_eq^T] [x]   [0   ]
        #                   [A_eq   0   ] [λ] = [b_eq]
        KKT = np.zeros((n_vars + n_constraints, n_vars + n_constraints))
        KKT[:n_vars, :n_vars] = Q + np.eye(n_vars) * 1e-6  # Add small regularization
        KKT[:n_vars, n_vars:] = A_eq.T
        KKT[n_vars:, :n_vars] = A_eq
        
        rhs = np.zeros(n_vars + n_constraints)
        rhs[n_vars:] = b_eq
        
        # Solve the system
        try:
            solution = np.linalg.solve(KKT, rhs)
            coeffs = solution[:n_vars]
        except np.linalg.LinAlgError:
            print("  Warning: KKT solve failed, using least squares")
            coeffs, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
        
        # Reshape to (n_segments, n_coeffs)
        coeffs = coeffs.reshape(n_segments, n_coeffs)
        
        return coeffs

    def _build_snap_cost_matrix(self, segment_times, poly_order):
        """
        Build the cost matrix Q for minimizing snap
        
        For 7th order: p(t) = c₀ + c₁t + c₂t² + ... + c₇t⁷
        Snap = d⁴p/dt⁴ = 24c₄ + 120c₅t + 360c₆t² + 840c₇t³
        
        Cost = ∫₀ᵀ (snap)² dt
        """
        n_segments = len(segment_times)
        n_coeffs = poly_order + 1
        n_vars = n_segments * n_coeffs
        
        Q = np.zeros((n_vars, n_vars))
        
        for seg in range(n_segments):
            T = segment_times[seg]
            Q_seg = np.zeros((n_coeffs, n_coeffs))
            
            # Only coefficients c₄ to c₇ contribute to snap (4th derivative)
            for i in range(4, n_coeffs):
                for j in range(4, n_coeffs):
                    # Derivative coefficients: d⁴/dt⁴(tⁱ) = i!/(i-4)! * t^(i-4)
                    if i >= 4 and j >= 4:
                        # Factorial terms
                        deriv_i = math.factorial(i) / math.factorial(i - 4)
                        deriv_j = math.factorial(j) / math.factorial(j - 4)
                        
                        # Power after multiplication
                        power = (i - 4) + (j - 4)
                        
                        # Integral: ∫₀ᵀ t^power dt = T^(power+1) / (power+1)
                        Q_seg[i, j] = deriv_i * deriv_j * (T ** (power + 1)) / (power + 1)
            
            # Place in global Q matrix
            idx_start = seg * n_coeffs
            idx_end = (seg + 1) * n_coeffs
            Q[idx_start:idx_end, idx_start:idx_end] = Q_seg
        
        return Q

    def _build_equality_constraints(self, dimension, segment_times, poly_order):
        """
        Build equality constraint matrices A_eq x = b_eq
        
        Constraints:
        1. Position at each waypoint (start and end of each segment)
        2. Velocity continuity at intermediate waypoints
        3. Acceleration continuity at intermediate waypoints
        4. Jerk continuity at intermediate waypoints
        5. Zero velocity at start and end
        6. Zero acceleration at start and end
        """
        n_segments = len(segment_times)
        n_coeffs = poly_order + 1
        n_vars = n_segments * n_coeffs
        
        constraints = []
        values = []
        
        def get_poly_coeffs(t, derivative=0):
            """Get coefficient multipliers for evaluating polynomial at time t"""
            coeffs = np.zeros(n_coeffs)
            for i in range(n_coeffs):
                if i >= derivative:
                    # Calculate derivative coefficient
                    deriv_factor = 1
                    for k in range(derivative):
                        deriv_factor *= (i - k)
                    coeffs[i] = deriv_factor * (t ** (i - derivative))
            return coeffs
        
        # CONSTRAINT 1: Position at waypoints
        for seg in range(n_segments):
            T = segment_times[seg]
            
            # Start of segment (t=0)
            row = np.zeros(n_vars)
            row[seg * n_coeffs:(seg + 1) * n_coeffs] = get_poly_coeffs(0, derivative=0)
            constraints.append(row)
            values.append(self.waypoints[seg][dimension])
            
            # End of segment (t=T)
            row = np.zeros(n_vars)
            row[seg * n_coeffs:(seg + 1) * n_coeffs] = get_poly_coeffs(T, derivative=0)
            constraints.append(row)
            values.append(self.waypoints[seg + 1][dimension])
        
        # CONSTRAINT 2-4: Continuity at intermediate waypoints
        for seg in range(n_segments - 1):
            T = segment_times[seg]
            
            # Continuity of velocity, acceleration, and jerk
            for deriv in [1, 2, 3]:
                row = np.zeros(n_vars)
                # End of current segment
                row[seg * n_coeffs:(seg + 1) * n_coeffs] = get_poly_coeffs(T, derivative=deriv)
                # Start of next segment
                row[(seg + 1) * n_coeffs:(seg + 2) * n_coeffs] = -get_poly_coeffs(0, derivative=deriv)
                constraints.append(row)
                values.append(0.0)
        
        # CONSTRAINT 5: Zero velocity at start and end
        # Start velocity = 0
        row = np.zeros(n_vars)
        row[:n_coeffs] = get_poly_coeffs(0, derivative=1)
        constraints.append(row)
        values.append(0.0)
        
        # End velocity = 0
        row = np.zeros(n_vars)
        T_last = segment_times[-1]
        row[-n_coeffs:] = get_poly_coeffs(T_last, derivative=1)
        constraints.append(row)
        values.append(0.0)
        
        # CONSTRAINT 6: Zero acceleration at start and end
        # Start acceleration = 0
        row = np.zeros(n_vars)
        row[:n_coeffs] = get_poly_coeffs(0, derivative=2)
        constraints.append(row)
        values.append(0.0)
        
        # End acceleration = 0
        row = np.zeros(n_vars)
        row[-n_coeffs:] = get_poly_coeffs(T_last, derivative=2)
        constraints.append(row)
        values.append(0.0)
        
        A_eq = np.array(constraints)
        b_eq = np.array(values)
        
        return A_eq, b_eq

    def _evaluate_polynomial_trajectory(self, coeffs, segment_times, poly_order, num_points):
        """
        Evaluate the polynomial trajectory at discrete time points
        """
        all_positions = []
        all_velocities = []
        all_accelerations = []
        
        for seg in range(len(segment_times)):
            T = segment_times[seg]
            t_seg = np.linspace(0, T, num_points)
            
            # Get coefficients for this segment
            c = coeffs[seg]
            
            positions = np.zeros(num_points)
            velocities = np.zeros(num_points)
            accelerations = np.zeros(num_points)
            
            # Evaluate polynomial at each time point
            for idx, t in enumerate(t_seg):
                # Position: p(t) = c₀ + c₁t + c₂t² + ... + c₇t⁷
                for k in range(poly_order + 1):
                    positions[idx] += c[k] * (t ** k)
                
                # Velocity: dp/dt
                for k in range(1, poly_order + 1):
                    velocities[idx] += k * c[k] * (t ** (k - 1))
                
                # Acceleration: d²p/dt²
                for k in range(2, poly_order + 1):
                    accelerations[idx] += k * (k - 1) * c[k] * (t ** (k - 2))
            
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_accelerations.append(accelerations)
        
        return (np.concatenate(all_positions),
                np.concatenate(all_velocities),
                np.concatenate(all_accelerations))

    def visualize_trajectory(self, trajectory_points=None, velocities=None, 
                           accelerations=None, ax=None):
        """Visualize the trajectory with velocity and acceleration vectors"""
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            standalone = True
        else:
            ax1 = ax
            standalone = False
        
        if trajectory_points is not None:
            # Plot 3D trajectory
            ax1.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    trajectory_points[:, 2], 'b-', linewidth=2, label='Spline Trajectory')
            
            # Plot waypoints
            ax1.plot(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], 
                    'ro-', markersize=8, linewidth=2, label='Waypoints')
            
            # Plot velocity vectors (sampled)
            if velocities is not None:
                step = max(1, len(trajectory_points) // 20)  # Show ~20 vectors
                for i in range(0, len(trajectory_points), step):
                    pos = trajectory_points[i]
                    vel = velocities[i] * 0.5  # Scale for visualization
                    ax1.quiver(pos[0], pos[1], pos[2], 
                             vel[0], vel[1], vel[2], 
                             color='green', alpha=0.7, arrow_length_ratio=0.1)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()
        
        if standalone and velocities is not None and accelerations is not None:
            # Plot velocity magnitude over time
            time_points = np.linspace(0, self.trajectory_duration, len(velocities))
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            ax2.plot(time_points, vel_magnitudes, 'g-', linewidth=2)
            if self.max_velocity is not None:
                ax2.axhline(y=self.max_velocity, color='r', linestyle='--', 
                           label=f'Max Vel: {self.max_velocity} m/s')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Profile')
            ax2.grid(True)
            ax2.legend()
            
            # Plot acceleration magnitude over time
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            ax3.plot(time_points, acc_magnitudes, 'm-', linewidth=2)
            if self.max_acceleration is not None:
                ax3.axhline(y=self.max_acceleration, color='r', linestyle='--', 
                           label=f'Max Acc: {self.max_acceleration} m/s²')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Acceleration (m/s²)')
            ax3.set_title('Acceleration Profile')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
        
        return ax1 if not standalone else None