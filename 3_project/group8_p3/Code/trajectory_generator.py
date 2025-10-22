import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using B-splines
    Complete implementation with velocity and acceleration profiles
    """
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.trajectory_func = None
        self.trajectory_duration = 15.0  # seconds
        self.max_velocity = 2.0  # m/s
        self.max_acceleration = 1.0  # m/s^2
    
    def calculate_segment_times(self):
        """Calculate time allocation for each segment based on distance"""
        if len(self.waypoints) < 2:
            return np.array([0.0])
        
        # Calculate distances between consecutive waypoints
        distances = []
        for i in range(len(self.waypoints) - 1):
            dist = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Allocate time based on distance (with velocity constraint)
        segment_times = distances / self.max_velocity
        
        # Ensure minimum time for each segment
        min_segment_time = 1.0
        segment_times = np.maximum(segment_times, min_segment_time)
        
        # Create cumulative time array
        cumulative_times = np.zeros(len(self.waypoints))
        cumulative_times[1:] = np.cumsum(segment_times)
        
        # Scale to desired total duration
        if cumulative_times[-1] > 0:
            cumulative_times = cumulative_times * self.trajectory_duration / cumulative_times[-1]
        
        return cumulative_times
    
    def generate_bspline_trajectory(self, num_points=None):
        """
        Generate B-spline trajectory with complete velocity/acceleration profiles
        
        Parameters:
        - num_points: number of points in the smooth trajectory
        
        Returns:
        - trajectory_points: numpy array of shape (num_points, 3)
        - time_points: numpy array of time stamps
        - velocities: numpy array of velocities (num_points, 3)
        - accelerations: numpy array of accelerations (num_points, 3)
        """
        print("Generating B-spline trajectory...")
        
        if len(self.waypoints) < 2:
            print("Error: Need at least 2 waypoints")
            return None, None, None, None
        
        if num_points is None:
            num_points = max(100, len(self.waypoints) * 20)
        
        # Calculate time allocation for waypoints
        waypoint_times = self.calculate_segment_times()
        
        try:
            # Handle special case of only 2 waypoints
            if len(self.waypoints) == 2:
                # Simple linear interpolation for 2 points
                time_points = np.linspace(0, self.trajectory_duration, num_points)
                trajectory_points = []
                velocities = []
                accelerations = []
                
                direction = self.waypoints[1] - self.waypoints[0]
                total_distance = np.linalg.norm(direction)
                unit_direction = direction / total_distance if total_distance > 0 else np.zeros(3)
                
                # Simple trapezoidal velocity profile
                accel_time = self.trajectory_duration * 0.2  # 20% acceleration phase
                decel_time = self.trajectory_duration * 0.2  # 20% deceleration phase
                cruise_time = self.trajectory_duration - accel_time - decel_time
                
                max_vel = min(self.max_velocity, total_distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time))
                
                for t in time_points:
                    if t <= accel_time:
                        # Acceleration phase
                        vel_mag = (max_vel / accel_time) * t
                        acc_mag = max_vel / accel_time
                        progress = 0.5 * (max_vel / accel_time) * t * t / total_distance
                    elif t <= accel_time + cruise_time:
                        # Cruise phase
                        vel_mag = max_vel
                        acc_mag = 0.0
                        progress = (0.5 * max_vel * accel_time + max_vel * (t - accel_time)) / total_distance
                    else:
                        # Deceleration phase
                        t_decel = t - accel_time - cruise_time
                        vel_mag = max_vel - (max_vel / decel_time) * t_decel
                        acc_mag = -max_vel / decel_time
                        progress = (0.5 * max_vel * accel_time + max_vel * cruise_time + 
                                  max_vel * t_decel - 0.5 * (max_vel / decel_time) * t_decel * t_decel) / total_distance
                    
                    progress = np.clip(progress, 0.0, 1.0)
                    position = self.waypoints[0] + progress * direction
                    velocity = vel_mag * unit_direction
                    acceleration = acc_mag * unit_direction
                    
                    trajectory_points.append(position)
                    velocities.append(velocity)
                    accelerations.append(acceleration)
                
                return (np.array(trajectory_points), time_points, 
                       np.array(velocities), np.array(accelerations))
            
            # For 3+ waypoints, use B-spline interpolation
            # Prepare waypoints for spline fitting
            waypoints_T = self.waypoints.T  # Transpose for splprep
            
            # Fit B-spline with automatic parameter estimation
            tck, u = splprep(waypoints_T, s=0.1, k=min(3, len(self.waypoints)-1))
            
            # Generate time points
            time_points = np.linspace(0, self.trajectory_duration, num_points)
            u_points = np.linspace(0, 1, num_points)
            
            # Evaluate B-spline for positions
            spline_points = splev(u_points, tck, der=0)
            trajectory_points = np.array(spline_points).T
            
            # Calculate velocities (first derivative)
            spline_velocities = splev(u_points, tck, der=1)
            velocities_raw = np.array(spline_velocities).T
            
            # Scale velocities by time parameterization
            dt = self.trajectory_duration / (num_points - 1)
            du_dt = 1.0 / self.trajectory_duration  # du/dt
            velocities = velocities_raw * du_dt
            
            # Calculate accelerations (second derivative)
            spline_accelerations = splev(u_points, tck, der=2)
            accelerations_raw = np.array(spline_accelerations).T
            accelerations = accelerations_raw * (du_dt ** 2)
            
            # Apply velocity and acceleration constraints
            velocities, accelerations = self.apply_kinematic_constraints(
                velocities, accelerations, dt)
            
            print(f"Generated trajectory with {len(trajectory_points)} points")
            print(f"Duration: {self.trajectory_duration:.1f}s")
            print(f"Max velocity: {np.max(np.linalg.norm(velocities, axis=1)):.2f} m/s")
            print(f"Max acceleration: {np.max(np.linalg.norm(accelerations, axis=1)):.2f} m/s^2")
            
            return trajectory_points, time_points, velocities, accelerations
            
        except Exception as e:
            print(f"B-spline generation failed: {e}")
            print("Falling back to linear interpolation...")
            return self.generate_linear_trajectory(num_points)
    
    def apply_kinematic_constraints(self, velocities, accelerations, dt):
        """Apply velocity and acceleration constraints"""
        # Velocity constraint
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        vel_scale = np.ones_like(vel_magnitudes)
        
        over_limit = vel_magnitudes > self.max_velocity
        vel_scale[over_limit] = self.max_velocity / vel_magnitudes[over_limit]
        
        velocities_constrained = velocities * vel_scale[:, np.newaxis]
        
        # Acceleration constraint
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        acc_scale = np.ones_like(acc_magnitudes)
        
        over_limit = acc_magnitudes > self.max_acceleration
        acc_scale[over_limit] = self.max_acceleration / acc_magnitudes[over_limit]
        
        accelerations_constrained = accelerations * acc_scale[:, np.newaxis]
        
        return velocities_constrained, accelerations_constrained
    
    def generate_linear_trajectory(self, num_points):
        """Fallback linear interpolation trajectory"""
        print("Generating linear trajectory...")
        
        time_points = np.linspace(0, self.trajectory_duration, num_points)
        trajectory_points = []
        velocities = []
        accelerations = []
        
        for t in time_points:
            progress = t / self.trajectory_duration
            
            if progress >= 1.0:
                trajectory_points.append(self.waypoints[-1])
                velocities.append(np.zeros(3))
                accelerations.append(np.zeros(3))
            else:
                # Find current segment
                segment_length = 1.0 / (len(self.waypoints) - 1)
                segment_idx = int(progress / segment_length)
                segment_idx = min(segment_idx, len(self.waypoints) - 2)
                
                local_progress = (progress - segment_idx * segment_length) / segment_length
                
                # Linear interpolation
                point = ((1 - local_progress) * self.waypoints[segment_idx] + 
                        local_progress * self.waypoints[segment_idx + 1])
                
                # Simple velocity calculation
                if segment_idx < len(self.waypoints) - 1:
                    segment_vector = self.waypoints[segment_idx + 1] - self.waypoints[segment_idx]
                    segment_time = self.trajectory_duration / (len(self.waypoints) - 1)
                    velocity = segment_vector / segment_time
                else:
                    velocity = np.zeros(3)
                
                trajectory_points.append(point)
                velocities.append(velocity)
                accelerations.append(np.zeros(3))
        
        return (np.array(trajectory_points), time_points, 
               np.array(velocities), np.array(accelerations))
    
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
                    trajectory_points[:, 2], 'b-', linewidth=2, label='B-spline Trajectory')
            
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