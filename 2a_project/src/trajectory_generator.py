import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using  
splines
    Complete implementation with velocity and acceleration profiles
    """
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.trajectory_duration = None  # seconds
        self.max_velocity = None  # m/s
        self.max_acceleration = None  # m/s^2

    ##############################################################
    #### TODO - Implement spline trajectory generation ###########
    
    #### TODO - Ensure velocity and acceleration constraints #####
    #### TODO - Add member functions as needed ###################
    ##############################################################
    def generate_spline_trajectory(self, num_points=None):
        """
        Generate spline trajectory with complete velocity/acceleration profiles
        
        Parameters:
        - num_points: number of points in the smooth trajectory
        
        Returns:
        - trajectory_points: numpy array of shape (num_points, 3)
        - time_points: numpy array of time stamps
        - velocities: numpy array of velocities (num_points, 3)
        - accelerations: numpy array of accelerations (num_points, 3)
        """
        print("Generating spline trajectory...")
        all_trajectory_points = []
        all_velocities = []
        all_accelerations = []
        
        for i in range(len(self.waypoints) - 1):
            po = self.waypoints[i]
            pf = self.waypoints[i + 1]
            
            to = 0
            tf = 5
            qo = po
            qf = pf
            vo = np.ones_like(qo) * 2.0     
            vf = np.ones_like(qf) * 1.0      
            ao = np.ones_like(qo) * 0.05      
            af = np.ones_like(qf) * 0.00     

            time_points = np.linspace(to, tf, num_points)

            A = np.array([[1, to, to**2, to**3, to**4, to**5],
                          [0, 1, 2*to, 3*to**2, 4*to**3, 5*to**4],
                          [0, 0, 2, 6*to, 12*to**2, 20*to**3],  
                            [1, tf, tf**2, tf**3, tf**4, tf**5],
                            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])    
            
            b = np.array([qo, vo, ao, qf, vf, af])
            
            x = np.linalg.solve(A, b)

            trajectory_points = []
            velocities = []
            accelerations = []

            for dim in range(len(qo)):
                b = np.array([qo[dim], vo[dim], ao[dim], qf[dim], vf[dim], af[dim]])
                x = np.linalg.solve(A, b)
                
                traj = x[0] + x[1]*time_points + x[2]*time_points**2 + \
                       x[3]*time_points**3 + x[4]*time_points**4 + x[5]*time_points**5
                vel = x[1] + 2*x[2]*time_points + 3*x[3]*time_points**2 + \
                      4*x[4]*time_points**3 + 5*x[5]*time_points**4
                acc = 2*x[2] + 6*x[3]*time_points + 12*x[4]*time_points**2 + \
                      20*x[5]*time_points**3


                trajectory_points.append(traj)
                velocities.append(vel)
                accelerations.append(acc)

            trajectory_points = np.stack(trajectory_points, axis=-1)  # shape (num_points, 3)
            velocities = np.stack(velocities, axis=-1)
            accelerations = np.stack(accelerations, axis=-1)

            all_trajectory_points.append(trajectory_points)
            all_velocities.append(velocities)
            all_accelerations.append(accelerations)


        ############## IMPLEMENTATION STARTS HERE ##############
        traj_points = np.concatenate(all_trajectory_points, axis=0)
        velocities = np.concatenate(all_velocities, axis=0)
        accelerations = np.concatenate(all_accelerations, axis=0)
            
        self.visualize_trajectory(traj_points, velocities, accelerations)

        return traj_points, time_points, velocities, accelerations
            

 
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