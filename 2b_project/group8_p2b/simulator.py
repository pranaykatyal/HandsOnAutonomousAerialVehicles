import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp
import time
import os
from matplotlib.animation import FuncAnimation

# Local imports
from environment import Environment3D
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator
from control import QuadrotorController
from video_gen import  ffmpeging_video, create_combined_video
# Dynamics and parameters
from quad_dynamics import model_derivative
import tello as drone_params
import cv2
from pathlib import Path
# splat rendering
from splat_render import SplatRenderer

class LiveQuadrotorSimulator:
    """
    Real-time live visualization quadrotor simulator
    Shows step-by-step: RRT* planning -> B-spline -> Execution
    """
    
    def __init__(self, map_file=None):
        # Initialize components
        self.env = Environment3D()
        if map_file:
            success = self.env.parse_map_file(map_file)
            if not success:
                print(f"Failed to load map file: {map_file}")
        
        self.planner = PathPlanner(self.env)
        self.traj_gen = None
        self.controller = QuadrotorController(drone_params)

        # renderer
        self.splatConfig    = './p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml'
        self.renderSettings = './vizflyt_viewer/render_settings/render_config.json'
        try:
            self.renderer = SplatRenderer(self.splatConfig, self.renderSettings)
            self.rendering_enabled = True
            print(" Gaussian Splat Renderer initialized")
        except Exception as e:
            print(f" Renderer failed: {e}")
            self.rendering_enabled = False
        
        
        self.render_every_n_steps = 2     
        self.frame_count = 0              
        self.saved_frames = []            
        self.render_rgb_dir = None        
        self.render_depth_dir = None      
        self.render_plot_dir = None       
        self.video_dir = None             
        self.setup_output_directories()
        # self.renderer = SplatRenderer(self.splatConfig, self.renderSettings)
        
        # Create log directory
        if not os.path.exists('./log'):
            os.makedirs('./log')
        
        # Simulation parameters
        self.dt = 0.02  # 50 Hz for smoother animation
        self.sim_time = 0.0
        self.max_sim_time = 30.0
        
        # Quadrotor state: [x,y,z,vx,vy,vz,qx,qy,qz,qw,p,q,r]
        self.state = np.zeros(13)
        self.state[9] = 1.0  # Initialize quaternion w = 1
        
        # Logging
        self.state_history = []
        self.time_history = []
        self.control_history = []

        # splat parameters
        self.robotHeading = [] # roll, pitch, yaw
        self.render_dir = './renders'
        os.makedirs(self.render_dir, exist_ok=True)
        
        # Simulation status
        self.goal_reached = False
        self.goal_tolerance = 0.001  # meters
        self.simulation_active = False
        
        # Visualization elements
        self.fig = None
        self.ax = None
        self.drone_point = None
        self.drone_trail = None
        self.trail_positions = []
        self.max_trail_length = 200
        
        # RRT* visualization elements
        self.rrt_tree_lines = []
        self.rrt_nodes_scatter = None
        self.rrt_path_line = None
        self.bspline_line = None
        
        # Planning phase tracking
        self.planning_complete = False
        self.trajectory_complete = False
        self.execution_started = False
     
     
    def _quaternion_to_rpy(self, quat):
        """
        Convert quaternion [qx, qy, qz, qw] to roll, pitch, yaw
        
        Args:
            quat: [qx, qy, qz, qw] from state[6:10]
        
        Returns:
            [roll, pitch, yaw] in radians
        """
        from pyquaternion import Quaternion
        q = Quaternion(quat[3], quat[0], quat[1], quat[2])  # w, x, y, z
        yaw, pitch, roll = q.yaw_pitch_roll
        return np.array([roll, pitch, yaw])  
     
    def create_side_by_side_video(self, left_video, right_video, output_name="combined"):
        """Create side-by-side video combining FPV and plot view"""
        import subprocess
        
        output_path = self.video_dir / f"{output_name}.mp4"
        
        print(f"🎬 Creating side-by-side video...")
        
        # Scale both to same height (1080p), then stack horizontally
        cmd = [
            'ffmpeg', '-y',
            '-i', str(left_video),
            '-i', str(right_video),
            '-filter_complex', 
            '[0:v]scale=-1:1080[left];[1:v]scale=-1:1080[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'slow',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f" Combined video created: {output_path}")
                return output_path
            else:
                print(f" ffmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f" Error creating combined video: {e}")
            return None
     
    def _render_and_save(self):
        """Render current view and save RGB and depth images"""
        try:
            # Get current position and orientation
            current_pos = self.state[0:3]
            current_quat = self.state[6:10]
            
            # Convert quaternion to roll, pitch, yaw
            rpy = self._quaternion_to_rpy(current_quat)
            
            # Render
            rgb_img, depth_img = self.renderer.render(current_pos, rpy)
            
            # Save images with timestamp
            frame_id = len(self.state_history)
            timestamp = f"{self.sim_time:.3f}".replace('.', '_')
            print("timestep", timestamp)
            rgb_filename = os.path.join(self.render_dir, f'rgb_{frame_id:05d}.png')#f'rgb_{frame_id:05d}_t{timestamp}.png
            depth_filename = os.path.join(self.render_dir, f'depth_{frame_id:05d}.png')

            plot_filename = os.path.join(self.render_dir, f'plot_{frame_id:05d}.png')

            import cv2
            cv2.imwrite(rgb_filename, rgb_img)
            cv2.imwrite(depth_filename, depth_img)

            self.fig.savefig(plot_filename,bbox_inches='tight', pad_inches=0)


            # # Optional: print progress
            # if frame_id % 50 == 0:
            #     print(f"   Rendered frame {frame_id} at t={self.sim_time:.2f}s")
                
        except Exception as e:
            print(f"   Rendering error at t={self.sim_time:.2f}s: {e}")
    
    
    def setup_visualization(self):
        """Setup the 3D visualization for step-by-step planning"""
        # plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the environment
        self._draw_environment()
        
        # Set equal aspect ratio and limits
        if self.env.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Quadrotor Path Planning and Execution')
        
        plt.show(block=True)


    def preview_environment(self, wait_time=3.0):
            """
            Show the environment and wait before starting planning
            
            Args:
                wait_time: Time to display environment in seconds (default: 3.0)
                        Set to None for manual continuation (press Enter)
            """
            print("\n" + "="*60)
            print("ðŸ‘€ ENVIRONMENT PREVIEW")
            print("="*60)
            
            # Print environment information
            print(self.env.get_environment_info())
            
            # Show the visualization
            self.ax.set_title('ðŸŒ Environment Preview - Inspecting workspace...')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.savefig('./report_outputs/Environment_Preview.png')
            
            if wait_time is None:
                # Manual continuation
                print("\nâ¸ï¸  Environment displayed!")
                print("ðŸ“ Start (green square) and Goal (gold star) marked")
                print("ðŸ§± Obstacles shown in the workspace")
                input("\nâ–¶ï¸  Press ENTER to start planning...")
            else:
                # Automatic continuation after wait_time
                print(f"\nâ¸ï¸  Displaying environment for {wait_time:.1f} seconds...")
                print("ðŸ“ Start (green square) and Goal (gold star) marked")
                print("ðŸ§± Obstacles shown in the workspace")
                
                # Countdown
                for i in range(int(wait_time), 0, -1):
                    print(f"   Starting planning in {i}...", end='\r')
                    time.sleep(1)
                print("\nâ–¶ï¸  Starting planning now!                    ")
            
            print("="*60 + "\n")
    
    def animated_rrt_planning(self, start=None, goal=None):
        """Show animated RRT* planning process"""
        print("ðŸŽ¬ Starting animated RRT* planning...")
        
        # Print environment information
        print(self.env.get_environment_info())
        
        # Set start and goal points with smart defaults
        # if not self.env.set_start_goal_points(start, goal):
        #     print("âŒ Failed to set start/goal points")
        #     return False
        
        # Verify the points are well-separated
        start_point = self.env.start_point
        goal_point = self.env.goal_point #self.renderer.getObstaclePoses(np.array(self.env.goal_point))
        print(f"Start point: {self.env.start_point}")
        print(f"Goal point: {self.env.goal_point}")
        print(f"Start valid: {self.env.is_point_in_free_space(self.env.start_point)}")
        print(f"Goal valid: {self.env.is_point_in_free_space(self.env.goal_point)}")
        
        distance = np.linalg.norm(np.array(goal_point) - np.array(start_point))
        
        print(f"ðŸ“ Start-to-goal distance: {distance:.2f} meters")
        if distance < 0.10:
            print("âš ï¸ Warning: Start and goal are quite close")
        elif distance > 15.0:
            print("âš ï¸ Warning: Start and goal are quite far - this may take longer")
        else:
            print("âœ… Good start-goal separation for planning")
        
        # Update title
        self.ax.set_title('Phase 1: RRT* Path Planning (Building Tree...)')

        plt.show()

        # time.sleep(1000)
        
        # Custom RRT* with visualization
        from path_planner import RRTNode
        start_node = RRTNode(start_point)
        tree = [start_node]
        
        # RRT* parameters - adjust based on environment size
        max_iterations = 10000#min(3000, max(1000, int(distance * 150)))  # Scale with distance
        step_size = min(1.5, distance / 10)  # Adaptive step size
        step_size = 0.01
        goal_radius = max(0.8, min(1.5, distance / 15))  # Adaptive goal radius
        goal_radius = 0.01
        search_radius = step_size * 2.5
        goal_bias = 0.15
        
        print(f"ðŸ”§ RRT* Parameters:")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Step size: {step_size:.3f}m")
        print(f"   Goal radius: {goal_radius:.3f}m")
        print(f"   Search radius: {search_radius:.3f}m")
        
        goal_node = None
        update_interval = max(25, max_iterations // 80)  # Adaptive update rate
        
        print(f"ðŸš€ Starting RRT* planning from {start_point} to {goal_point}")
        
        for iteration in range(max_iterations):
            # Sample point
            if np.random.random() < goal_bias:
                sample_point = np.array(goal_point)
            else:
                sample = self.env.generate_random_free_point()
                if sample is None:
                    continue
                sample_point = np.array(sample)
            
            # Find nearest node
            nearest_node = self.planner.find_nearest_node(tree, sample_point)
            if nearest_node is None:
                continue
            
            # Steer towards sample
            new_position = self.planner.steer(nearest_node.position, sample_point, step_size)
            
            # Check validity
            if not self.env.is_point_in_free_space(new_position):
                continue
            if not self.env.is_line_collision_free(nearest_node.position, new_position):
                continue
            
            # Find near nodes and choose best parent
            near_nodes = self.planner.find_near_nodes(tree, new_position, search_radius)
            new_node = self.planner.choose_parent(near_nodes, nearest_node, new_position)
            tree.append(new_node)

            # Rewire tree
            self.planner.rewire_tree(new_node, near_nodes)

            # Check if goal reached
            if self.planner.reached_goal(new_node):
                    goal_node = new_node
                    print(f"Goal reached at iteration {iteration}! Final cost: {new_node.cost:.2f}")
                    break
            # Update visualization periodically
            if iteration % update_interval == 0:
                self._update_rrt_visualization(tree, iteration, max_iterations)
                time.sleep(0.03)  # Small delay for animation effect
        print("all done")
        # Final visualization update
        self._update_rrt_visualization(tree, iteration, max_iterations, final=True)
        
        # Store results
        self.planner.tree_nodes = tree
        
        if goal_node is not None:
            print("it foind a goal")
            self.planner.waypoints = self.planner.extract_path(goal_node)
            original_waypoints = len(self.planner.waypoints)
            self.planner.waypoints = self.planner.simplify_path(self.planner.waypoints)
            simplified_waypoints = len(self.planner.waypoints)
            
            print(f"â RRT* planning successful!")
            print(f"   Original path: {original_waypoints} waypoints")
            print(f"   Simplified path: {simplified_waypoints} waypoints")
            print(f"   Path cost: {goal_node.cost:.2f} meters")
            print(f"   Tree size: {len(tree)} nodes")
            
            # Show final path
            self._show_final_rrt_path()
            return True
        else:
            print("deepak special charicters")
            print(f"â RRT* planning failed after {max_iterations} iterations")
            print(f"   Tree size: {len(tree)} nodes")
            print("   Try increasing max_iterations or adjusting parameters")
            return False
    
    def _update_rrt_visualization(self, tree, iteration, max_iterations, final=False):
        """Update RRT* tree visualization"""
        # Clear previous tree visualization
        for line in self.rrt_tree_lines:
            line.remove()
        self.rrt_tree_lines = []
        
        if self.rrt_nodes_scatter is not None:
            self.rrt_nodes_scatter.remove()
            self.rrt_nodes_scatter = None
        
        # Draw tree edges (sample only for performance)
        sample_rate = max(1, len(tree) // 500)  # Limit to ~500 edges
        for i, node in enumerate(tree[::sample_rate]):
            if node.parent is not None:
                line, = self.ax.plot([node.parent.position[0], node.position[0]],
                                   [node.parent.position[1], node.position[1]],
                                   [node.parent.position[2], node.position[2]],
                                   'b-', alpha=0.3, linewidth=0.5)
                self.rrt_tree_lines.append(line)
        
        # Draw nodes (sample for performance)
        if len(tree) > 1:
            sampled_nodes = tree[::max(1, len(tree) // 200)]  # Limit to ~200 nodes
            positions = np.array([node.position for node in sampled_nodes])
            self.rrt_nodes_scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                                   c='blue', s=8, alpha=0.6)
        
        # Update title
        progress = (iteration / max_iterations) * 100
        status = "COMPLETE" if final else f"{progress:.1f}%"
        self.ax.set_title(f'Phase 1: RRT* Planning - {status} (Nodes: {len(tree)}, Iter: {iteration})')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _show_final_rrt_path(self):
        """Show the final RRT* path"""
        time.sleep(1)  # Pause to show final tree
        
        # Draw final path
        if len(self.planner.waypoints) > 0:
            waypoints = np.array(self.planner.waypoints)
            self.rrt_path_line, = self.ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                                             'ro-', markersize=8, linewidth=4, 
                                             label='RRT* Path', alpha=0.9)
        
        self.ax.set_title('Phase 1: RRT* Planning - COMPLETE! Final path shown.')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig('./report_outputs/final_RRT_path.png')
        
        time.sleep(2)  # Show final path
        self.planning_complete = True
    
    def show_bspline_trajectory(self):
        """Show B-spline trajectory generation"""
        print("ðŸ“ˆ Generating B-spline trajectory...")
        
        self.ax.set_title('Phase 2: B-spline Trajectory Generation...')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Generate trajectory
        self.traj_gen = TrajectoryGenerator(self.planner.waypoints,environment=self.env)
        self.traj_gen.trajectory_duration = min(20.0, len(self.planner.waypoints) * 2.0)
        
        num_points = 50
        result = self.traj_gen.generate_bspline_trajectory(num_points=num_points)
        
        if result[0] is not None:
            trajectory_points, time_points, velocities, accelerations = result
            self.controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
            self.max_sim_time = self.traj_gen.trajectory_duration + 5.0
            
            # Draw B-spline trajectory
            # Sample trajectory for visualization
            traj_sample = trajectory_points[::5]  # Every 5th point
            self.bspline_line, = self.ax.plot(traj_sample[:, 0], traj_sample[:, 1], traj_sample[:, 2], 
                                            'g-', linewidth=3, alpha=0.8, label='B-spline Trajectory')
            
            # Add velocity vectors at key points
            vector_sample = max(1, len(trajectory_points) // 20)
            for i in range(0, min(len(trajectory_points), len(velocities)), vector_sample):
                pos = trajectory_points[i]
                vel = velocities[i] * 0.4  # Scale for visualization
                self.ax.quiver(pos[0], pos[1], pos[2], 
                             vel[0], vel[1], vel[2], 
                             color='orange', alpha=0.7, arrow_length_ratio=0.1)
            
            self.ax.set_title('Phase 2: B-spline Trajectory - COMPLETE!')
            self.ax.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.savefig('/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs/final_bspline_trajectory.png')

            time.sleep(2)  # Show trajectory
            self.trajectory_complete = True
            return True
        else:
            print("âŒ Trajectory generation failed")
            return False
    
    def initialize_execution_phase(self):
        """Initialize the execution phase"""
        print("ðŸš Starting execution phase...")
        
        # Set initial state
        self.state[0:3] = self.env.start_point
        self.state[3:6] = 0  # Zero initial velocity
        self.state[6:10] = [0, 0, 0, 1]  # Identity quaternion
        self.state[10:13] = 0  # Zero angular rates
        
        # Initialize drone visualization
        current_pos = self.state[0:3]
        self.drone_point = self.ax.scatter(*current_pos, c='red', s=200, marker='o', 
                                         label='Quadrotor', edgecolors='black', linewidths=2)
        
        # Initialize trail
        self.trail_positions = [current_pos.copy()]
        self.drone_trail, = self.ax.plot([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                                       'purple', linewidth=4, alpha=0.9, label='Executed Path')
        
        # Reset metrics
        self.controller.reset_metrics()
        self.sim_time = 0.0
        self.goal_reached = False
        
        self.ax.set_title('Phase 3: Executing Trajectory...')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self.execution_started = True
    
    def simulation_step(self):
        """Single simulation time step"""
        if not self.simulation_active:
            return False
        
        # Get control input
        control_input = self.controller.compute_control(self.state, self.sim_time)
        
        # Dynamics integration
        def dynamics(t, x):
            return model_derivative(t, x.reshape(-1, 1), 
                                  control_input.reshape(-1, 1), 
                                  drone_params).flatten()
        
        # Integrate one step
        sol = solve_ivp(dynamics, [self.sim_time, self.sim_time + self.dt], 
                       self.state, method='RK45')
        
        self.state = sol.y[:, -1]
        self.sim_time += self.dt
        
        # Log data
        self.state_history.append(self.state.copy())
        self.time_history.append(self.sim_time)
        self.control_history.append(control_input.copy())
        
        # Update trail
        current_pos = self.state[0:3]
        self.trail_positions.append(current_pos.copy())
        
        if len(self.state_history) % self.render_interval == 0:
            print("saveing FPV")
            self._render_and_save()



        # Limit trail length
        if len(self.trail_positions) > self.max_trail_length:
            self.trail_positions.pop(0)
        
        # P2B: Render frames periodically
        step_number = len(self.state_history)
        if self.rendering_enabled and (step_number % self.render_every_n_steps == 0):
            rgb_frame, depth_frame = self.render_current_view()
            if rgb_frame is not None:
                self.save_rendered_frame(rgb_frame, depth_frame)
                self.save_plot_frame()
                
                # Progress updates
                if self.frame_count % 25 == 0:
                    print(f"  Rendered {self.frame_count} frames at t={self.sim_time:.2f}s")
        
        # Check goal reached
        if self.env.goal_point is not None:
            goal_distance = np.linalg.norm(current_pos - np.array(self.env.goal_point))
            if goal_distance < self.goal_tolerance and not self.goal_reached:
                self.goal_reached = True
                print(f"\nðŸŽ‰ GOAL REACHED at time {self.sim_time:.2f}s!")
                print(f"ðŸ“ Final distance to goal: {goal_distance:.3f}m")
                return False
        
        # Check time limit
        if self.sim_time >= self.max_sim_time:
            print(f"\nâ° Simulation time limit reached: {self.sim_time:.2f}s")
            return False
        
        print(f'sim time:{self.sim_time}')
        return True
    
    def update_execution_visualization(self):
        """Update the execution visualization"""
        if not hasattr(self, 'drone_point') or self.drone_point is None:
            return
        
        current_pos = self.state[0:3]
        
        # Update drone position
        self.drone_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Update trail
        if len(self.trail_positions) > 1:
            trail_array = np.array(self.trail_positions)
            self.drone_trail.set_data_3d(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2])
        
        # Update title with current info
        goal_dist = "N/A"
        if self.env.goal_point is not None:
            goal_dist = f"{np.linalg.norm(current_pos - np.array(self.env.goal_point)):.2f}m"
        
        self.ax.set_title(f'Phase 3: Executing - Time: {self.sim_time:.1f}s, Goal Dist: {goal_dist}')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # if len(self.state_history) % self.render_interval == 0:
        #     plot_filename = os.path.join(self.render_dir, f'plot_{len(self.state_history):05d}.png')
        #     self.fig.savefig(plot_filename)

    def run_live_simulation(self, start=None, goal=None):
        """
        Run the complete simulation showing all phases:
        1. RRT* planning (animated)
        2. B-spline trajectory generation
        3. Trajectory execution
        """
        print("ðŸŽ¬ Starting complete quadrotor simulation with all phases...")
        
        # Setup visualization
        self.setup_visualization()

        # self.preview_environment()
        
        # Phase 1: Animated RRT* Planning
        if not self.animated_rrt_planning(start, goal):
            return False
        
        # Phase 2: B-spline Trajectory Generation
        if not self.show_bspline_trajectory():
            return False
        
        # Phase 3: Trajectory Execution
        self.initialize_execution_phase()
        
        # Start execution
        self.simulation_active = True
        
        print("\nðŸš€ Executing trajectory...")
        print("ðŸ’¡ Close the plot window to stop simulation")
        
        execution_update_rate = 25  # Hz
        update_interval = 1.0 / execution_update_rate
        last_update_time = time.time()
        
        try:
            while self.simulation_active and plt.get_fignums():
                # Run simulation step
                continue_sim = self.simulation_step()
                
                # Update visualization at specified rate
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    self.update_execution_visualization()
                    last_update_time = current_time
                
                # Control real-time execution
                time.sleep(max(0, self.dt - (time.time() - current_time)))
                
                if not continue_sim:
                    break
                    
        except KeyboardInterrupt:
            print("\nâ¹ï¸ Simulation stopped by user")
        except Exception as e:
            print(f"\nðŸ’¥ Simulation error: {e}")
        finally:
            self.simulation_active = False
        
        # Final update
        self.update_execution_visualization()
        
        # Print results
        self._print_simulation_results()
        
        # create videos
        if self.rendering_enabled:              
            self.finalize_videos()              
        
        
        # Keep plot open
        print("\nSimulation complete. Close plot window to continue...")
        # plt.ioff()
        # plt.show()
        
        print("saveing video")
        ffmpeging_video(img_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/renders',
                        delineator='rgb_',
                        fps=10,
                        save_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs',

        )

        ffmpeging_video(img_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/renders',
                        delineator='plot_',
                        fps=10,
                        save_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs',

        )

        create_combined_video(
            dataset_dir='/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs',
            video_paths=['/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs/rgb_.mp4', 
            '/home/hkortus/RBE595/HandsOnAutonomousAerialVehicles/2b_project/group8_p2b/report_outputs/plot_.mp4'
            ]
        )

        return True
    
    # Keep all the other methods from before (environment drawing, results, etc.)
    def _draw_environment(self):
        """Draw the static environment elements"""
        # Draw boundary
        if self.env.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            vertices = self._create_cube_vertices(xmin, ymin, zmin, xmax, ymax, zmax)
            faces = self._create_cube_faces(vertices)
            
            for face in faces:
                face_array = np.array(face + [face[0]])
                self.ax.plot(face_array[:, 0], face_array[:, 1], face_array[:, 2], 
                           'k--', alpha=0.3, linewidth=1)
        
        # Draw obstacles
        for block_coords, block_color in self.env.blocks:
            vertices = self._create_cube_vertices(*block_coords)
            faces = self._create_cube_faces(vertices)
            
            poly3d = [[tuple(vertex) for vertex in face] for face in faces]
            self.ax.add_collection3d(Poly3DCollection(poly3d, 
                                                     facecolors=block_color, 
                                                     alpha=0.8,
                                                     edgecolors='black',
                                                     linewidths=0.5))
        
        # Draw start and goal
        if self.env.start_point:
            self.ax.scatter(*self.env.start_point, c='green', s=150, marker='s', 
                           edgecolors='black', linewidth=2, label='Start')
        if self.env.goal_point:
            self.ax.scatter(*self.env.goal_point, c='gold', s=150, marker='*', 
                           edgecolors='black', linewidth=2, label='Goal')
    
    def _create_cube_vertices(self, xmin, ymin, zmin, xmax, ymax, zmax):
        """Create cube vertices"""
        return [
            self.renderer.getObstaclePoses(np.array([xmin, ymin, zmin])), self.renderer.getObstaclePoses(np.array([xmax, ymin, zmin])), self.renderer.getObstaclePoses(np.array([xmax, ymax, zmin])), self.renderer.getObstaclePoses(np.array([xmin, ymax, zmin])),
            self.renderer.getObstaclePoses(np.array([xmin, ymin, zmax])), self.renderer.getObstaclePoses(np.array([xmax, ymin, zmax])), self.renderer.getObstaclePoses(np.array([xmax, ymax, zmax])), self.renderer.getObstaclePoses(np.array([xmin, ymax, zmax])),
        ]
    
    def _create_cube_faces(self, vertices):
        """Create cube faces"""
        return [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
            [vertices[1], vertices[5], vertices[6], vertices[2]],  # right
            [vertices[4], vertices[0], vertices[3], vertices[7]],  # left
        ]
    
    def _print_simulation_results(self):
        """Print simulation results summary"""
        print("\n" + "="*60)
        print("ðŸ COMPLETE SIMULATION RESULTS")
        print("="*60)
        
        final_pos = self.state[0:3]
        
        print(f"ðŸ“‹ PHASE SUMMARY:")
        print(f"   âœ… Phase 1: RRT* Planning - {len(self.planner.waypoints)} waypoints")
        print(f"   âœ… Phase 2: B-spline Trajectory - {len(self.controller.trajectory_points) if self.controller.trajectory_points is not None else 0} points")
        print(f"   âœ… Phase 3: Execution - {self.sim_time:.2f}s")
        
        if self.env.goal_point is not None:
            goal_distance = np.linalg.norm(final_pos - np.array(self.env.goal_point))
            start_goal_dist = np.linalg.norm(np.array(self.env.goal_point) - np.array(self.env.start_point))
            success_rate = max(0, (1 - goal_distance / start_goal_dist) * 100)
            
            print(f"\nðŸŽ¯ EXECUTION RESULTS:")
            print(f"   Status: {'âœ… GOAL REACHED' if self.goal_reached else 'âŒ NOT REACHED'}")
            print(f"   Final distance to goal: {goal_distance:.3f} m")
            print(f"   Success rate: {success_rate:.1f}%")
        
        print(f"\nðŸ“Š PERFORMANCE:")
        print(f"   Execution time: {self.sim_time:.2f} s")
        print(f"   Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
        print(f"   Path length: {len(self.trail_positions)} points")
        
        if self.controller.position_errors:
            mean_pos_error = np.mean(self.controller.position_errors)
            max_pos_error = np.max(self.controller.position_errors)
            print(f"   Tracking error: {mean_pos_error:.3f}m avg, {max_pos_error:.3f}m max")
        
        print("="*60)
    
    def save_results(self, filename_prefix='complete_simulation'):
        """Save complete simulation results"""
        import scipy.io
        
        data = {
            'time': np.array(self.time_history),
            'states': np.array(self.state_history),
            'controls': np.array(self.control_history),
            'rrt_waypoints': np.array(self.planner.waypoints),
            'bspline_trajectory': self.controller.trajectory_points,
            'executed_trail': np.array(self.trail_positions),
            'goal_reached': self.goal_reached,
            'sim_time': self.sim_time,
            'start_point': np.array(self.env.start_point),
            'goal_point': np.array(self.env.goal_point)
        }
        
        filename = f'./log/{filename_prefix}.mat'
        scipy.io.savemat(filename, data)
        print(f"ðŸ’¾ Complete simulation results saved to {filename}")