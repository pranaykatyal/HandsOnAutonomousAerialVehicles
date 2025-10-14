import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import all simulation components
try:
    from simulator import LiveQuadrotorSimulator
    from environment import Environment3D
    from path_planner import PathPlanner
    from trajectory_generator import TrajectoryGenerator
    from control import QuadrotorController
    
    # Import dynamics and parameters
    from quad_dynamics import model_derivative
    import tello as drone_params
    
    print("All modules imported successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are in the same directory:")
    print("  - quad_dynamics.py")
    print("  - tello.py") 
    print("  - control.py")
    print("  - environment.py")
    print("  - path_planner.py")
    print("  - trajectory_generator.py")
    print("  - simulator.py")
    sys.exit(1)

def run_environment_visualization(map_file):
    """Just visualize the environment with start/goal points"""
    print("Running environment visualization mode...")
    
    env = Environment3D()
    if not env.parse_map_file(map_file):
        print(f"Failed to load map file: {map_file}")
        return False
    
    # Generate start and goal points
    if not env.set_start_goal_points():
        print("Failed to generate start/goal points")
        return False
    
    # Visualize environment
    env.visualize_environment()
    
    print("Environment visualization completed!")
    return True

def run_path_planning_demo(map_file, start=None, goal=None):
    """Demonstrate path planning without full simulation"""
    print(" Running path planning demonstration...")
    
    # Initialize components
    env = Environment3D()
    if not env.parse_map_file(map_file):
        print(f"Failed to load map file: {map_file}")
        return False
    
    # Set start and goal
    if not env.set_start_goal_points(start, goal):
        print(" Failed to set start/goal points")
        return False
    
    # Plan path
    planner = PathPlanner(env)
    if not planner.plan_path():
        print("Path planning failed")
        return False
    
    # Visualize results
    fig = plt.figure(figsize=(15, 5))
    
    # Environment
    ax1 = fig.add_subplot(131, projection='3d')
    env.visualize_environment(ax1)
    ax1.set_title('Environment')
    
    # RRT* tree
    ax2 = fig.add_subplot(132, projection='3d')
    env.visualize_environment(ax2, show_start_goal=False)
    planner.visualize_tree(ax2)
    ax2.set_title('RRT* Tree')
    
    # Final path
    ax3 = fig.add_subplot(133, projection='3d')
    env.visualize_environment(ax3)
    if planner.waypoints:
        waypoints = np.array(planner.waypoints)
        ax3.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                'ro-', markersize=8, linewidth=3, label='Final Path')
        ax3.legend()
    ax3.set_title('Final Path')
    
    plt.tight_layout()
    plt.show()
    
    print("Path planning demonstration completed!")
    return True

def run_trajectory_demo(map_file, start=None, goal=None):
    """Demonstrate trajectory generation"""
    print("Running trajectory generation demonstration...")
    
    # Initialize and plan path
    env = Environment3D()
    if not env.parse_map_file(map_file):
        return False
    
    if not env.set_start_goal_points(start, goal):
        return False
    
    planner = PathPlanner(env)
    if not planner.plan_path():
        return False
    
    # Generate trajectory
    traj_gen = TrajectoryGenerator(planner.waypoints)
    result = traj_gen.generate_bspline_trajectory(num_points=200)
    
    if result[0] is not None:
        trajectory_points, time_points, velocities, accelerations = result
        
        # Visualize trajectory
        traj_gen.visualize_trajectory(trajectory_points, velocities, accelerations)
        
        print("Trajectory generation demonstration completed!")
        return True
    else:
        print("Trajectory generation failed")
        return False

def run_live_simulation(map_file, start=None, goal=None, save_data=False):
    """Run the live real-time simulation"""
    print("Running live quadrotor simulation...")
    
    # Initialize simulator
    sim = LiveQuadrotorSimulator(map_file)
    
    # Run simulation
    success = sim.run_live_simulation(start=start, goal=goal)
    
    if not success:
        print("Live simulation failed")
        return False
    
    # Save data if requested
    if save_data:
        sim.save_results()
    
    print("Live simulation completed!")
    return True

def run_offline_simulation(map_file, start=None, goal=None, save_data=False):
    """Run offline simulation with detailed analysis plots"""
    print("Running offline simulation with analysis...")
    
    # This would use a different simulator class for offline analysis
    # For now, we'll run the live simulation and then show analysis
    sim = LiveQuadrotorSimulator(map_file)
    
    # Initialize without live visualization
    if not sim.initialize_simulation(start, goal):
        return False
    
    print("🏃 Running fast simulation...")
    sim.simulation_active = True
    
    try:
        while sim.sim_time < sim.max_sim_time and not sim.goal_reached:
            if not sim.simulation_step():
                break
    except Exception as e:
        print(f"Simulation error: {e}")
        return False
    
    # Print results
    sim._print_simulation_results()
    
    # Create analysis plots
    if len(sim.state_history) > 0:
        print("Creating analysis plots...")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D path
        ax1 = fig.add_subplot(231, projection='3d')
        sim.env.visualize_environment(ax1)
        if sim.trail_positions:
            trail = np.array(sim.trail_positions)
            ax1.plot(trail[:, 0], trail[:, 1], trail[:, 2], 'g-', linewidth=3, label='Actual Path')
        ax1.set_title('Flight Path')
        ax1.legend()
        
        # Position vs time
        ax2 = fig.add_subplot(232)
        times = np.array(sim.time_history)
        states = np.array(sim.state_history)
        ax2.plot(times, states[:, 0], 'r-', label='X')
        ax2.plot(times, states[:, 1], 'g-', label='Y') 
        ax2.plot(times, states[:, 2], 'b-', label='Z')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True)
        
        # Velocity vs time
        ax3 = fig.add_subplot(233)
        ax3.plot(times, states[:, 3], 'r-', label='Vx')
        ax3.plot(times, states[:, 4], 'g-', label='Vy')
        ax3.plot(times, states[:, 5], 'b-', label='Vz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # Control inputs
        ax4 = fig.add_subplot(234)
        if sim.control_history:
            controls = np.array(sim.control_history)
            ax4.plot(times, controls[:, 0], label='Motor 1')
            ax4.plot(times, controls[:, 1], label='Motor 2')
            ax4.plot(times, controls[:, 2], label='Motor 3')
            ax4.plot(times, controls[:, 3], label='Motor 4')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Control Input')
            ax4.set_title('Motor Commands')
            ax4.legend()
            ax4.grid(True)
        
        # Tracking errors
        ax5 = fig.add_subplot(235)
        if sim.controller.position_errors:
            ax5.plot(times, sim.controller.position_errors, 'r-', linewidth=2)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Position Error (m)')
            ax5.set_title('Position Tracking Error')
            ax5.grid(True)
        
        # Goal distance over time
        ax6 = fig.add_subplot(236)
        if sim.env.goal_point is not None:
            goal_distances = []
            for state in sim.state_history:
                dist = np.linalg.norm(state[0:3] - np.array(sim.env.goal_point))
                goal_distances.append(dist)
            ax6.plot(times, goal_distances, 'purple', linewidth=2)
            ax6.axhline(y=sim.goal_tolerance, color='red', linestyle='--', label='Goal Tolerance')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Distance to Goal (m)')
            ax6.set_title('Goal Distance vs Time')
            ax6.legend()
            ax6.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Save data if requested
    if save_data:
        sim.save_results()
    
    print("Offline simulation completed!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Quadrotor Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py map1.txt                           # Live real-time simulation
  python main.py map1.txt --visualize-only          # Just show environment  
  python main.py map1.txt --path-planning-only      # Show path planning process
  python main.py map1.txt --trajectory-only         # Show trajectory generation
  python main.py map1.txt --start 1 2 3 --goal 8 8 4  # Custom start/goal
  python main.py map1.txt --offline --save-data     # Offline analysis with data saving
        """)
    
    parser.add_argument('map_file', help='Path to the map file (e.g., map1.txt)')
    
    # Mode selection
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only visualize the environment')
    parser.add_argument('--path-planning-only', action='store_true',
                       help='Demonstrate path planning only')
    parser.add_argument('--trajectory-only', action='store_true',
                       help='Demonstrate trajectory generation only')
    parser.add_argument('--offline', action='store_true',
                       help='Run offline simulation with analysis plots')
    
    # Start/goal specification
    parser.add_argument('--start', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Start position [x y z]')
    parser.add_argument('--goal', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Goal position [x y z]')
    
    # Options
    parser.add_argument('--save-data', action='store_true',
                       help='Save simulation data to files')
    
    args = parser.parse_args()
    
    print("Quadrotor Simulation Framework")
    print("=" * 50)
    print(f"Map file: {args.map_file}")
    
    if args.start:
        print(f"Start position: {args.start}")
    if args.goal:
        print(f"Goal position: {args.goal}")
    
    print("=" * 50)
    
    # Determine mode and run appropriate function
    try:
        if args.visualize_only:
            success = run_environment_visualization(args.map_file)
        
        elif args.path_planning_only:
            success = run_path_planning_demo(args.map_file, args.start, args.goal)
        
        elif args.trajectory_only:
            success = run_trajectory_demo(args.map_file, args.start, args.goal)
        
        elif args.offline:
            # Offline simulation with analysis
            success = run_offline_simulation(
                args.map_file,
                start=args.start,
                goal=args.goal,
                save_data=args.save_data
            )
        
        else:
            # Default: Live real-time simulation
            success = run_live_simulation(
                args.map_file,
                start=args.start,
                goal=args.goal,
                save_data=args.save_data
            )
        
        if success:
            print("\nSimulation framework completed successfully!")
        else:
            print("\n Simulation framework encountered errors!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n Simulation interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()