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


def main():
    parser = argparse.ArgumentParser(
        description='Quadrotor Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py map1.txt                           # Live real-time simulation

        """)
    
    #parser.add_argument('map_file', help='Path to the map file (e.g., map1.txt)')

    # Options
    #parser.add_argument('--save-data', action='store_true',
    #                  help='Save simulation data to files')
    
    #args = parser.parse_args()
    
    print("Quadrotor Simulation Framework")
    print("=" * 50)
    #print(f"Map file: {args.map_file}")
    
    print("=" * 50)
    
    # Determine mode and run appropriate function
    try:
        # Default: Live real-time simulation
        success = run_live_simulation(
            './maps/map2.txt',  #args.map_file,
            start=None,
            goal=None,
            #save_data=args.save_data
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