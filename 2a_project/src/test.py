from environment import Environment3D
from path_planner import PathPlanner
from simulator import LiveQuadrotorSimulator
import numpy as np

def test_simulator():
    sim = LiveQuadrotorSimulator(map_file='./maps/map4.txt')

def main():
    # Check matplotlib backend
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    
    print("Testing RRT* Path Planner")
    print("=" * 40)
    
    # Load environment
    env = Environment3D()
    success = env.parse_map_file(filename='./maps/map1.txt')
    
    if not success:
        print("ERROR: Failed to load map file")
        return
    
    print(f"Environment loaded: {len(env.blocks)} obstacles")
    print(f"Boundary: {env.boundary}")
    
    # Create and configure planner
    planner = PathPlanner(env)
    planner.set_goal_bias(0.1)
    planner.set_range(2.0)
    planner.set_goal_threshold(0.5)
    
    # Plan path
    print(f"\nPlanning path...")
    start_time = time.time()
    success = planner.plan_path(env.start_point, env.goal_point, 
                               max_time=0.02, max_iterations=2000)
    planning_time = time.time() - start_time
    
    if success:
        print(f"SUCCESS! Found path in {planning_time:.1f}s")
        stats = planner.get_planning_statistics()
        print(f"Path cost: {stats['best_cost']:.2f}")
        print(f"Waypoints: {stats['waypoints_count']}")
        print(f"Tree size: {stats['tree_size']}")

        
        # Enable interactive mode
        plt.ion()
        
        # Create enhanced visualization
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Call the visualization method
        planner.visualize_tree(ax)
        
        # Set a good initial viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Enable mouse interaction explicitly
        ax.mouse_init()
        
        # Keep the plot open for interaction
        plt.show(block=True)
        
    else:
        print(f"FAILED: No path found in {planning_time:.1f}s")

if __name__ == "__main__":
    env = Environment3D()
    env.parse_map_file(filename='./maps/map4.txt')
    env.is_point_in_free_space([12,12,12])
    print(f" line is collision free: {env.is_line_collision_free([0,0,0],[-6,-6,-6])}")
    env.get_environment_info()
    path = PathPlanner(environment=env)
    # print("new step", path.step(start=np.array([1,1,1]), end=np.array([5,1,1])))
    path.plan()
    path.visualize_tree()
