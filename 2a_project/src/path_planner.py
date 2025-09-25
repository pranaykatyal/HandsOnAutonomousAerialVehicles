import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rrtstar import RRTStar, State

class RRTNode:
    """Node for RRT* tree"""
    def __init__(self, position, parent=None):
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.cost = 0.0
        self.children = []

class PathPlanner(RRTStar):
    """
    Robust RRT* implementation for 3D path planning
    Inherits from RRTStar and provides interface compatibility with the simulation framework
    """
    
    def __init__(self, environment):
        super().__init__(environment)
        self.env = environment  # Keep reference for compatibility
        self.waypoints = []
        self.tree_nodes = []
        
        # Set default RRT* parameters optimized for 3D quadrotor planning
        self.set_goal_bias(0.15)  # 15% bias towards goal
        self.set_range(1.5)       # Step size
        self.set_goal_threshold(1.0)  # Goal radius
        self.set_rewire_factor(1.2)   # Slightly more aggressive rewiring
    
    def plan_path(self, start_pos, goal_pos, max_time=15.0, max_iterations=3000):
        """
        Main path planning function
        
        Args:
            start_pos: [x, y, z] starting position
            goal_pos: [x, y, z] goal position  
            max_time: Maximum planning time in seconds
            max_iterations: Maximum number of iterations
            
        Returns:
            bool: True if path found, False otherwise
        """
        print(f"PathPlanner: Planning from {start_pos} to {goal_pos}")
        
        # Convert to State objects
        start_state = State(np.array(start_pos))
        goal_state = State(np.array(goal_pos))
        
        # Run RRT* planning
        success = self.solve(start_state, goal_state, 
                            max_iterations=max_iterations, max_time=max_time)
        
        if success:
            # Extract path and convert back to position arrays
            path_states = self.get_solution_path()
            self.waypoints = [state.position.copy() for state in path_states]
            
            # Convert Motion objects to RRTNode for visualization compatibility
            self._convert_tree_for_visualization()
            
            print(f"PathPlanner: Found path with {len(self.waypoints)} waypoints")
            print(f"PathPlanner: Path cost: {self.best_cost:.3f}")
            return True
        else:
            print("PathPlanner: No path found")
            self.waypoints = []
            self.tree_nodes = []
            return False
    
    def _convert_tree_for_visualization(self):
        """Convert RRTStar Motion objects to RRTNode objects for visualization"""
        self.tree_nodes = []
        motion_to_node = {}
        
        # Create RRTNode for each Motion
        for motion in self.motions:
            node = RRTNode(motion.state.position)
            node.cost = motion.cost
            self.tree_nodes.append(node)
            motion_to_node[motion] = node
        
        # Set parent-child relationships
        for motion in self.motions:
            if motion.parent and motion in motion_to_node:
                node = motion_to_node[motion]
                parent_node = motion_to_node[motion.parent]
                node.parent = parent_node
                parent_node.children.append(node)
    
    def simplify_path(self, waypoints=None, tolerance=0.1):
        """
        Simplify the path by removing redundant waypoints
        
        Args:
            waypoints: List of waypoints to simplify (uses self.waypoints if None)
            tolerance: Maximum deviation tolerance
            
        Returns:
            List of simplified waypoints
        """
        if waypoints is None:
            waypoints = self.waypoints
            
        if len(waypoints) < 3:
            return waypoints
        
        simplified = [waypoints[0]]  # Always keep start
        
        i = 0
        while i < len(waypoints) - 1:
            # Look ahead to find the farthest point we can reach directly
            j = len(waypoints) - 1
            while j > i + 1:
                if self.env.is_line_collision_free(waypoints[i], waypoints[j]):
                    simplified.append(waypoints[j])
                    i = j
                    break
                j -= 1
            else:
                # If we can't skip ahead, move to next point
                i += 1
                if i < len(waypoints):
                    simplified.append(waypoints[i])
        
        return simplified
    
    def find_nearest_node(self, tree, point):
        """
        Find nearest node in tree to given point
        Compatibility function for existing code
        """
        if not tree:
            return None
        distances = [np.linalg.norm(node.position - point) for node in tree]
        return tree[np.argmin(distances)]
    
    def is_path_valid(self, start_pos, end_pos):
        """
        Check if path between two positions is valid
        Compatibility function for existing code
        """
        return self.env.is_line_collision_free(start_pos, end_pos)
    
    # def distance(self, pos1, pos2):
    #     """
    #     Calculate distance between two positions
    #     Compatibility function for existing code
    #     """
    #     return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def steer_numpy(self, from_pos, to_pos, step_size):
        """
        Steer from one position toward another with given step size
        Compatibility function for existing code
        """
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance <= step_size:
            return to_pos
        
        unit_direction = direction / distance
        return from_pos + unit_direction * step_size
    
    def find_near_nodes(self, tree, position, radius):
        """
        Find nodes within given radius of position
        Compatibility function for existing code
        """
        near_nodes = []
        for node in tree:
            if np.linalg.norm(node.position - position) <= radius:
                near_nodes.append(node)
        return near_nodes
    
    def choose_parent_compat(self, near_nodes, new_position):
        """
        Choose best parent from near nodes
        Compatibility function for existing code
        """
        if not near_nodes:
            return None, float('inf')
        
        best_parent = None
        best_cost = float('inf')
        
        for node in near_nodes:
            if self.is_path_valid(node.position, new_position):
                cost = node.cost + np.linalg.norm(np.array(node.position) - np.array(new_position))
                if cost < best_cost:
                    best_cost = cost
                    best_parent = node
        
        return best_parent, best_cost
    
    def rewire_tree_compat(self, tree, new_node, near_nodes):
        """
        Rewire tree through new node if it provides better paths
        Compatibility function for existing code
        """
        for node in near_nodes:
            if node == new_node.parent:
                continue
                
            new_cost = new_node.cost + np.linalg.norm(np.array(new_node.position) - np.array(node.position))
            if new_cost < node.cost and self.is_path_valid(new_node.position, node.position):
                # Remove from old parent
                if node.parent:
                    node.parent.children.remove(node)
                
                # Set new parent
                node.parent = new_node
                node.cost = new_cost
                new_node.children.append(node)
                
                # Update descendants (simplified version)
                self._update_descendants_cost_compat(node)
    
    def _update_descendants_cost_compat(self, node):
        """Update costs for all descendants of a node (compatibility version)"""
        for child in node.children:
            child.cost = node.cost + np.linalg.norm(np.array(node.position) - np.array(child.position))
            self._update_descendants_cost_compat(child)
    
    def extract_path(self, goal_node):
        """
        Extract path from start to goal node
        Compatibility function for existing code
        """
        path = []
        current = goal_node
        while current is not None:
            path.append(current.position.copy())
            current = current.parent
        path.reverse()
        return path
    
    def get_path_length(self, waypoints=None):
        """Calculate total path length"""
        if waypoints is None:
            waypoints = self.waypoints
            
        if len(waypoints) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(waypoints)):
            total_length += np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i-1]))
        
        return total_length
    
    def visualize_tree(self, ax=None):
        """Visualize the RRT* tree"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False
        
        # Draw tree edges
        for node in self.tree_nodes:
            if node.parent is not None:
                ax.plot([node.parent.position[0], node.position[0]],
                       [node.parent.position[1], node.position[1]],
                       [node.parent.position[2], node.position[2]],
                       'b-', alpha=0.3, linewidth=0.5)
        
        # Draw tree nodes
        if self.tree_nodes:
            positions = np.array([node.position for node in self.tree_nodes])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='blue', s=10, alpha=0.6)
        
        # Draw final path
        if len(self.waypoints) > 0:
            waypoints = np.array(self.waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   'ro-', markersize=8, linewidth=3, label='RRT* Path')
        
        # Draw start and goal if available
        if hasattr(self.env, 'start_point') and self.env.start_point:
            ax.scatter(*self.env.start_point, c='green', s=100, marker='s', 
                      label='Start', edgecolors='black', linewidth=2)
        if hasattr(self.env, 'goal_point') and self.env.goal_point:
            ax.scatter(*self.env.goal_point, c='red', s=100, marker='*', 
                      label='Goal', edgecolors='black', linewidth=2)
        
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('RRT* Tree and Path')
            ax.legend()
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def get_planning_statistics(self):
        """Get detailed planning statistics"""
        stats = self.get_statistics()  # From RRTStar parent class
        
        # Add path-specific statistics
        if self.waypoints:
            stats.update({
                'path_length': self.get_path_length(),
                'waypoints_count': len(self.waypoints),
                'path_found': True
            })
        else:
            stats.update({
                'path_length': 0.0,
                'waypoints_count': 0,
                'path_found': False
            })
        
        return stats
    
    def reset(self):
        """Reset the planner state"""
        self.waypoints = []
        self.tree_nodes = []
        self.motions.clear()
        self.start_motions.clear()
        self.goal_motions.clear()
        self.best_goal_motion = None
        self.best_cost = float('inf')
        self.iterations = 0


# Example usage and testing
if __name__ == "__main__":
    # Mock environment for testing
    class MockEnvironment:
        def __init__(self):
            self.boundary = [0, 0, 0, 10, 10, 10]  # [xmin, ymin, zmin, xmax, ymax, zmax]
            self.start_point = [1, 1, 1]
            self.goal_point = [9, 9, 9]
        
        def is_point_in_free_space(self, point):
            # Simple bounds checking
            return (self.boundary[0] <= point[0] <= self.boundary[3] and
                   self.boundary[1] <= point[1] <= self.boundary[4] and
                   self.boundary[2] <= point[2] <= self.boundary[5])
        
        def is_line_collision_free(self, start, end):
            # Simplified line checking
            return (self.is_point_in_free_space(start) and 
                   self.is_point_in_free_space(end))
    
    # Test the planner
    print("Testing PathPlanner...")
    env = MockEnvironment()
    planner = PathPlanner(env)
    
    # Plan a path
    success = planner.plan_path([1, 1, 1], [9, 9, 9])
    
    if success:
        print(f"Path planning successful!")
        print(f"Statistics: {planner.get_planning_statistics()}")
        
        # Test path simplification
        original_waypoints = len(planner.waypoints)
        simplified_waypoints = planner.simplify_path()
        print(f"Path simplified from {original_waypoints} to {len(simplified_waypoints)} waypoints")
        
        # Visualize if matplotlib available
        try:
            planner.visualize_tree()
        except:
            print("Visualization not available")
    else:
        print("Path planning failed")