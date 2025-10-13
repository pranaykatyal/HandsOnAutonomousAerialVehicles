import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

Position = np.ndarray

class RRTNode:
    """Node for RRT* tree"""
    def __init__(self, position, parent=None):
        self.position: Position = np.array(position, dtype=float)
        self.parent:RRTNode|None = parent
        self.cost:float = 0.0
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
        self.tree_nodes:List[RRTNode] = []
        
        # RRT* parameters
        self.max_iterations = 3000
        self.simplify_iterations = 1000
        self.step_size = 1.0
        self.goal_radius = 0.25
        self.search_radius = 2.5
        self.goal_bias = 0.15  # 15% bias towards goal
        
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
    
    def plan(self):
        for i in range(self.max_iterations):
            if np.random.rand() < self.goal_bias:
                random_point: Position = self.env.goal_point
            else:
                random_point: Position = self.env.generate_random_free_point()
            
            nearest_node: RRTNode = self.find_nearest_node(node_list=self.tree_nodes, point=random_point)
            node_position: Position = self.steer(start=nearest_node.position,end=random_point, step_size=self.step_size)
            
            if not self.env.is_line_collision_free(p1=nearest_node.position, p2=node_position):
                continue

            neighbors: List[RRTNode] = self.find_near_nodes(node_list=self.tree_nodes, point=node_position, search_radius=self.search_radius)
            new_node:RRTNode = self.choose_parent(neighbors=neighbors,nearest_node=nearest_node,new_position=node_position)
           
            self.rewire_tree(neighbors=neighbors, new_node=new_node)

            if self.reached_goal(new_node=new_node):
                # self.extract_path(final_node=new_node)
                self.waypoints = self.extract_path(final_node=new_node)
                return True

    #### TODO - Store the final path in self.waypoints as a list of 3D points ##################################
    def extract_path(self, final_node:RRTNode):
        final_path:List[Position] = []
        node:RRTNode = final_node
        while node is not None:
            final_path.append(node.position)
            node = node.parent
        return final_path[::-1]

    #### TODO - Add is_path_valid by taking the dot product and rejecting large angles #################################################################



    def find_nearest_node(self, 
                        node_list: List[RRTNode], 
                        point: Position
                        ) -> RRTNode:
        
        distances = []
        for node in node_list:
            distance = np.linalg.norm(node.position - point)
            # print(f"got distance {distance}")
            distances.append(distance)
        #     print("distance list", distances)
        # print("lengin of node list", len(distances))
        if len(distances) > 1:
            nearest_idx = np.argmin(distances)
            return node_list[nearest_idx]
        else:
            return node_list[0]

    def steer(self, 
            start: Position, 
            end: Position, 
            step_size) -> Position:
        
        direction = end - start
        distance = np.linalg.norm(direction)
        if distance <= step_size:
            return end
        unit_direction = direction / distance
        node_position = start + step_size * unit_direction
        return node_position
    
    def find_near_nodes(self, 
                        node_list: List[RRTNode], 
                        point: Position, 
                        search_radius
                        ) -> List[RRTNode]:
        neighbors: List[RRTNode] = []
        for node in node_list:
            if np.linalg.norm(node.position - point) <= search_radius:
                neighbors.append(node)
        return neighbors

    def choose_parent(self, 
                    neighbors: List[RRTNode], 
                    nearest_node: RRTNode, 
                    new_position: Position
                    ) -> RRTNode:
        
        best_parent: RRTNode = nearest_node
        best_cost = nearest_node.cost + self.distance(nearest_node.position, new_position)
        for neighbor in neighbors:
            neighbor_to_position_distance = self.distance(neighbor.position, new_position)
            if neighbor.cost + neighbor_to_position_distance <= best_cost and self.env.is_line_collision_free(p1=neighbor.position,p2=new_position):
                best_parent = neighbor
                best_cost = neighbor.cost + neighbor_to_position_distance
        new_node = RRTNode(position=new_position, parent=best_parent)
        new_node.cost = best_cost
        best_parent.children.append(new_node)
        self.tree_nodes.append(new_node)
        return new_node
    
    def rewire_tree(self, new_node: RRTNode, neighbors: List[RRTNode]) -> None:
        for neighbor in neighbors:
            if neighbor == new_node.parent:  # Skip parent
                continue
            new_cost = new_node.cost + self.distance(neighbor.position, new_node.position)
            # Fix: Compare with neighbor's current cost, not new_node's cost
            if new_cost < neighbor.cost and self.env.is_line_collision_free(p1=neighbor.position, p2=new_node.position):
                # Remove neighbor from its old parent's children list
                if neighbor.parent is not None:
                    neighbor.parent.children.remove(neighbor)
                
                # Set new parent
                neighbor.parent = new_node
                neighbor.cost = new_cost
                new_node.children.append(neighbor)
                
                # Propagate cost changes to all descendants
                self._update_descendants_cost(neighbor)

    def _update_descendants_cost(self, node: RRTNode) -> None:
        """Recursively update costs for all descendants of a node"""
        for child in node.children:
            child.cost = node.cost + self.distance(node.position, child.position)
            self._update_descendants_cost(child)

    def reached_goal(self, new_node: RRTNode) -> bool:
        distance_error = np.linalg.norm(new_node.position - self.env.goal_point)
        if distance_error <= self.goal_radius:
            new_node.position = self.env.goal_point
            return True
        else:
            return False

    def distance(self, new_position: Position, goal_point:Position):
        return np.linalg.norm(np.array(new_position) - np.array(goal_point))
    
    def simplify_path(self, path_waypoints: List[Position], max_skip_distance=3.0, boundary_threshold=2.0):
        if len(path_waypoints) <= 2:
            return path_waypoints
        
        def is_near_boundary(point):
            """Check if point is close to map boundaries"""
            if not self.env.boundary:
                return False
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            return (point[0] - xmin < boundary_threshold or
                    xmax - point[0] < boundary_threshold or
                    point[1] - ymin < boundary_threshold or
                    ymax - point[1] < boundary_threshold or
                    point[2] - zmin < boundary_threshold or
                    zmax - point[2] < boundary_threshold)
    
        def is_safe_shortcut(p1, p2):
            """Check if shortcut maintains safe clearance"""
            # Check line collision
            if not self.env.is_line_collision_free(p1, p2):
                return False
            
            # Sample intermediate points along the shortcut
            num_samples = max(3, int(np.linalg.norm(p2 - p1) / 0.3))
            for i in range(1, num_samples):
                t = i / num_samples
                intermediate = p1 + t * (p2 - p1)
                
                # Verify intermediate point is in free space
                if not self.env.is_point_in_free_space(intermediate):
                    return False
                
                # Extra check: ensure minimum clearance from obstacles
                # Sample a small sphere around the point
                clearance_samples = [
                    intermediate + np.array([0.15, 0, 0]),
                    intermediate + np.array([-0.15, 0, 0]),
                    intermediate + np.array([0, 0.15, 0]),
                    intermediate + np.array([0, -0.15, 0]),
                    intermediate + np.array([0, 0, 0.15]),
                    intermediate + np.array([0, 0, -0.15])
                ]
                
                for sample in clearance_samples:
                    if not self.env.is_point_in_free_space(sample):
                        return False
            
            return True
        
        simplified_path = [path_waypoints[0]]
        current_idx = 0
        
        while current_idx < len(path_waypoints) - 1:
            furthest_visible_idx = current_idx + 1
            
            for i in range(current_idx + 2, len(path_waypoints)):
                # If waypoint is near boundary, don't skip it
                if is_near_boundary(path_waypoints[i]):
                    if i - current_idx > 1:  # Keep at least this waypoint
                        furthest_visible_idx = i
                    break
                
                # Check distance constraint
                distance = np.linalg.norm(path_waypoints[current_idx] - path_waypoints[i])
                if distance > max_skip_distance:
                    break
                
                # Use safer collision checking
                if is_safe_shortcut(path_waypoints[current_idx], path_waypoints[i]):
                    furthest_visible_idx = i
                else:
                    break
            
            current_idx = furthest_visible_idx
            simplified_path.append(path_waypoints[current_idx])
        
        if not np.allclose(simplified_path[-1], path_waypoints[-1]):
            simplified_path.append(path_waypoints[-1])
        
        return simplified_path

    ############################################################################################################
    def visualize_tree(self, ax=None):
        """Visualize the RRT* tree"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False

        ax.plot(self.env.start_point[0],self.env.start_point[1], self.env.start_point[2], 'ro-', markersize=12, linewidth=3, label='Start')
        ax.plot(self.env.goal_point[0],self.env.goal_point[1], self.env.goal_point[2], 'ro-', markersize=12, linewidth=3, label='End')

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