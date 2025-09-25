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

class PathPlanner:
    """
    Robust RRT* implementation for 3D path planning
    """
    
    def __init__(self, environment):
        self.env = environment
        self.waypoints = []
        self.tree_nodes:List[RRTNode] = []
        
        # RRT* parameters
        self.max_iterations = 3000
        self.step_size = 1.0
        self.goal_radius = 1.0
        self.search_radius = 2.5
        self.goal_bias = 0.15  # 15% bias towards goal
        
    
    def plan(self):
        root=RRTNode(self.env.start_point)
        self.tree_nodes.append(root)

        for i in range(self.max_iterations):
            if np.random.rand() < self.goal_bias:
                random_point: Position = self.env.goal_point
            else:
                random_point: Position = self.env.generate_random_free_point()
            
            nearest_node: RRTNode = self.get_nearest_node(node_list=self.tree_nodes, point=random_point)
            node_position: Position = self.step(start=nearest_node.position,end=random_point)
            
            if self.env.is_line_collision_free(p1=nearest_node.position, p2=node_position):
                neighbors: List[RRTNode] = self.get_neighbors(node_list=self.tree_nodes, point=node_position)
                new_node:RRTNode = self.set_best_parent(neighbors=neighbors,nearest_node=nearest_node,new_position=node_position)
                self.tree_nodes.append(new_node)
                self.rewire(neighbors=neighbors, new_node=new_node)

                if self.reached_goal(new_node=new_node):
                    self.save_path_to_waypoint(final_node=new_node)
                    self.waypoints = self.save_path_to_waypoint(final_node=new_node)
                    return True

    #### TODO - Store the final path in self.waypoints as a list of 3D points ##################################
    def save_path_to_waypoint(self, final_node:RRTNode):
        final_path:List[Position] = []
        node:RRTNode = final_node
        while node is not None:
            final_path.append(node.position)
            node = node.parent
        return final_path[::-1]

    #### TODO - Add member functions as needed #################################################################

    def get_nearest_node(self, node_list: List[RRTNode], point: Position):
        distances = []
        for node in node_list:
            distance = np.linalg.norm(node.position - point)
            distances.append(distance)
        nearest_idx = np.argmin(distances)
        return node_list[nearest_idx]

    def step(self, start: Position, end: Position):
        direction = end - start
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return end
        unit_direction = direction / distance
        node_position = start + self.step_size * unit_direction
        return node_position
    
    def get_neighbors(self, node_list: List[RRTNode], point: Position):
        neighbors: List[RRTNode] = []
        for node in node_list:
            if np.linalg.norm(node.position - point) <= self.search_radius:
                neighbors.append(node)
        return neighbors

    def set_best_parent(self, neighbors: List[RRTNode], nearest_node: RRTNode, new_position: Position):
        best_parent: RRTNode = nearest_node
        best_cost = nearest_node.cost + np.linalg.norm(nearest_node.position - new_position)
        for neighbor in neighbors:
            neighbor_to_position_distance = np.linalg.norm(neighbor.position - new_position)
            if neighbor.cost + neighbor_to_position_distance <= best_cost:
                best_parent = neighbor
                best_cost = neighbor.cost + neighbor_to_position_distance
        new_node = RRTNode(position=new_position, parent=best_parent)
        new_node.cost = best_cost
        return new_node

    def rewire(self, neighbors: List[RRTNode], new_node: RRTNode):
        for neighbor in neighbors:
            if neighbor == new_node.parent:  # Skip parent
                continue
            new_cost = new_node.cost + np.linalg.norm(neighbor.position - new_node.position)
            if new_cost < new_node.cost and self.env.is_line_collision_free(p1=neighbor.position,p2=new_node.position):
                neighbor.parent = new_node
                neighbor.cost = new_cost

    def reached_goal(self, new_node: RRTNode):
        distance_error = np.linalg.norm(new_node.position - self.env.goal_point)
        if distance_error <= self.goal_radius:
            return True

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
        
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('RRT* Tree and Path')
            ax.legend()
            plt.tight_layout()
            plt.show()
        
        return ax