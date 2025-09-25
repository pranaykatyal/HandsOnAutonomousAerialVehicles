"""
RRT* (Rapidly-exploring Random Tree Star) Implementation for 3D Quadrotor Path Planning

This implementation is based on the OMPL RRTstar.cpp file and adapted for 3D quadrotor navigation.
The algorithm finds optimal paths in 3D space while avoiding obstacles.

Reference: OMPL geometric::RRTstar implementation

"""

import numpy as np
import math
import random
from typing import List, Optional, Tuple, Dict
import time
from environment import Environment3D  


class State:
    """
    Represents a state in 3D space (x, y, z coordinates)
    
    Corresponds to base::State in OMPL but simplified for 3D coordinates
    """
    def __init__(self, position: np.ndarray = None):
        if position is not None:
            self.position = position.copy()  # [x, y, z]
        else:
            self.position = np.zeros(3)
    
    def copy(self):
        return State(self.position.copy())
    
    def __str__(self):
        return f"State({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})"


class Motion:
    """
    Tree node representing a motion/state with parent-child relationships and costs
    
    Corresponds to ompl::geometric::RRTstar::Motion class 
    Stores state, parent pointer, children list, and cost information for tree structure
    """
    def __init__(self, state: State):
        self.state = state.copy()           # The actual 3D position
        self.parent: Optional['Motion'] = None      # Parent node in tree
        self.children: List['Motion'] = []          # Child nodes
        self.cost: float = 0.0             # Cost from start (g-cost)
        self.inc_cost: float = 0.0         # Incremental cost from parent
        self.in_goal: bool = False         # Whether this node reaches goal region
    
    def __str__(self):
        return f"Motion(pos={self.state.position}, cost={self.cost:.2f})"


class RRTStar:
    """
    RRT* path planner for 3D quadrotor navigation
    
    Based on ompl::geometric::RRTstar class implementation
    Main algorithm in solve() method corresponds to RRTstar.cpp line ~200+
    """
    
    def __init__(self, environment):
        """
        Initialize RRT* planner
        
        Args:
            environment: Environment3D object with collision checking methods
        """
        # Core algorithm components
        self.environment = environment  # 3D environment with obstacle checking
        self.motions: List[Motion] = []  # All tree nodes (corresponds to nn_ in OMPL)
        
        # Tree structure (corresponds to OMPL member variables line ~320+ in RRTstar.h)
        self.start_motions: List[Motion] = []     # Start state(s)
        self.goal_motions: List[Motion] = []      # Goal state(s) found
        self.best_goal_motion: Optional[Motion] = None  # Best goal found so far
        
        # Algorithm parameters (corresponds to OMPL parameters line ~320+ in RRTstar.h)
        self.goal_bias: float = 0.05       # Probability of sampling toward goal (goalBias_)
        self.max_distance: float = 1.0     # Maximum step size (maxDistance_)
        self.rewire_factor: float = 1.1    # Rewiring radius factor (rewireFactor_)
        self.goal_threshold: float = 0.3   # Distance to consider goal reached
        
        # Algorithm state
        self.iterations: int = 0           # Number of iterations completed
        self.best_cost: float = float('inf')  # Best solution cost found
        self.dimension: int = 3            # 3D space
        
        # RRT* mathematical constants (calculated in calculateRewiringLowerBounds())
        self.k_rrt: float = 0.0           # k-nearest constant
        self.r_rrt: float = 0.0           # radius constant
        
        # Initialize constants
        self._calculate_rewiring_bounds()


    def _calculate_rewiring_bounds(self):
        """
        Calculate RRT* rewiring radius bounds for optimality
        
        Corresponds to ompl::geometric::RRTstar::calculateRewiringLowerBounds() 
        in RRTstar.cpp line ~680+
        
        Mathematical foundation from RRT* paper:
        - k_rrt > 2^(d + 1) * e * (1 + 1/d) for k-nearest
        - r_rrt involves space measure and ball volume
        """
        self.k_rrt = self.rewire_factor * (2**(self.dimension + 1) * math.e * (1.0 + 1.0/self.dimension))
        
        if self.environment.boundary and len(self.environment.boundary) == 6:
            x_min, y_min, z_min, x_max, y_max, z_max = self.environment.boundary
            space_measure = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        else:
            space_measure = 1000.0
        
        unit_ball_volume = (4.0/3.0) * math.pi
        self.r_rrt = self.rewire_factor * ((2 * (1 + 1.0/self.dimension)) ** (1.0/self.dimension)) * ((space_measure / unit_ball_volume) ** (1.0/self.dimension))
        


    def set_goal_bias(self, bias: float):
        """Set probability of sampling toward goal (corresponds to setGoalBias())"""
        self.goal_bias = max(0.0, min(1.0, bias))
    
    
    def set_range(self, distance: float):
        """Set maximum step size (corresponds to setRange())"""
        self.max_distance = max(0.1, distance)
    
    
    def set_rewire_factor(self, factor: float):
        """Set rewiring aggressiveness (corresponds to setRewireFactor())"""
        self.rewire_factor = max(1.0, factor)
        self._calculate_rewiring_bounds()
    
    
    def set_goal_threshold(self, threshold: float):
        """Set distance threshold for reaching goal"""
        self.goal_threshold = max(0.1, threshold)


    def distance(self, state1: State, state2: State) -> float:
        """
        Calculate Euclidean distance between two states
        
        Corresponds to distanceFunction() in RRTstar.h line ~270
        Uses si_->distance(a->state, b->state) in OMPL
        """
        return np.linalg.norm(state1.position - state2.position)
        


    def sample_uniform(self, goal_state: State) -> State:
        """
        Sample a random state in the environment, with goal biasing
        
        Corresponds to sampleUniform() and goal biasing logic in solve() method
        RRTstar.cpp line ~230+ handles goal biasing with goalBias_ probability
        """

        env_bounds = self.environment.boundary # [xmin, ymin, zmin, xmax, ymax, zmax]
        if env_bounds and len(env_bounds) == 6:
            xmin, ymin, zmin, xmax, ymax, zmax = env_bounds
            if random.random() < self.goal_bias:
                # Sample near goal
                perturb = np.random.uniform(-self.goal_threshold, self.goal_threshold, size=3)
                sampled_pos = goal_state.position + perturb
                sampled_pos = np.clip(sampled_pos, [xmin, ymin, zmin], [xmax, ymax, zmax])
                return State(sampled_pos)
            else:
                # Uniform sampling in bounds
                sampled_pos = np.array([
                    random.uniform(xmin, xmax),
                    random.uniform(ymin, ymax),
                    random.uniform(zmin, zmax)
                ])
                return State(sampled_pos)
        else:
            # No bounds defined, return origin
            return State(np.zeros(3))



    def get_nearest(self, state: State) -> Optional[Motion]:
        """
        Find nearest node in tree to given state
        
        Corresponds to nn_->nearest(rmotion) in RRTstar.cpp line ~250
        OMPL uses nearest neighbor data structures, we use linear search
        """
        if not self.motions:
            return None
        return min(self.motions, key=lambda m: self.distance(m.state, state))



    def steer(self, from_state: State, to_state: State) -> State:
        """
        Steer from one state toward another, limited by max_distance
        
        Corresponds to interpolation logic in RRTstar.cpp line ~260+
        If distance > maxDistance_, interpolate to maxDistance_ away
        """
        distance = self.distance(from_state, to_state)
        if distance <= self.max_distance:
            return to_state.copy()
        else:
            direction = to_state.position - from_state.position
            unit_direction = direction / np.linalg.norm(direction)
            new_position = from_state.position + unit_direction * self.max_distance
            return State(new_position)
        


    def is_collision_free(self, from_state: State, to_state: State) -> bool:
        # Check if both endpoints are in free space
        if not self.environment.is_point_in_free_space(from_state.position):
            return False
        if not self.environment.is_point_in_free_space(to_state.position):
            return False
        
        # Check if the line between them is collision-free
        return self.environment.is_line_collision_free(from_state.position, to_state.position)


    def get_neighbors(self, motion: Motion) -> List[Motion]:
        """
        Find neighbors within rewiring radius of given motion
        
        Corresponds to getNeighbors() in RRTstar.cpp line ~400+
        Uses either k-nearest or radius-based neighbor search
        RRT* theory requires radius that shrinks as log(n)/n
        """
        if not self.motions:
            return []
        card_v = len(self.motions) + 1
        radius = min(self.max_distance, self.r_rrt * (math.log(card_v)/card_v)**(1/self.dimension))
        neighbors = []
        for m in self.motions:
            if m != motion and self.distance(motion.state, m.state) <= radius:
                neighbors.append(m)
        return neighbors


    def choose_parent(self, new_state: State, neighbors: List[Motion]) -> Tuple[Optional[Motion], float]:
        """
        Choose best parent from neighbors to minimize cost
        
        Corresponds to "Finding the nearest neighbor to connect to" section
        in RRTstar.cpp line ~300+. This is the "rewire incoming" step.
        """
        if not neighbors:
            return None, float('inf')
        
        best_parent = None
        best_cost = float('inf')
        for neighbor in neighbors:
            if self.is_collision_free(neighbor.state, new_state):
                cost = neighbor.cost + self.distance(neighbor.state, new_state)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = neighbor
        return best_parent, best_cost



    def rewire_neighbors(self, new_motion: Motion, neighbors: List[Motion]):
        for neighbor in neighbors:
            if neighbor == new_motion.parent:
                continue
                
            new_cost_for_neighbor = new_motion.cost + self.distance(new_motion.state, neighbor.state)
            
            # Check BOTH conditions before rewiring
            if new_cost_for_neighbor < neighbor.cost and self.is_collision_free(new_motion.state, neighbor.state):
                # Remove from old parent
                self.remove_from_parent(neighbor)
                
                # Set new parent
                neighbor.parent = new_motion
                neighbor.cost = new_cost_for_neighbor
                new_motion.children.append(neighbor)
                
                # Update descendants
                self.update_child_costs(neighbor)
         



    def update_child_costs(self, motion: Motion):
        """
        Recursively update costs for motion's children after rewiring
        
        Corresponds to updateChildCosts() in RRTstar.cpp line ~420+
        When a node's cost changes, all descendants must be updated
        """
        for child in motion.children:
            child.cost = motion.cost + child.inc_cost
            self.update_child_costs(child)
        


    def remove_from_parent(self, motion: Motion):
        """
        Remove motion from its parent's children list
        
        Corresponds to removeFromParent() in RRTstar.cpp line ~410+
        Used during rewiring to break old parent-child connections
        """
        if motion.parent is not None:
            motion.parent.children.remove(motion)


    def solve(self, start: State, goal: State, max_iterations: int = 2000, 
              max_time: float = 10.0) -> bool:
        """
        Main RRT* algorithm implementation
        
        Corresponds to solve() method in RRTstar.cpp line ~200+
        This is the core algorithm loop that builds the tree and finds optimal paths
        
        Args:
            start: Starting state
            goal: Goal state  
            max_iterations: Maximum number of iterations
            max_time: Maximum planning time in seconds
            
        Returns:
            True if path found, False otherwise
        """
        # Initialize algorithm (corresponds to setup in solve())
        self.iterations = 0
        self.motions.clear()
        self.start_motions.clear()
        self.goal_motions.clear()
        self.best_goal_motion = None
        self.best_cost = float('inf')
        
        start_time = time.time()
        
        # Add start state to tree (corresponds to start state handling)
        start_motion = Motion(start)
        start_motion.cost = 0.0
        self.motions.append(start_motion)
        self.start_motions.append(start_motion)
        
        print(f"RRT*: Starting search from {start} to {goal}")
        print(f"Max iterations: {max_iterations}, Max time: {max_time}s")
        
        # Main algorithm loop (corresponds to while(ptc == false) in RRTstar.cpp line ~230)
        while self.iterations < max_iterations and (time.time() - start_time) < max_time:
            self.iterations += 1
            
            # Progress reporting
            if self.iterations % 500 == 0:
                print(f"Iteration {self.iterations}, Tree size: {len(self.motions)}, "
                      f"Best cost: {self.best_cost:.3f}")
            
            # STEP 1: Sample random state (corresponds to goal biasing in RRTstar.cpp line ~235)
            # TODO: Implement sampling
            random_state = self.sample_uniform(goal)
            
            # STEP 2: Find nearest node (corresponds to nn_->nearest() line ~250)
            # TODO: Find nearest node
            nearest_motion = self.get_nearest(random_state)
            
            if nearest_motion is None:
                continue
            
            # STEP 3: Steer toward sample (corresponds to interpolation line ~260)
            # TODO: Steer from nearest toward random_state
            new_state = self.steer(nearest_motion.state, random_state)
            
            # STEP 4: Check collision (corresponds to checkMotion line ~270)
            # TODO: Check if path is collision-free
            if not self.is_collision_free(nearest_motion.state, new_state):
                continue
            
            # STEP 5: Find neighbors (corresponds to getNeighbors() line ~280)
            new_motion = Motion(new_state)
            # TODO: Get neighbors within rewiring radius
            neighbors = self.get_neighbors(new_motion)
            
            # STEP 6: Choose best parent (corresponds to parent selection line ~300+)
            # TODO: Find best parent among neighbors
            best_parent, best_cost = self.choose_parent(new_state, neighbors)
            
            if best_parent is None:
                # Use nearest as parent if no better option found
                best_parent = nearest_motion
                best_cost = nearest_motion.cost + self.distance(nearest_motion.state, new_state)
            
            # Set up new motion with best parent
            new_motion.parent = best_parent
            new_motion.cost = best_cost
            new_motion.inc_cost = self.distance(best_parent.state, new_state)
            
            # Add to tree
            self.motions.append(new_motion)
            best_parent.children.append(new_motion)
            
            # STEP 7: Rewire neighbors (corresponds to rewiring loop line ~370+)
            # TODO: Try to rewire neighbors through new_motion
            self.rewire_neighbors(new_motion, neighbors)
            
            # STEP 8: Check if goal reached (corresponds to goal checking line ~390+)
            distance_to_goal = self.distance(new_motion.state, goal)
            if distance_to_goal <= self.goal_threshold:
                new_motion.in_goal = True
                self.goal_motions.append(new_motion)
                
                # Update best solution if this is better
                if new_motion.cost < self.best_cost:
                    self.best_goal_motion = new_motion
                    self.best_cost = new_motion.cost
                    print(f"New best solution! Cost: {self.best_cost:.3f} at iteration {self.iterations}")
        
        # Algorithm complete
        elapsed_time = time.time() - start_time
        print(f"RRT* completed: {self.iterations} iterations in {elapsed_time:.2f}s")
        print(f"Tree size: {len(self.motions)} nodes")
        print(f"Solutions found: {len(self.goal_motions)}")
        
        if self.best_goal_motion:
            print(f"Best solution cost: {self.best_cost:.3f}")
            return True
        else:
            print("No solution found")
            return False


    def get_solution_path(self) -> List[State]:
        """
        Extract solution path from start to best goal
        
        Corresponds to path construction in solve() method line ~430+
        Traces back from goal to start through parent pointers
        """
        if not self.best_goal_motion:
            return []
        
        # TODO: Implement path extraction
        # Trace back from best_goal_motion to start using parent pointers
        # Return list of states from start to goal
        path = []
        motion = self.best_goal_motion
        while motion is not None:
            path.append(motion.state)
            motion = motion.parent
        path.reverse()  # From start to goal
        
        return path


    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all tree edges for visualization
        
        Returns:
            List of (parent_pos, child_pos) tuples
        """
        edges = []
        for motion in self.motions:
            if motion.parent is not None:
                edges.append((motion.parent.state.position, motion.state.position))
        return edges


    def get_statistics(self) -> Dict:
        """Get planning statistics"""
        return {
            'iterations': self.iterations,
            'tree_size': len(self.motions),
            'solutions_found': len(self.goal_motions),
            'best_cost': self.best_cost if self.best_goal_motion else float('inf'),
            'goal_bias': self.goal_bias,
            'max_distance': self.max_distance,
            'rewire_factor': self.rewire_factor
        }


# Example usage and integration with your project
if __name__ == "__main__":
    # This would integrate with your environment.py and path_planner.py
    
    # Placeholder for testing
    class MockEnvironment:
        def __init__(self):
            self.bounds = [(0, 10), (0, 10), (0, 10)]  # 3D bounds
        
        def is_point_in_free_space(self, point):
            # Simple bounds checking
            return all(self.bounds[i][0] <= point[i] <= self.bounds[i][1] for i in range(3))
        
        def is_line_collision_free(self, start, end):
            # Simplified line checking
            return (self.is_point_in_free_space(start) and 
                   self.is_point_in_free_space(end))
    
    # Test the planner structure
    env = MockEnvironment()
    planner = RRTStar(env)
    
    start = State(np.array([1, 1, 1]))
    goal = State(np.array([9, 9, 9]))
    
    print("RRT* Planner initialized successfully!")
    print("TODO: Implement the missing methods to complete the algorithm")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Statistics: {planner.get_statistics()}")