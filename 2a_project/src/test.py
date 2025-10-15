from environment import Environment3D
from path_planner import PathPlanner
from simulator import LiveQuadrotorSimulator
import numpy as np

def test_simulator():
    sim = LiveQuadrotorSimulator(map_file='./maps/map4.txt')


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
