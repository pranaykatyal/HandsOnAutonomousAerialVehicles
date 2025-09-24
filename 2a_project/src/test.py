from environment import Environment3D


if __name__ == "__main__":
    env = Environment3D()
    env.parse_map_file(filename='./maps/map4.txt')
    env.is_point_in_free_space([12,12,12])
    print(f" line is collision free: {env.is_line_collision_free([0,0,0],[-6,-6,-6])}")