from environment import Environment3D


if __name__ == "__main__":
    env = Environment3D()
    env.parse_map_file(filename='./maps/map4.txt')
    env.is_point_in_free_space([2,2,2])
    env.is_line_collision_free([0,0,0],[.5,0.5,0])