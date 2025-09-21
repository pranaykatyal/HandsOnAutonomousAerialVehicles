import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        self.start_point = [7.954360487979886, 6.822833826909669, 1.058209137433761]
        self.goal_point = [44.304797815557095, 29.328280798754054, 4.454834705539382]
        self.safety_margin = 0.5  # Safety margin around obstacles



    def parse_map_file(self, filename):
        """
        Parse the map file and extract boundary and blocks
        coords = [xmin, ymin, zmin, xmax, ymax, zmax]
        colors = [r, g, b] each in [0, 1] (make sure color values are in range 0-1)
        self.blocks.append((coords, colors))
        self.boundary = [xmin, ymin, zmin, xmax, ymax, zmax]
        return True if successful, False otherwise (True if file was parsed successfully, without any error.)
        """
        with open(filename, 'r') as file:
            for line in file:
                # print("reading lines")
                sanatized_line = line.strip()
                # print("striped files")
                if '#' in sanatized_line:
                    #comment, do nothing
                    # print("skipping comment")
                    pass
                elif sanatized_line == "":
                    # print("slipping whitepace")
                    pass
                elif 'boundary' in sanatized_line:
                    boundary_string = sanatized_line.replace('boundary', "")
                    self.boundary = list(map(float, boundary_string.split()))
                    # print("set boundary to", self.boundary)
                elif 'block' in sanatized_line:
                    block_string = sanatized_line.replace('block', "")
                    block_item = list(map(float, block_string.split()))
                    color_255 = block_item[5:8]
                    color = [x/255 for x in block_item[6:9]]
                    coord = block_item[0:6]
                    self.blocks.append((coord,color))
                    # print("saved block to list of blocks", self.blocks)
                else:
                    print("Critical error: undefined value in file:", sanatized_line)
                    return False
            return True


    def is_point_in_free_space(self, point):
        """
        Check if a point is in free space (not inside any obstacle)
        Complete implementation with collision checking
        return True if free, False if in collision
        """

        for block in self.blocks:
            block_coords = block[0]
            print("block_coords", block_coords)
            if block_coords[0] <= point[0] <= block_coords[3] and \
               block_coords[1] <= point[1] <= block_coords[4] and \
               block_coords[2] <= point[2] <= block_coords[5] \
               :
                print("occupied point")
                return False
            else:
                print("free point")
                return True


    ##############################################
    #### TODO - Implement line - collision checking #####
    ##############################################
    def is_line_collision_free(self, p1, p2, num_checks=20):
        print("is collisoin free")
        """
        Check if a line segment between two points is collision-free
        Used for RRT* edge validation
        return True if free, False if in collision
        """
        zero = 1e-12
        for block in self.blocks:
            # print("run twice")
            block_coords = block[0]
            t0, t1 = 0.0, 1.0

            for axis in range(0,3):
                block_max = block_coords[2+axis]
                block_min = block_coords[axis]
                delta = p1[axis] - p2[axis]
                start = p1[axis]

                if abs(delta) < zero: #if line is parralel to plane
                    if block_min <= start <= block_max:
                        #line is within block
                        print("parralel line")
                        return False
                ta = (block_min - start) / delta
                tb = (block_max - start) / delta

                #make sure ta <= tb
                if ta > tb:
                    temp = ta
                    ta = tb
                    tb = temp


                if ta > t0:
                    t0 = ta
                if tb < t1:
                    t1 = tb

                if t0 > t1:
                    print("collision")
                    return False
                else:
                    print("axis does not intersect", t0, t1)
        return True    

    
    def generate_random_free_point(self):
        """
        Generate a random point in free space
        Used for RRT* sampling
        """
        if not self.boundary:
            return None
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(xmin + self.safety_margin, xmax - self.safety_margin)
            y = np.random.uniform(ymin + self.safety_margin, ymax - self.safety_margin)
            z = np.random.uniform(zmin + self.safety_margin, zmax - self.safety_margin)
            
            point = [x, y, z]
            if self.is_point_in_free_space(point):
                return point
        
        print("Warning: Could not generate random free point after", max_attempts, "attempts")
        return None

    def get_environment_info(self):
        """Get information about the environment layout"""
        if not self.boundary:
            return "No boundary defined"
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        info = f"""
        Environment Information:
        Boundary: [{xmin}, {ymin}, {zmin}] to [{xmax}, {ymax}, {zmax}]
        Size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} meters
        Volume: {(xmax-xmin)*(ymax-ymin)*(zmax-zmin):.1f} cubic meters
        Obstacles: {len(self.blocks)} blocks
        Safety margin: {self.safety_margin} meters
        """
        
        if self.start_point and self.goal_point:
            distance = np.linalg.norm(np.array(self.goal_point) - np.array(self.start_point))
            info += f"  Start-Goal distance: {distance:.2f} meters\n"
        
        return info
