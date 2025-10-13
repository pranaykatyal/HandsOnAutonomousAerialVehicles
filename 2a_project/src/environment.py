import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tello
class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        
        
        # Switch-case for start/goal points based on input i
        # Usage: Environment3D(i=1) for map1, i=2 for map2, etc.
        i = 3 # Default to map2 if not set externally
        if i == 1:
            self.start_point = [6.0, 20.0, 6.0]  # map1.txt
            self.goal_point = [0.0, -5.0, 1.0]
        elif i == 2:
            self.start_point = [0.0, 20.0, 2.0]  # map2.txt
            self.goal_point = [10.0, 20.0, 3.0]
        elif i == 3:
            self.start_point = [0.0, 3.0, 2.0]  # map3.txt
            self.goal_point = [20.0, 2.0, 4.0]
        elif i == 4:
            self.start_point = [7.954360487979886, 6.822833826909669, 1.058209137433761]  # map4.txt
            self.goal_point = [44.304797815557095, 29.328280798754054, 4.454834705539382]
        else:
            # Default/fallback
            self.start_point = [0.0, 20.0, 2.0]
            self.goal_point = [10.0, 20.0, 3.0]
        
        
        
        
        self.robotmarginxy = tello.margin_xy  # Robot margin for obstacle bloating
        print("robot margin xy:", self.robotmarginxy)
        
        self.robotmarginz = tello.margin_z    # Robot margin for obstacle bloating
        print("robot margin z:", self.robotmarginz)
        
        self.safety_margin = 0.25  # Safety margin around obstacles

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
        Check if a point is in free space (not inside any obstacle AND within boundaries)
        return True if free, False if in collision or out of bounds
        """
        # FIRST: Check if point is within boundaries
        if self.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
            if not (xmin <= point[0] <= xmax and 
                    ymin <= point[1] <= ymax and 
                    zmin <= point[2] <= zmax):
                # print(f"Point {point} is outside boundary")
                return False
        
        # SECOND: Check if point collides with any obstacle
        for block in self.blocks:
            block_coords = block[0]
            if ((block_coords[0] - self.robotmarginxy) <= point[0] <= (block_coords[3] + self.robotmarginxy) and
                (block_coords[1] - self.robotmarginxy) <= point[1] <= (block_coords[4] + self.robotmarginxy) and
                (block_coords[2] - self.robotmarginz) <= point[2] <= (block_coords[5] + self.robotmarginz)):
                # print("occupied point")
                return False
        
        return True 


    def is_line_collision_free(self, p1, p2, eps=1e-12):
        """Check if line segment is collision-free AND stays within boundaries"""
        
        # FIRST: Check if both endpoints are within boundaries
        if self.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
            
            for point in [p1, p2]:
                if not (xmin <= point[0] <= xmax and 
                        ymin <= point[1] <= ymax and 
                        zmin <= point[2] <= zmax):
                    return False  # Endpoint outside boundary
        
        # SECOND: Check collision with obstacles (your existing code)
        Ax, Ay, Az = p1
        Bx, By, Bz = p2
        dx, dy, dz = Bx - Ax, By - Ay, Bz - Az

        for block in self.blocks:
            coords = block[0]
            mn = (coords[0] - self.robotmarginxy, coords[1] - self.robotmarginxy, coords[2] - self.robotmarginz)
            mx = (coords[3] + self.robotmarginxy, coords[4] + self.robotmarginxy, coords[5] + self.robotmarginz)

            t0, t1 = 0.0, 1.0
            intersects = True
            
            for a, d, lo, hi in (
                (Ax, dx, mn[0], mx[0]),
                (Ay, dy, mn[1], mx[1]),
                (Az, dz, mn[2], mx[2]),
            ):
                if abs(d) < eps:
                    if a < lo or a > hi:
                        intersects = False
                        break
                    continue

                tA = (lo - a) / d
                tB = (hi - a) / d
                if tA > tB:
                    tA, tB = tB, tA

                if tA > t0: t0 = tA
                if tB < t1: t1 = tB
                if t0 > t1:
                    intersects = False
                    break

            if intersects and t0 <= t1:
                return False

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
            x = np.random.uniform(xmin + (self.safety_margin*0.5), xmax - (self.safety_margin*0.5))
            y = np.random.uniform(ymin + (self.safety_margin*0.5), ymax - (self.safety_margin*0.5))
            z = np.random.uniform(zmin + (self.safety_margin*0.5), zmax - (self.safety_margin*0.5))
            
            point = np.array([x, y, z])
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
