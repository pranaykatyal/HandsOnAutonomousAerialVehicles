import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        self.start_point = [6.0, 20.0, 6.0]
        self.goal_point = [0.0, -5.0, 1.0]
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
        
        Current issue: Your implementation has a logic error - it returns True 
        immediately when checking ANY block, instead of checking ALL blocks
        """
        x, y, z = point
        
        # Check if point is within boundary
        if not self.boundary:
            return False
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
            return False
        
        # Check if point is inside any obstacle (with safety margin)
        for block_coords, _ in self.blocks:
            bxmin, bymin, bzmin, bxmax, bymax, bzmax = block_coords
            
            # Add safety margin
            bxmin -= self.safety_margin
            bymin -= self.safety_margin  
            bzmin -= self.safety_margin
            bxmax += self.safety_margin
            bymax += self.safety_margin
            bzmax += self.safety_margin
            
            # If point is inside this block, it's NOT free
            if (bxmin <= x <= bxmax and 
                bymin <= y <= bymax and 
                bzmin <= z <= bzmax):
                return False
        
        # Point is free if it's not inside any obstacle
        return True


    def is_line_collision_free(self, p1, p2, eps=1e-12):
        Ax, Ay, Az = p1
        Bx, By, Bz = p2
        dx, dy, dz = Bx - Ax, By - Ay, Bz - Az

        for block in self.blocks:
            coords = block[0]
            mn = (coords[0], coords[1], coords[2])
            mx = (coords[3], coords[4], coords[5])

            t0, t1 = 0.0, 1.0  # clamp to SEGMENT

            for a, d, lo, hi in (
                (Ax, dx, mn[0], mx[0]),
                (Ay, dy, mn[1], mx[1]),
                (Az, dz, mn[2], mx[2]),
            ):
                if abs(d) < eps:
                    # Parallel to this axis: must already be inside its slab
                    if a < lo or a > hi:
                        # No intersection with THIS block; move to next block
                        t0, t1 = 1.0, 0.0  # mark empty to skip post-check
                        break
                    # inside slab → no constraint from this axis
                    continue

                tA = (lo - a) / d
                tB = (hi - a) / d
                if tA > tB:
                    tA, tB = tB, tA

                if tA > t0: t0 = tA
                if tB < t1: t1 = tB
                if t0 > t1:
                    # No intersection with THIS block; try next block
                    break

            # If after all axes the window is non-empty, the segment hits this block
            if t0 <= t1:
                return False  # in collision with this block

        return True  # free of all blocks

    
    
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
