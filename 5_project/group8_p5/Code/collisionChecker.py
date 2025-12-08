#!/usr/bin/env python3

import numpy as np
import open3d as o3d


class CollisionChecker:
    def __init__(
        self,
        ply_path: str = "../data/occupancy_grid/splat.ply",
        collision_threshold: float = 0.01,
        drone_radius: float = 0.01,
        num_points: int = 10,
    ):
        """
        Simple collision checker around a spherical drone.

        Args:
            ply_path: Path to the occupancy grid PLY file.
            collision_threshold: Distance (m) below which we consider it a collision.
            drone_radius: Radius (m) of the drone sphere.
            num_points: Resolution for sampling the sphere surface.
        """
        self.occupancy_grid = o3d.io.read_point_cloud(ply_path)

        if not self.occupancy_grid.has_points():
            raise RuntimeError("Occupancy PLY file is empty or not loaded correctly")

        self.collision_threshold = collision_threshold
        self.drone_radius = drone_radius
        self.num_points = num_points

        # Precompute unit sphere samples around origin
        phi = np.linspace(0, np.pi, self.num_points)
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        phi, theta = np.meshgrid(phi, theta)

        x = self.drone_radius * np.sin(phi) * np.cos(theta)
        y = self.drone_radius * np.sin(phi) * np.sin(theta)
        z = self.drone_radius * np.cos(phi)

        self.sphere_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    def check_collision(self, position):
        """
        Check if a given 3D position is in collision.

        Args:
            position: (x, y, z) as list/tuple/np.ndarray

        Returns:
            bool: True if collision, False otherwise.
        """
        position = np.asarray(position, dtype=float).reshape(1, 3)

        # Translate precomputed sphere around current position
        sphere_points = self.sphere_points + position

        pcd_robot = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(sphere_points)
        )
        distances = np.array(
            self.occupancy_grid.compute_point_cloud_distance(pcd_robot)
        )

        if distances.size == 0:
            # No obstacles in occupancy grid to compare to
            return False

        return np.any(distances < self.collision_threshold)


# Convenience function if you don't want to deal with the class externally
_default_checker = None


def doesItCollide(
    pos,
    ply_path: str = "../data/occupancy_grid/splat.ply",
    collision_threshold: float = 0.01,
    drone_radius: float = 0.001,
    num_points: int = 10,
) -> bool:
    """
    Convenience function: given a pose (x, y, z), return True/False for collision.

    Lazily creates a global CollisionChecker the first time it's called
    (so the PLY is only loaded once if you call this repeatedly).

    Args:
        x, y, z: Drone center position.
        ply_path, collision_threshold, drone_radius, num_points:
            Same as CollisionChecker.__init__.

    Returns:
        bool: True if collision, False otherwise.
    """
    global _default_checker

    if _default_checker is None:
        _default_checker = CollisionChecker(
            ply_path=ply_path,
            collision_threshold=collision_threshold,
            drone_radius=drone_radius,
            num_points=num_points,
        )

    return _default_checker.check_collision([pos[0], pos[1], pos[2]])


if __name__ == "__main__":
    # Example usage
    checker = CollisionChecker()

    test_pos = [0.0, 0.0, 0.0]
    collided = checker.check_collision(test_pos)
    print(f"Position {test_pos} collision: {collided}")

    # Or using the convenience function:
    collided2 = doesItCollide(0.0, 0.0, 0.0)
    print(f"Position (0,0,0) collision (via will_collide): {collided2}")
