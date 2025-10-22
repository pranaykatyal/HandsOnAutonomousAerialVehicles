import numpy as np
import torch
import cv2
import json

from pathlib import Path
from dataclasses import fields
from transforms3d.euler import euler2mat

from torch.serialization import add_safe_globals

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.utils import CameraState, get_camera
from nerfstudio.cameras.cameras import CameraType

import math
import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import mat2euler
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(x_rot, y_rot, z_rot, w):
    """Convert quaternion to roll, pitch, yaw (Euler angles)"""
    t0 = +2.0 * (w * x_rot + y_rot * z_rot)
    t1 = +1.0 - 2.0 * (x_rot * x_rot + y_rot * y_rot)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y_rot - z_rot * x_rot)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z_rot + x_rot * y_rot)
    t4 = +1.0 - 2.0 * (y_rot * y_rot + z_rot * z_rot)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler Angles to Quaternion
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def quaternion_multiply(quaternion1, quaternion0):
    """
    Multiplies 2 quaternions
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    make rotation matrix from rol, pitch, yaw
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    return R



class SplatRenderer:
    def __init__(self, config_path: str, json_path: str, aspect_ratio: float = 4/3):
# splat_render.py


        add_safe_globals([np.core.multiarray.scalar])  # allowlist NumPy scalar from old checkpoints

        config, pipeline, _, _ = eval_setup(Path(config_path), eval_num_rays_per_chunk=None, test_mode="test")

        config, pipeline, _, _ = eval_setup(Path(config_path), eval_num_rays_per_chunk=None, test_mode="test")
        self.config = config
        self.pipeline = pipeline
        self.model = pipeline.model
        self.model.eval()
        self.device = self.model.device
        self.aspect_ratio = aspect_ratio

        # Set background
        self.background_color = torch.tensor([0.1490, 0.1647, 0.2157], device=self.device)
        self.model.set_background(self.background_color)

        # Load camera settings
        with open(json_path, 'r') as f:
            camera_data = json.load(f)
        cam_matrix = np.array(camera_data["camera"]["c2w_matrix"])
        self.init_position = cam_matrix[:, 3]
        self.init_orientation = cam_matrix[:, :3]

        fov = camera_data["camera"].get("fov_radians", 1.3089969389957472)
        resolution = camera_data["camera"].get("render_resolution", 1080)
        self.fov = fov
        self.image_height = resolution
        self.image_width = int(resolution * aspect_ratio)

        self.colormap_options_rgb = ColormapOptions(colormap='default', normalize=True)
        self.colormap_options_depth = ColormapOptions(colormap='gray', normalize=True)


    def getNWUPose(self,position, orientation_rpy):
        pos_update = self.init_orientation @ np.array([
            [position[1]],  # East
            [-position[2]], # Up
            [-position[0]]  # Forward
        ])
        pos = self.init_position + pos_update.flatten()

        # Orientation (NED to GSplat)
        R_ned = euler2mat(orientation_rpy[0], orientation_rpy[1], orientation_rpy[2])
        Rot = self.init_orientation @ euler2mat(
            orientation_rpy[1], -orientation_rpy[2], -orientation_rpy[0]
        )

        rotation = R.from_matrix(Rot)
        quat = rotation.as_quat()  # x, y, z, w
        quat_wxyz = np.roll(quat, 1) # w, x, y, z
        return pos, quat_wxyz  
    
    def getNEDPose(self, nwu_position, R_nwu):
        # Position Conversion (from NWU to NED)
        pos_update = self.init_orientation @ np.array([
            [nwu_position[1]],  # West -> East
            [nwu_position[2]],  # Up -> Down
            [-nwu_position[0]]  # North -> North (no change)
        ])
        pos = self.init_position + pos_update.flatten()

        # Orientation Conversion (from NWU to NED)
        # R_nwu = rotation_matrix_from_euler(nwu_orientation_rpy[0], nwu_orientation_rpy[1], nwu_orientation_rpy[2])
        Rot = self.init_orientation @ R_nwu
        
        # Get quaternion from the rotation matrix
        rotation = R.from_matrix(Rot)
        quat = rotation.as_quat()  # x, y, z, w
        quat_wxyz = np.roll(quat, 1)  # w, x, y, z
        
        # Convert the quaternion back to Euler angles (roll, pitch, yaw)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler_angles  # In degrees
        
        return pos, quat_wxyz, roll, pitch, yaw
    
    def getNEDPose(self, splatRT):
        Rsplat = splatRT[:3, :3]
        tsplat = splatRT[:3, 3]

        RsplatInv = Rsplat.T
        euf = self.init_orientation.T @ (tsplat - self.init_position)
        x = -euf[2]
        y = euf[0]
        z = -euf[1]
        print('NED Position ', x, y, z)

        # now get the orientation
        R_ned = self.init_orientation.T @ Rsplat
        pitch, yaw, roll = mat2euler(R_ned)
        print('NED Orientation (rpy) ', np.degrees(-roll), np.degrees(pitch), np.degrees(-yaw))


        return [x,y,z, np.degrees(-roll), np.degrees(pitch), np.degrees(-yaw)]



    def render(self, position: np.ndarray, orientation_rpy: np.ndarray):
        """
        position: (3,) [x, y, z] in meters (in NED frame)
        orientation_rpy: (3,) [roll, pitch, yaw] in radians (in NED frame)
        Returns: RGB and Depth images (uint8)
        """
        # Position transform from NED → GSplat (NWU)
        pos_update = self.init_orientation @ np.array([
            [position[1]],  # East
            [-position[2]], # Up
            [-position[0]]  # Forward
        ])
        pos_cam = self.init_position + pos_update.flatten()

        # Orientation (NED to GSplat)
        # R_ned = euler2mat(orientation_rpy[0], orientation_rpy[1], orientation_rpy[2])
        R_cam = self.init_orientation @ euler2mat(
            orientation_rpy[1], -orientation_rpy[2], -orientation_rpy[0]
        )
        # print('R NED ', R_ned)
        # print("R Camera ", R_cam)
        c2w = torch.tensor(np.column_stack([R_cam, pos_cam]), dtype=torch.float32, device=self.device)
        # print('Camera to World Transform ', c2w)
        camera_state = CameraState(
            fov=self.fov,
            aspect=self.aspect_ratio,
            c2w=c2w,
            camera_type=CameraType.PERSPECTIVE
        )
        # print('Camera State ', camera_state)

        camera = get_camera(camera_state, self.image_height, self.image_width).to(self.device)
        outputs = self.model.get_outputs_for_camera(camera)

        # print('outputs ', outputs["rgb"].dtype)

        rgb = apply_colormap(outputs["rgb"], self.colormap_options_rgb)
        rgb = (rgb * 255).type(torch.uint8).cpu().numpy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = apply_colormap(outputs["depth"], self.colormap_options_depth)
        depth = (depth * 255).type(torch.uint8).cpu().numpy()

        return rgb, depth, outputs["depth"].cpu().numpy()
