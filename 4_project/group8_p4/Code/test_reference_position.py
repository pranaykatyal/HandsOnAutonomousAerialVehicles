#!/usr/bin/env python3
"""
Start from the REFERENCE camera position from render_settings.json
This is where the Gaussian splat was trained - should show scene correctly
"""

import sys
sys.path.insert(0, '/mnt/project')

import numpy as np
import cv2
import json
from splat_render import SplatRenderer

print("="*70)
print("RENDERING FROM REFERENCE CAMERA POSITION")
print("="*70)

# Load renderer
config_path = "../data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml"
json_path = "../render_settings/render_settings.json"

renderer = SplatRenderer(config_path, json_path)

# Load reference camera matrix
with open(json_path, 'r') as f:
    settings = json.load(f)

c2w = np.array(settings['camera']['c2w_matrix'])
print("\nReference c2w matrix:")
print(c2w)
print(f"\nReference position in splat world: {c2w[:3, 3]}")

# The reference camera should show the scene correctly
# Start by checking what NED=[0,0,0] renders
print("\n" + "="*70)
print("SOLUTION: Use NED [0, 0, 0] as starting point")
print("="*70)
print("\nThe splat_render.py applies transformations to map NED to splat world.")
print("NED [0, 0, 0] should map close to the reference camera position.")

# Test NED origin and nearby positions
rpy = np.radians([0.0, 0.0, 0.0])

test_positions = [
    ("reference", np.array([0.0, 0.0, 0.0]), "NED Origin (should match reference)"),
    ("back_1m", np.array([-1.0, 0.0, 0.0]), "1m back"),
    ("back_2m", np.array([-2.0, 0.0, 0.0]), "2m back"),
    ("back_5m", np.array([-5.0, 0.0, 0.0]), "5m back"),
    ("forward_1m", np.array([1.0, 0.0, 0.0]), "1m forward"),
    ("forward_2m", np.array([2.0, 0.0, 0.0]), "2m forward"),
    ("forward_5m", np.array([5.0, 0.0, 0.0]), "5m forward"),
    ("left_2m", np.array([0.0, -2.0, 0.0]), "2m left"),
    ("right_2m", np.array([0.0, 2.0, 0.0]), "2m right"),
    ("up_2m", np.array([0.0, 0.0, -2.0]), "2m up"),
]

print("\nRendering from positions around origin:")
for name, pos, desc in test_positions:
    print(f"\n  {desc}")
    print(f"    NED: {pos}")
    
    rgb, _, _ = renderer.render(pos, rpy)
    
    # Add label
    img = rgb.copy()
    text = f"{desc} | NED: {pos}"
    cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
               1.5, (0, 255, 0), 3)
    
    filename = f"ref_{name}.png"
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"    Saved: {filename}")

print("\n" + "="*70)
print("CHECK: ref_*.png files")
print("="*70)
print("\nThe ref_reference.png should show a good view of the scene.")
print("If window is visible there, use [0, 0, 0] as starting position.")
print("If window is slightly off, use one of the nearby positions.")
print("="*70)