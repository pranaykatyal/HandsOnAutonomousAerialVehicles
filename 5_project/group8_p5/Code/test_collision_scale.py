"""
Test collision checker to find the actual scale of the environment
"""

from collisionChecker import doesItCollide as _doesItCollide_original
import numpy as np

print("="*60)
print("COLLISION CHECKER SCALE CALIBRATION")
print("="*60)

# Test origin
print("\n1. Testing origin [0, 0, 0]:")
result = _doesItCollide_original([0, 0, 0], drone_radius=0.001, collision_threshold=0.01)
print(f"   Collision: {result}")

# Test known good position from your P4
print("\n2. Testing P4 good position [0.2, -0.2, 0.0]:")
result = _doesItCollide_original([0.2, -0.2, 0.0], drone_radius=0.001, collision_threshold=0.01)
print(f"   Collision: {result}")

# Test with different drone radii to find what makes sense
print("\n3. Testing different drone_radius values at origin:")
for radius in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
    result = _doesItCollide_original([0, 0, 0], drone_radius=radius, collision_threshold=0.01)
    print(f"   drone_radius={radius:.3f}: Collision={result}")

# Test nearby positions to understand scale
print("\n4. Testing positions near origin to find scale:")
test_positions = [
    [0.01, 0, 0],   # 1cm forward (if scale is 1:1)
    [0.05, 0, 0],   # 5cm forward
    [0.10, 0, 0],   # 10cm forward
    [0.20, 0, 0],   # 20cm forward
    [0.50, 0, 0],   # 50cm forward
    [1.00, 0, 0],   # 1m forward
]

for pos in test_positions:
    result = _doesItCollide_original(pos, drone_radius=0.001, collision_threshold=0.01)
    print(f"   Position {pos}: Collision={result}")

# Test the trajectory that failed
print("\n5. Testing trajectory that collided:")
failed_pos = [0.48, 0.27, 0.02]
result = _doesItCollide_original(failed_pos, drone_radius=0.001, collision_threshold=0.01)
print(f"   Position {failed_pos}: Collision={result}")

# Test with smaller drone radius
result_small = _doesItCollide_original(failed_pos, drone_radius=0.0001, collision_threshold=0.01)
print(f"   Same position, drone_radius=0.0001: Collision={result_small}")

# Grid search around a known collision point
print("\n6. Grid search around collision point [0.48, 0.27, 0.02]:")
print("   Format: X position vs Y position (Z=0.02)")
print("   '.' = clear, 'X' = collision")
print()
print("   Y position:")
print("   ", end="")
for y in np.arange(0.1, 0.4, 0.05):
    print(f"{y:5.2f} ", end="")
print()

for x in np.arange(0.3, 0.6, 0.05):
    print(f"X={x:.2f}: ", end="")
    for y in np.arange(0.1, 0.4, 0.05):
        result = _doesItCollide_original([x, y, 0.02], drone_radius=0.001, collision_threshold=0.01)
        print("  X   " if result else "  .   ", end="")
    print()

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print()
print("Based on the results above:")
print("1. If positions < 0.1 are clear but > 0.5 collide:")
print("   → Scale is roughly 1:1 (1 unit = 1 meter)")
print()
print("2. If everything collides even at 0.01:")
print("   → Environment is VERY dense OR scale is much smaller")
print()
print("3. If collision with drone_radius=0.001 but not 0.0001:")
print("   → Default 1mm radius is actually hitting obstacles")
print("   → Real drone radius should be much smaller (0.0001 or less)")
print()
print("4. Recommended drone_radius:")
print("   - If origin is clear: Use default 0.001")
print("   - If origin collides with 0.001: Try 0.0001 or 0.00001")
print("   - The goal is: don't collide at known good positions")
print()