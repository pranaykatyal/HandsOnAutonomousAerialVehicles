"""
Diagnostic script for Blender scene setup
Run this to check camera and object orientations
"""

import bpy
import mathutils

print("=" * 70)
print("Blender Scene Diagnostics")
print("=" * 70)

# Check all objects
print("\nObjects in scene:")
for obj in bpy.data.objects:
    print(f"  {obj.name} ({obj.type})")
    if obj.type == 'MESH':
        print(f"    Location: {obj.location}")
        print(f"    Rotation: {obj.rotation_euler}")
        print(f"    Scale: {obj.scale}")
    elif obj.type == 'CAMERA':
        print(f"    Location: {obj.location}")
        print(f"    Rotation: {obj.rotation_euler}")

# Check Window object specifically
window = bpy.data.objects.get('Window')
if window:
    print(f"\nWindow object details:")
    print(f"  Location: {window.location}")
    print(f"  Rotation (degrees): ({window.rotation_euler.x * 57.3:.1f}, {window.rotation_euler.y * 57.3:.1f}, {window.rotation_euler.z * 57.3:.1f})")
    
    # Check if normals are flipped
    mesh = window.data
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.polygons)}")
    
    # Check face normal direction
    if len(mesh.polygons) > 0:
        face = mesh.polygons[0]
        normal = face.normal
        print(f"  First face normal: ({normal.x:.3f}, {normal.y:.3f}, {normal.z:.3f})")
        print(f"    → Normal points in {'front (+Z)' if normal.z > 0 else 'back (-Z)'} direction")

# Check camera
camera = bpy.data.objects.get('Camera')
if camera:
    print(f"\nCamera details:")
    print(f"  Location: {camera.location}")
    print(f"  Rotation (degrees): ({camera.rotation_euler.x * 57.3:.1f}, {camera.rotation_euler.y * 57.3:.1f}, {camera.rotation_euler.z * 57.3:.1f})")
    
    # Check what direction camera is pointing
    # Camera looks down -Z in its local space
    camera_forward = camera.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
    print(f"  Camera forward direction: ({camera_forward.x:.3f}, {camera_forward.y:.3f}, {camera_forward.z:.3f})")

print("\n" + "=" * 70)
print("Recommendations:")
print("=" * 70)

# Give recommendations
if window:
    face_normal_z = mesh.polygons[0].normal.z if len(mesh.polygons) > 0 else 0
    
    if face_normal_z < 0:
        print("⚠ Window normals face BACKWARD (-Z)")
        print("  Fix: Select Window → Edit Mode → Mesh → Normals → Flip")
        print("  Or in Python: bpy.ops.mesh.flip_normals()")
    else:
        print("✓ Window normals face FORWARD (+Z)")
    
    if window.location.z != 0:
        print(f"⚠ Window is at Z={window.location.z:.3f} (not at origin)")
        print(f"  Camera positions should account for this offset")

if camera:
    if camera.location.z < 0:
        print(f"⚠ Camera is at Z={camera.location.z:.3f} (in front of window)")
        print("  Camera should be at positive Z to look at window")
    else:
        print(f"✓ Camera is at Z={camera.location.z:.3f} (behind window)")

print("=" * 70)