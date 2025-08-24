import bpy
import bmesh
import numpy as np
import json
import math

    
def add_sphere(name, location, radius=0.015, material=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    sphere = bpy.context.object
    sphere.name = name
    sphere.data.name = name
    my_coll.objects.link(sphere)

    # Assign material
    if material:
        if len(sphere.data.materials) == 0:
            sphere.data.materials.append(material)
        else:
            sphere.data.materials[0] = material

def create_material(name, color):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True  # Enable nodes
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*color, 1)  # RGBA
    return mat

# Predefine materials for reuse
green_material = create_material("Start_Green", (0, 1, 0))  # Green
red_material = create_material("Goal_Red", (1, 0, 0))  # Red
blue_material = create_material("Path_Blue", (0, 0, 1))  # Blue

# exp_name = 'statues'
exp_name = 'stonehenge'

base = bpy.path.abspath('//') + f'result/{exp_name}/path.json'
# base = bpy.path.abspath('//') + f'catnips_data/{exp_name}/path.json'

my_coll = bpy.data.collections.new(f'Statistical_test')
bpy.context.scene.collection.children.link(my_coll)

with open(base, 'r') as f:
    meta = json.load(f)
    
locations = meta["traj"]


for iter, points in enumerate(locations):

    traj = np.array(points)[:, :3]

    # Create curve for the path
    crv = bpy.data.curves.new(f'crv_{iter}', 'CURVE')
    crv.dimensions = '3D'
    spline = crv.splines.new(type='POLY')
    spline.points.add(len(traj) - 1)

    # assign the point coordinates to the spline points
    for p, new_co in zip(spline.points, traj[:, :3]):
        p.co = (new_co.tolist() + [1.0])  # (add nurbs weight)

    # make a new object with the curve
    obj = bpy.data.objects.new(f'Traj_{iter}', crv)
    obj.data.bevel_depth = 0.005
    my_coll.objects.link(obj)

    # Assign blue material to the path
    if len(obj.data.materials) == 0:
        obj.data.materials.append(blue_material)
    else:
        obj.data.materials[0] = blue_material

    # Add start and goal spheres
    start_loc = traj[0]  # First point in trajectory
    goal_loc = traj[-1]  # Last point in trajectory

    add_sphere(f"Start_{iter}", start_loc, material=green_material)
    add_sphere(f"Goal_{iter}", goal_loc, material=red_material)
    