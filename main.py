from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface

import trimesh
import skimage
import pyrender
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mesh', type=str, default='models/chair.obj')

args = parser.parse_args()

# load mesh
mesh = trimesh.load(args.mesh)

# convert mesh to voxels
voxels = mesh_to_voxels(mesh, 64, pad=False)
vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# view voxel mesh
voxel_mesh.show()
print("voxel mesh viewed")


# add gaussian noise to mesh
mesh.vertices += np.random.normal(0, 0.001, size=mesh.vertices.shape)
# view noisy mesh
mesh.show()
print("noisy mesh viewed")


# run sdf
points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
colors = np.zeros(points.shape)

# green for inside, red for outside
colors[sdf < 0, 1] = 5
colors[sdf > 0, 0] = 1

cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
print("sdf viewed")

