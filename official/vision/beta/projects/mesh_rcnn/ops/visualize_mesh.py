"""Mesh Visualization"""

import matplotlib.pyplot as plt
import tensorflow as tf
import open3d as o3d
import pyvista as pv
import numpy as np
from mpl_toolkits.mplot3d import art3d

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify


def create_voxels(grid_dims, batch_size, occupancy_locs):
  ones = tf.ones(shape=[len(occupancy_locs)])
  voxels = tf.scatter_nd(
      indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
      updates=ones,
      shape=[batch_size, grid_dims, grid_dims, grid_dims])

  return voxels

def visualize_mesh(verts, faces, verts_mask, faces_mask):
  v = verts.numpy()
  f = faces.numpy()
  vm = verts_mask.numpy() == 1
  fm = faces_mask.numpy() == 1
  new_f = f[fm]
  out = np.insert(new_f, 0, 3, axis=1)
  ##meshz = o3d.geometry.TriangleMesh()
  ##meshz.vertices = o3d.utility.Vector3dVector(v)
  ##meshz.triangles = o3d.utility.Vector3iVector(new_f)
  ##meshz.compute_vertex_normals()
  ##o3d.visualization.draw_geometries([meshz])
  surf = pv.PolyData(v, faces=out)
  smooth = surf.smooth(200)
  smooth.plot()

  ##new_f = f[fm]

  ##fig = plt.figure()
  ##ax = fig.add_subplot(projection="3d")

  ##pc = art3d.Poly3DCollection(
      ##v[new_f], facecolors=(1, 0.5, 1, 1), edgecolor="black")

  ##ax.add_collection(pc)

  ##plt.axis('off')

if __name__ == '__main__':
  _grid_dims = 2
  _batch_size = 5
  _occupancy_locs = [
      [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],

      [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],

      [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
      [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
  ]
  voxels = create_voxels(_grid_dims, _batch_size, _occupancy_locs)

  mesh = cubify(voxels, 0.5)
  _verts = mesh['verts']
  _faces = mesh['faces']
  _verts_mask = mesh['verts_mask']
  _faces_mask = mesh['faces_mask']

  batch_to_view = 0
  visualize_mesh(_verts[batch_to_view, :],
                 _faces[batch_to_view, :],
                 _verts_mask[batch_to_view, :],
                 _faces_mask[batch_to_view, :])
  ##plt.show()
