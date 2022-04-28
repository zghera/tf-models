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

def visualize_mesh_pyvista(verts, faces, verts_mask, faces_mask):
  v = verts.numpy()
  f = faces.numpy()
  vm = verts_mask.numpy() == 1
  fm = faces_mask.numpy() == 1
  new_f = f[fm]
  ## Appends a 3 to each value in the face numpy array. This indicates the number of points for each face to Pyvista.
  faces_with_dimensions = np.insert(new_f, 0, 3, axis=1)
  # Creates a Pyvista Polydata object using the vertices and faces
  surf = pv.PolyData(v, faces=faces_with_dimensions)
  # Smoothens any rough edges of the object
  smooth = surf.smooth(200)
  # Plots the object
  smooth.plot()


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

