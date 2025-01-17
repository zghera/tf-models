# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test for mesh ops."""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import (
    MeshSampler, compute_edges, vert_align, get_verts_from_indices)
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import create_voxels


class ComputeEdgesTest(parameterized.TestCase, tf.test.TestCase):

  def test_compute_edges(self):
    faces = tf.random.uniform(shape=[2, 1000, 3], maxval=1000, dtype=tf.int32)
    faces_mask = tf.random.uniform(shape=[2, 1000], maxval=1, dtype=tf.int32)

    edges, edges_mask = compute_edges(faces, faces_mask)

    self.assertAllEqual(tf.shape(edges)[1], tf.shape(faces)[1] * 3)
    self.assertAllEqual(tf.shape(edges)[1], tf.shape(edges_mask)[1])

    edges = edges.numpy()
    edges_mask = edges_mask.numpy()

    for edge, mask in zip(edges, edges_mask):
      valid_edges = edge[np.array(mask, dtype=bool), :]
      unique_edges = np.unique(valid_edges, axis=1)
      self.assertEqual(valid_edges.shape[0], unique_edges.shape[0])

class VertAlignTest(parameterized.TestCase, tf.test.TestCase):

  def test_vert_align(self):
    feature_map = tf.random.uniform(shape=[2, 4, 4, 3], minval=-1, maxval=1)
    verts = tf.convert_to_tensor([
        [
            [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]
        ],
        [
            [1.1, 1.2, 1.3], [-1.1, -1.2, -1.3]
        ]
    ])

    features = vert_align(feature_map, verts, True, 'border')
    self.assertAllEqual(tf.shape(features)[:-1], tf.shape(verts)[:-1])
    self.assertAllEqual(tf.shape(features)[-1], tf.shape(feature_map)[-1])


@parameterized.named_parameters(
    {'testcase_name': 'unit_mesh',
     'grid_dims': 1,
     'batch_size': 1,
     'occupancy_locs':
          [
              [0, 0, 0, 0]
          ],
    },
    {'testcase_name': 'batched_unit_mesh',
     'grid_dims': 1,
     'batch_size': 2,
     'occupancy_locs':
          [
              [0, 0, 0, 0],
              [1, 0, 0, 0]
          ],
    },
    {'testcase_name': 'batched_small_mesh',
     'grid_dims': 2,
     'batch_size': 2,
     'occupancy_locs':
          [
              [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
              [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
          ],
    },
    {'testcase_name': 'batched_large_mesh_with_empty_samples',
     'grid_dims': 4,
     'batch_size': 5,
     'occupancy_locs':
          [
              [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
              [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
              [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
              [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
          ],
    }
)
class MeshSamplerTest(parameterized.TestCase, tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._num_samples = 100
    self._eps = 1e-6
    self._thresh = 0.5

  def _get_verts_and_faces(
      self, grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert occupancy grid to tensor representing cubified mesh."""
    # pylint: disable=missing-param-doc
    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    mesh = cubify(voxels, self._thresh)

    verts = tf.cast(mesh['verts'], tf.float32)
    faces = tf.cast(mesh['faces'], tf.int32)
    verts_mask = tf.cast(mesh['verts_mask'], tf.int8)
    faces_mask = tf.cast(mesh['faces_mask'], tf.int8)

    return verts, faces, verts_mask, faces_mask

  def test_sampled_points(self,
                          grid_dims: int,
                          batch_size: int,
                          occupancy_locs: List[List[int]]) -> None:
    """Verify that sampled points lie on the mesh faces."""
    # pylint: disable=missing-param-doc
    verts, faces, verts_mask, faces_mask = self._get_verts_and_faces(
        grid_dims, batch_size, occupancy_locs
    )


    sampler = MeshSampler(self._num_samples)
    samples, _, sampled_verts_ind = sampler.sample_meshes(
        verts, verts_mask, faces, faces_mask,
    )


    ## Verify shapes.
    self.assertAllEqual(tf.shape(samples), [batch_size, self._num_samples, 3])
    self.assertAllEqual(tf.shape(sampled_verts_ind),
                        [batch_size, self._num_samples, 2])

    ## Verify sampled points fall within bounds for vertices.
    self.assertAllLessEqual(samples, tf.math.reduce_max(verts) + self._eps)
    self.assertAllGreaterEqual(samples, -1)

    ## Verify sampled points are on valid faces.
    # BoolTensor[B, Ns]: Tensor that contains true if the sampled face is in a
    # mesh is valid and false otherwise (masked off).
    sampled_faces_valid = tf.cast(
        tf.gather_nd(faces_mask, sampled_verts_ind), tf.bool
    )
    mesh_is_valid = tf.repeat(
        tf.reduce_sum(faces_mask, axis=-1, keepdims=True) != 0,
        self._num_samples,
        axis=-1
    )
    self.assertAllEqual(sampled_faces_valid, mesh_is_valid)

    # Compute vectors on face of the mesh for asserts below.
    v0, v1, v2 = get_verts_from_indices(
        verts, verts_mask, faces, faces_mask, num_inds_per_set=3
    )
    va = tf.gather_nd(v0, sampled_verts_ind)
    vb = tf.gather_nd(v1, sampled_verts_ind)
    vc = tf.gather_nd(v2, sampled_verts_ind)
    vp = samples
    ap = va - vp
    bp = vb - vp
    cp = vc - vp
    ab = va - vb
    ac = va - vc
    bc = vb - vc
    ca = vc - va

    ## Verify sampled points fall on same plane as face.
    # Note: Use cross product of vectors from face vertices rather than
    # `normals` returned from `sample_meshes` to keep seperation between the
    # this test and the normals test `test_sampled_point_normals`.
    ab_x_ac = tf.linalg.cross(ab, ac)
    ap_dot_face_norm = tf.math.reduce_sum(ap * ab_x_ac, axis=-1)
    self.assertAllClose(ap_dot_face_norm,
                        tf.zeros((batch_size, self._num_samples)))

    ## Verify sampled points fall within triangular face.
    ab_x_ap = tf.linalg.cross(ab, ap)
    bc_x_bp = tf.linalg.cross(bc, bp)
    ca_x_cp = tf.linalg.cross(ca, cp)

    # Confirm each cross product has the same direction.
    ax_dot_bx = tf.math.reduce_sum(ab_x_ap * bc_x_bp)
    self.assertGreater(ax_dot_bx, 0)
    ax_dot_cx = tf.math.reduce_sum(ab_x_ap * ca_x_cp)
    self.assertGreater(ax_dot_cx, 0)


  def test_sampled_point_normals(self,
                                 grid_dims: int,
                                 batch_size: int,
                                 occupancy_locs: List[List[int]]) -> None:
    """Verify the correctness of the sampled point normal vectors."""
    # pylint: disable=missing-param-doc
    verts, faces, verts_mask, faces_mask = self._get_verts_and_faces(
        grid_dims, batch_size, occupancy_locs
    )


    sampler = MeshSampler(self._num_samples)
    _, normals, sampled_verts_ind = sampler.sample_meshes(
        verts, verts_mask, faces, faces_mask
    )


    ## Verify shapes.
    self.assertAllEqual(tf.shape(normals), [batch_size, self._num_samples, 3])
    self.assertAllEqual(tf.shape(sampled_verts_ind),
                        [batch_size, self._num_samples, 2])

    ## Verify normal vector is actually normal to the face.
    v0, v1, _ = get_verts_from_indices(
        verts, verts_mask, faces, faces_mask, num_inds_per_set=3
    )
    va = tf.gather_nd(v0, sampled_verts_ind)
    vb = tf.gather_nd(v1, sampled_verts_ind)
    ab = va - vb

    ab_dot_norm = tf.math.reduce_sum(ab * normals, axis=-1)
    self.assertAllClose(ab_dot_norm, tf.zeros((batch_size, self._num_samples)))


if __name__ == '__main__':
  tf.test.main()
