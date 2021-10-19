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
"""Tests for Mesh R-CNN Heads."""

from typing import Tuple

import tensorflow as tf  # type: ignore
from absl.testing import parameterized  # type: ignore

from official.vision.beta.projects.mesh_rcnn.modeling.heads import voxel_head


@parameterized.named_parameters(
  {'testcase_name': 'shapenet',
  'predict_classes': False, 'class_based_voxel': False, 'num_conv': 2,
  'voxel_depth': 48, 'batch_size': 32, 'num_input_channels': 2048},
  {'testcase_name': 'pix3d-class-agnostic',
  'predict_classes': True, 'class_based_voxel': False, 'num_conv': 1,
  'voxel_depth': 24, 'batch_size': 1, 'num_input_channels': 256},
  {'testcase_name': 'pix3d-class-based',
  'predict_classes': True, 'class_based_voxel': True, 'num_conv': 0,
  'voxel_depth': 24, 'batch_size': 32, 'num_input_channels': 256},
)
class VoxelHeadTest(parameterized.TestCase, tf.test.TestCase):
  """Test for Mesh R-CNN Voxel Prediction Head."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._num_classes = 5
    self._conv_dims = 256
    self._use_group_norm = False

  def _get_expected_out_shape(self,
                            predict_classes: bool,
                            class_based_voxel: bool,
                            voxel_depth: int,
                            batch_size: int) -> Tuple[int, ...]:
    """Get the output shape of the voxel head."""
    # pylint: disable=missing-param-doc
    expected_shape: Tuple[int, ...]
    if predict_classes:
      expected_num_classes: int = self._num_classes if class_based_voxel else 1
      expected_shape = (batch_size, expected_num_classes,
                        voxel_depth, voxel_depth, voxel_depth)
    else:
      expected_shape = (batch_size, voxel_depth, voxel_depth, voxel_depth)
    return expected_shape

  def _get_input_shape(self,
                       voxel_depth: int,
                       batch_size: int,
                       num_input_channels: int) -> Tuple[int, int, int, int]:
    """Get the output input shape of the voxel head."""
    return (batch_size, voxel_depth // 2, voxel_depth // 2, num_input_channels)

  def test_network_creation(self,
                            predict_classes: bool,
                            class_based_voxel: bool,
                            num_conv: int,
                            voxel_depth: int,
                            batch_size: int,
                            num_input_channels: int) -> None:
    """Verify the output shapes of the voxel head."""
    # pylint: disable=missing-param-doc
    tf.keras.backend.set_image_data_format('channels_last')
    head = voxel_head.VoxelHead(voxel_depth, self._conv_dims, num_conv,
                                self._use_group_norm, predict_classes,
                                not predict_classes, class_based_voxel,
                                self._num_classes)

    input_shape = self._get_input_shape(voxel_depth, batch_size,
                                        num_input_channels)
    input_tensor = tf.ones(input_shape, dtype=tf.float32)
    output = head(input_tensor)

    expected_shape = self._get_expected_out_shape(predict_classes,
                                                  class_based_voxel,
                                                  voxel_depth, batch_size)

    self.assertAllEqual(output.shape.as_list(), expected_shape)

  def test_serialize_deserialize(self,
                                 predict_classes: bool,
                                 class_based_voxel: bool,
                                 num_conv: int,
                                 voxel_depth: int,
                                 batch_size: int,
                                 num_input_channels: int) -> None:
    """Create a network object that sets all of its config options."""
    # pylint: disable=missing-param-doc
    tf.keras.backend.set_image_data_format('channels_last')
    head = voxel_head.VoxelHead(voxel_depth, self._conv_dims, num_conv,
                                self._use_group_norm, predict_classes,
                                not predict_classes, class_based_voxel,
                                self._num_classes)

    input_shape = self._get_input_shape(voxel_depth, batch_size,
                                        num_input_channels)
    input_tensor = tf.ones(input_shape, dtype=tf.float32)
    _ = head(input_tensor)

    serialized = head.get_config()
    deserialized = voxel_head.VoxelHead.from_config(serialized)

    self.assertAllEqual(head.get_config(), deserialized.get_config())

if __name__ == '__main__':
  # from mesh_rcnn.utils.run_utils import prep_gpu
  # prep_gpu()
  tf.test.main()
