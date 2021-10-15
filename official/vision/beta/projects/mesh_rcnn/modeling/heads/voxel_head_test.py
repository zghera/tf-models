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

import tensorflow as tf  # type: ignore
import voxel_head
from absl.testing import parameterized  # type: ignore


@parameterized.product(
    predict_classes=[False, True],
    class_based_voxel=[False, True],
    num_classes=[1, 5],
    voxel_depth=[24, 48],
    conv_dims=[256],
    num_conv=[0, 2],
    use_group_norm=[False, True],
)
class VoxelHeadTest(parameterized.TestCase, tf.test.TestCase):
  """Test for Mesh R-CNN Voxel Prediction Head."""

  def test_network_output(self,
                          predict_classes: bool,
                          class_based_voxel: bool,
                          num_classes: int,
                          voxel_depth: int,
                          conv_dims: int,
                          num_conv: int,
                          use_group_norm: bool) -> None:
    """Verify the output shapes of the voxel head."""
    # pylint: disable=missing-param-doc
    tf.keras.backend.set_image_data_format('channels_last')
    head = voxel_head.VoxelHead(voxel_depth, conv_dims, num_conv,
                                use_group_norm, predict_classes,
                                not predict_classes, class_based_voxel,
                                num_classes)
    batch_size = 32
    num_input_channels = 256
    input_shape = [batch_size, voxel_depth // 2, voxel_depth // 2,
                   num_input_channels]
    input_tensor = tf.ones(input_shape, dtype=tf.float32)
    output = head(input_tensor)

    if predict_classes:
      expected_num_classes = num_classes if class_based_voxel else 1
      expected_shape = [batch_size, expected_num_classes, voxel_depth,
                        voxel_depth, voxel_depth]
    else:
      expected_shape = [batch_size, voxel_depth, voxel_depth, voxel_depth]

    self.assertAllEqual(output.shape.as_list(), expected_shape)

if __name__ == '__main__':
  # from mesh_rcnn.utils.run_utils import prep_gpu
  # prep_gpu()
  tf.test.main()
