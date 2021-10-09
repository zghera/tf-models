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
"""Currrent Questions
1. Should I be setting the default values for the head based on what they use
   in shapenet/config/config.py or one of the actual configs like
   configs/shapenet/voxmesh_R50.yaml
2. I am not sure what the correct kernel initializer is, based on this post
https://discuss.pytorch.org/t/crossentropyloss-expected-object-of-type-torch-longtensor/28683/6?u=ptrblck
   I think it is HeNormal but I could be wrong.
3. It looks like Pytorch impl uses something called group normalization
https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/config/config.py#L30
https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/batch_norm.py#L141
   I added a flag in __init__ to use this. But my question is should this layer
   be placed before or after the ReLU layer since there is no way to directly
   add this as an option to the Conv2d layer like they do in Pytorch? Based on
   what I read about BatchNorm, if GroupNorm behaves similarly then it should
   go before ReLU so that is what I did. But please correct me if I am wrong
   here.
"""
from typing import Any, Optional

import tensorflow as tf  # type: ignore

import tensorflow_addons as tfa  # type: ignore


class VoxelHead(tf.keras.layers.Layer):
  """Mesh R-CNN Voxel Branch Prediction Head."""

  def __init__(self,
               input_channels: int,
               voxel_size: int = 28,
               conv_dims: int = 256,
               num_conv: int = 0,
               use_group_norm: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_initializer: str = 'HeNormal',
               kernel_regularizer:
               Optional[tf.keras.regularizers.Regularizer] = None,
               bias_regularizer:
               Optional[tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes a Voxel Branch Prediction Head.
    Args:
      input_channels: Number of channels in layer preceeding the voxel head.
        This the final conv5_3 layer of the backbone network for ShapeNet
        model and the RoIAlign layer following the RPN for Pix3D.
      voxel_size: The number of depth channels for the predicted voxels.
      conv_dims: Number of output features for each Conv2D layer in the
        Voxel head.
      num_conv: Number of Conv2D layers prior to the Conv2DTranspose layer.
      use_group_norm: Whether or not to use GropNormalization in fully
        connected layer(s).
      norm_momentum: Normalization momentum for the moving average.
      norm_epsilon: Small float added to variance to avoid dividing by zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    assert self.voxel_size % 2 == 0

    self._input_channels = input_channels
    self._voxel_size = voxel_size
    self._conv_dims = conv_dims
    self._num_conv = num_conv
    self._use_group_norm = use_group_norm

    self._base_config = dict(
        activation=None,  # Apply ReLU separately in case we want to use GroupNorm
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

    self._fully_conv2d_config = dict(
        filters=self._conv_dims,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=1,
        use_bias=not self._use_group_norm,
        data_format='channels_last',
        **self._base_config)

    self._deconv2d_config = dict(
        filters=self._conv_dims,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding=0,
        use_bias=True,
        **self._base_config)
    self._deconv2d_config['activation'] = 'relu'

    self._predict_conv2d_config = dict(
        filters=self._voxel_size,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=0,
        use_bias=True,
        **self._base_config)

  def build(self, input_shape: Any) -> None:
    """TODO(zghera)
    """
    #pylint: disable=unused-argument, missing-param-doc
    self._interpolate = tf.keras.layers.UpSampling2D(
        size=(self._voxel_size // 2, self._voxel_size // 2),
        interpolation="bilinear")

    self._conv2d_norm_relu_layers = []
    for _ in range(self._num_conv):
      conv = tf.keras.layers.Conv2D(**self._fully_conv2d_config)
      self._conv2d_norm_relu_layers.append(conv)
      if self._use_group_norm:
        group_norm = tfa.layers.GroupNormalization(groups=32, axis=-1)
        self._conv2d_norm_relu_layers.append(group_norm)
      relu = tf.keras.layers.ReLU()
      self._conv2d_norm_relu_layers.append(relu)

    self._deconv = tf.keras.layers.Conv2DTranspose(**self._deconv2d_config)
    self._predictor = tf.keras.layers.Conv2D(**self._predict_conv2d_config)

    # TODO(zghera): Weight and bias initializations

  def call(self, inputs: Any) -> Any:
    """TODO(zghera)
    Args:
      inputs: ...
    Return:
      ...
    """
    # pylint: disable=arguments-differ
    x = self._interpolate(inputs)
    for layer in self._conv2d_norm_relu_layers:
      x = layer(x)
    x = self._deconv(x)
    return self._predictor(x)

  @property
  def output_depth(self) -> int:
    return self._voxel_size

  def get_config(self) -> dict:
    config = dict(
        input_channels=self._input_channels,
        voxel_size=self._voxel_size,
        conv_dims=self._conv_dims,
        num_conv=self._num_conv,
        use_group_norm=self._use_group_norm,
        **self._base_config)
    return config
