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
"""Mesh R-CNN Heads.

TODO(zghera): Remove questions below once complete.

Currrent Questions
1. It looks like Pytorch impl uses something called group normalization
https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/config/config.py#L30
https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/batch_norm.py#L141
   I added a flag in __init__ to use this. But my question is should this layer
   be placed before or after the ReLU layer since there is no way to directly
   add this as an option to the Conv2d layer like they do in Pytorch? Based on
   what I read about BatchNorm, if GroupNorm behaves similarly then it should
   go before ReLU so that is what I did. But please correct me if I am wrong
   here.
2. The PyTorch implementation using a padding of 1 for the initial conv2d
   layers. But it appears that tensorflow only provides the options same and
   valid. So if my understanding is correct, if we use a kernel size of 3, then
   there are cases where 1 padding will not be the same as 'same' padding
   (e.g. 22 x 22).
3. Is it okay to not write argument docs for my tests as they are the same
   as the voxel head arguments?
"""
from typing import Optional

import tensorflow as tf  # type: ignore
import tensorflow_addons as tfa  # type: ignore


class VoxelHead(tf.keras.layers.Layer):
  """Mesh R-CNN Voxel Branch Prediction Head."""

  def __init__(self,
               voxel_depth: int,
               conv_dims: int,
               num_conv: int,
               use_group_norm: bool,
               predict_classes: bool,
               bilinearly_upscale_input: bool,
               class_based_voxel: bool,
               num_classes: int,
               kernel_regularizer:
               Optional[tf.keras.regularizers.Regularizer] = None,
               conv_bias_regularizer:
               Optional[tf.keras.regularizers.Regularizer] = None,
               conv_activ_regularizer:
               Optional[tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes a Voxel Branch Prediction Head.
    Args:
      voxel_depth: The number of depth channels for the predicted voxels.
      conv_dims: Number of output features for each Conv2D layer in the
        Voxel head.
      num_conv: Number of Conv2D layers prior to the Conv2DTranspose layer.
      use_group_norm: Whether or not to use GropNormalization in the fully
        connected layers.
      predict_classes: Whether or not to reshape the final predictor output
        from (N, CD, H, W) to (N, C, D, H, W) where C is `num_classes` to
        predict and D is `voxel_depth`. This option is used by the Pix3D
        Mesh R-CNN architecture.
      bilinearly_upscale_input: Whether or not to bilinearly resize the voxel
        head input tensor such that width and height of feature maps are equal
        to (`voxel_depth` // 2). This option is used by the ShapeNet Mesh R-CNN
        architecture.
      class_based_voxel: Whether or predict one of `num_classes` for each voxel
        grid output. If `predict_classes` is True but `class_based_voxel` is
        False, we will only predict 1 class. This option is used by the Pix3d
        Mesh R-CNN architecture.
      num_classes: If `class_based_voxel` is predict one of `num_classes`
        classes for each voxel. This option is used by the Pix3d Mesh R-CNN
        architecture.
      kernel_regularizer: Convolutional layer weight regularizer object.
      conv_bias_regularizer: Convolutional layer bias regularizer object.
      conv_activ_regularizer: Convolutional layer activation regularizer object.
      **kwargs: other keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._voxel_depth = voxel_depth
    self._conv_dims = conv_dims
    self._num_conv = num_conv
    self._use_group_norm = use_group_norm
    self._predict_classes = predict_classes
    self._bilinearly_upscale_input = bilinearly_upscale_input
    self._num_classes = num_classes if (
        predict_classes and class_based_voxel) else 1

    self._base_config = dict(
        activation=None,  # Apply ReLU separately in case we want to use GroupNorm
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=conv_bias_regularizer,
        activity_regularizer=conv_activ_regularizer)

    self._non_predictor_initializers = dict(
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'), # HeNormal with fan out
        bias_initializer=None if self._use_group_norm else 'zeros'
    )

    self._fully_conv2d_config = dict(
        filters=self._conv_dims,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=not self._use_group_norm,
        **self._non_predictor_initializers,
        **self._base_config)

    self._deconv2d_config = dict(
        filters=self._conv_dims,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        use_bias=True,
        **self._non_predictor_initializers,
        **self._base_config)
    self._deconv2d_config['activation'] = 'relu'

    self._predict_conv2d_config = dict(
        filters=self._num_classes * self._voxel_depth,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        bias_initializer=tf.keras.initializers.Zeros(),
        **self._base_config)

  def build(self, input_shape: tf.TensorShape) -> None:
    """Creates the voxel head layers and initializes their weights and biases.
    Args:
      input_shape: Shape of the input tensor to the voxel head.
        This the shape of the final layer of the backbone network for the
        ShapeNet model and the RoIAlign layer following the RPN for Pix3D.
    """
    #pylint: disable=unused-argument
    vd = self._voxel_depth
    self._interpolate = tf.keras.layers.Resizing(
        height=(vd // 2), width=(vd // 2), interpolation='bilinear')
    self._reshape = tf.keras.layers.Reshape((self._num_classes, vd, vd, vd))

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

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of the voxel head for the ShapeNet Mesh R-CNN model.
    Args:
      inputs: This is the tensor output of the final layer of the backbone
        network for the ShapeNet model and the RoIAlign layer following the
        RPN for Pix3D.
    Returns:
      (N, V, V, V) for ShapeNet model and (N, C, V, V, V) for Pix3D model
      where N = batch size, V = `voxel_depth`, and C = `num_classes`.
    """
    x = tf.cond(self._bilinearly_upscale_input,
                true_fn=lambda: self._interpolate(inputs),
                false_fn=lambda: tf.keras.layers.Lambda(lambda x: x)(inputs))
    for layer in self._conv2d_norm_relu_layers:
      x = layer(x)
    x = self._deconv(x)
    x = self._predictor(x)
    x = tf.cond(self._predict_classes,
                true_fn=lambda: self._reshape(x),
                false_fn=lambda: tf.keras.layers.Lambda(lambda x: x)(x))
    return x

  @property
  def output_depth(self) -> int:
    return self._voxel_depth

  def get_config(self) -> dict:
    config = dict(
        input_channels=self._input_channels,
        voxel_depth=self._voxel_depth,
        conv_dims=self._conv_dims,
        num_conv=self._num_conv,
        use_group_norm=self._use_group_norm,
        **self._base_config)
    return config
