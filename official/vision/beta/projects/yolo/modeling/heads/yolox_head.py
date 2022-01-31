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

# Lint as: python3
"""Yolox heads."""
import tensorflow as tf
from tensorflow.keras.models import Sequential

from official.vision.beta.projects.yolo.modeling.layers import nn_blocks
from official.vision.beta.projects.yolo.ops import box_ops


class YOLOXHead(tf.keras.layers.Layer):
  """YOLOX Prediction Head."""

  def __init__(
      self,
      min_level,
      max_level,
      classes=80,
      boxes_per_level=1,
      output_extras=0,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_regularizer=None,
      activation='silu',
      smart_bias=False,
      use_separable_conv=False,
      width_scaling=1.0,
      prior_prob=1e-2,
      **kwargs):

    """YoloX Prediction Head initialization function.

    Args:
      min_level: `int`, the minimum backbone output level.
      max_level: `int`, the maximum backbone output level.
      classes: `int`, number of classes per category.
      boxes_per_level: `int`, number of boxes to predict per level.
      output_extras: `int`, number of additional output channels that the head.
        should predict for non-object detection and non-image classification
        tasks.
      norm_momentum: `float`, normalization momentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      activation: `str`, the activation function to use. Default value: "silu".
      smart_bias: `bool`, whether to use smart bias.
      use_separable_conv: `bool` wether to use separable convs.
      width_scaling: `float`, factor by which the filters should be scaled.
      prior_prob: 'float', prior probability of custom value between 0.0 and 1. 
        Defaults to 1e-2.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level

    self._key_list = [
        str(key) for key in range(self._min_level, self._max_level + 1)
    ]

    self._classes = classes
    self._boxes_per_level = boxes_per_level
    self._output_extras = output_extras
    self._width_scaling = width_scaling
    self._smart_bias = smart_bias
    self._use_separable_conv = use_separable_conv
    self._prior_prob = prior_prob

    self._stems = dict()

    self._bias = -tf.math.log((1 - self._prior_prob) / self._prior_prob)

    self._base_config = dict(
        activation=activation,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)



  def build(self, input_shape):

    self._cls_convs = dict()
    self._reg_convs = dict()

    self._cls_preds = dict()
    self._reg_preds = dict()
    self._obj_preds = dict()

    self._cls_head = dict()
    self._obj_head = dict()
    self._reg_head = dict()

    for k in self._key_list:
      self._stems[k] = nn_blocks.ConvBN(
          filters=int(256 * self._width_scaling),
          kernel_size=(1, 1),
          strides=(1, 1),
          use_bn=True,
          use_separable_conv=self._use_separable_conv,
          **self._base_config,
      )

      self._cls_convs[k] = Sequential(
          [
              nn_blocks.ConvBN(
                  filters=int(256 * self._width_scaling),
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  use_bn=True,
                  use_separable_conv=self._use_separable_conv,
                  **self._base_config,
              ),
              nn_blocks.ConvBN(
                  filters=int(256 * self._width_scaling),
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  use_bn=True,
                  use_separable_conv=self._use_separable_conv,
                  **self._base_config,
              ),
          ]
      )

      self._reg_convs[k] = Sequential(
          [
              nn_blocks.ConvBN(
                  filters=int(256 * self._width_scaling),
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  use_bn=True,
                  use_separable_conv=self._use_separable_conv,
                  **self._base_config,
              ),
              nn_blocks.ConvBN(
                  filters=int(256 * self._width_scaling),
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  use_bn=True,
                  use_separable_conv=self._use_separable_conv,
                  **self._base_config,
              ),
          ]
      )

      self._cls_preds[k] = tf.keras.layers.Conv2D(
          filters=self._boxes_per_level * self._classes,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same',
          bias_initializer=tf.keras.initializers.constant(self._bias))

      self._reg_preds[k] = tf.keras.layers.Conv2D(
          filters=4,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same')

      self._obj_preds[k] = tf.keras.layers.Conv2D(
          filters=1 * self._boxes_per_level,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same',
          bias_initializer=tf.keras.initializers.constant(self._bias))

    for key in self._key_list:
      self._cls_head[key] = Sequential()
      self._cls_head[key].add(self._stems[key])
      self._cls_head[key].add(self._cls_convs[key])
      self._cls_head[key].add(self._cls_preds[key])

      self._obj_head[key] = Sequential()
      self._obj_head[key].add(self._stems[key])
      self._obj_head[key].add(self._reg_convs[key])
      self._obj_head[key].add(self._obj_preds[key])

      self._reg_head[key] = Sequential()
      self._reg_head[key].add(self._stems[key])
      self._reg_head[key].add(self._reg_convs[key])
      self._reg_head[key].add(self._reg_preds[key])

  def call(self, inputs, *args, **kwargs):
    outputs = dict()

    for k in self._key_list:
      ordered_preds = []
      cls_output = self._cls_head[k](inputs[k])
      reg_output = self._reg_head[k](inputs[k])
      obj_output = self._obj_head[k](inputs[k])

      for b in range(self._boxes_per_level):
        ordered_preds.append(reg_output[:,:,:,4 * b: 4 * (b + 1)])
        ordered_preds.append(obj_output[:,:,:,b: b + 1])
        ordered_preds.append(cls_output[:,:,:,self._classes * b: self._classes * (b + 1)])
      
      output = tf.concat(ordered_preds, axis=-1)
      outputs[k] = output
    #Outputs are not flattened here.
    return outputs

  def get_config(self):
      config = dict(
          min_level=self._min_level,
          max_level=self._max_level,
          classes=self._classes,
          boxes_per_level=self._boxes_per_level,
          output_extras=self._output_extras)
      return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
