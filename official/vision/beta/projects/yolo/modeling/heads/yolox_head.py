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
from loguru import logger
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
      width_scaling = 1.0,
      strides=[8, 16, 32],
      in_channels=[256, 512, 1024],
      depthwise=False,
      prior_prob = 1e-2,
      **kwargs
  ):


    """
    Args:
        num_classes: `int`, number of classes per category.
        act (str): activation type of conv. Defalut value: "silu".
        depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
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

    self._smart_bias = smart_bias
    self._use_separable_conv = use_separable_conv
    self._prior_prob = prior_prob

    self._cls_convs = dict()
    self._reg_convs = dict()
    
    self._cls_preds = dict()
    self._reg_preds = dict()
    self._obj_preds = dict()
    self._stems = dict()

    
    self._bias= -tf.math.log((1 - self._prior_prob) / self._prior_prob)

    Conv = nn_blocks.DWConv if depthwise else nn_blocks.BaseConv 
    for k in self._key_list:
      self._stems[k] = nn_blocks.BaseConv(
          filters=int(256 * width_scaling),
          kernel_size=(1,1),
          strides=(1,1),
          padding='same', 
          # use_bn = True,
          activation=activation,
      )

      self._cls_convs[k] = Sequential(
          [
          Conv(
            filters=int(256 * width_scaling),
            kernel_size=(3,3),
            strides=(1,1),
            # use_bn = True,
            activation=activation,
          ),
          Conv(
            filters=int(256 * width_scaling),
            kernel_size=(3,3),
            strides=(1,1),
            # use_bn = True,
            activation=activation,
          ),
          ]
        )

      self._reg_convs[k] = Sequential(
          [
            Conv(
              filters=int(256 * width_scaling),
              kernel_size=(3,3),
              strides=(1,1),
              # use_bn = True,
              activation=activation,
            ),
            Conv(
              filters=int(256 * width_scaling),
              kernel_size=(3,3),
              strides=(1,1),
              # use_bn = True,
              activation=activation,
            ),
          ]
        )
      
      self._cls_preds[k] = tf.keras.layers.Conv2D(
          filters=self._boxes_per_level * self._classes,
          kernel_size=(1,1),
          strides=(1,1),
          padding='same',
          bias_initializer=tf.keras.initializers.constant(self._bias))
      
      self._reg_preds[k] = tf.keras.layers.Conv2D(
          filters=4,
          kernel_size=(1,1),
          strides=(1,1),
          padding='same')
      
      self._obj_preds[k] = tf.keras.layers.Conv2D(
          filters=self._boxes_per_level * 1,
          kernel_size=(1,1),
          strides=(1,1),
          padding='same',
          bias_initializer=tf.keras.initializers.constant(self._bias))

  def build(self, input_shape):
    self._cls_head = dict()
    self._obj_head = dict()
    self._reg_head = dict()

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
    outputs=dict()
    for k in self._key_list:    
        cls_output = self._cls_head[k](inputs[k])
        reg_output = self._reg_head[k](inputs[k])
        obj_output = self._obj_head[k](inputs[k])
        output=tf.concat([reg_output,obj_output,cls_output], axis = -1)
        outputs[k] = output
      #TODO flatten

    return outputs
