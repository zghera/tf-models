# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import csv
# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import decoder


def _generate_source_id(image_bytes):
  return tf.strings.as_string(
      tf.strings.to_hash_bucket_fast(image_bytes, 2 ** 63 - 1))


class TfdsExampleDecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               include_mask=False,
               regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id
    if include_mask:
        raise ValueError("TensorFlow Datasets doesn't support masks")

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    return parsed_tensors['image']

  def _decode_boxes(self, parsed_tensors):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    return parsed_tensors['objects']['bbox']

  def _decode_classes(self, parsed_tensors):
    return parsed_tensors['objects']['label']

  def _decode_areas(self, parsed_tensors):
    ymin = parsed_tensors['objects']['bbox'][..., 0]
    xmin = parsed_tensors['objects']['bbox'][..., 1]
    ymax = parsed_tensors['objects']['bbox'][..., 2]
    xmax = parsed_tensors['objects']['bbox'][..., 3]
    shape = tf.cast(tf.shape(parsed_tensors['image']), tf.float32)
    width = shape[0]
    height = shape[1]
    return (ymax - ymin) * (xmax - xmin) * width * height
    # return parsed_tensors['objects']['area']

  def _decode_masks(self, parsed_tensors):
    """Decode a set of PNG masks to the tf.float32 tensors."""
    return

  def decode(self, serialized_example):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - source_id: a string scalar tensor.
        - image: a uint8 tensor of shape [None, None, 3].
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    parsed_tensors = serialized_example
#    for k in parsed_tensors:
#      if isinstance(parsed_tensors[k], tf.SparseTensor):
#        if parsed_tensors[k].dtype == tf.string:
#          parsed_tensors[k] = tf.sparse.to_dense(
#              parsed_tensors[k], default_value='')
#        else:
#          parsed_tensors[k] = tf.sparse.to_dense(
#              parsed_tensors[k], default_value=0)

#    if self._regenerate_source_id:
#      source_id = _generate_source_id(parsed_tensors['image/id'])
#    else:
#      source_id = tf.cond(
#          tf.greater(tf.strings.length(parsed_tensors['image/id']), 0),
#          lambda: parsed_tensors['image/id'],
#          lambda: _generate_source_id(parsed_tensors['image']))
    source_id = parsed_tensors['image/id']
    image = self._decode_image(parsed_tensors)
    boxes = self._decode_boxes(parsed_tensors)
    classes = self._decode_classes(parsed_tensors)
    areas = self._decode_areas(parsed_tensors)
    is_crowds = tf.cond(
        tf.greater(tf.shape(parsed_tensors['objects']['label'])[0], 0),
        lambda: tf.cast(parsed_tensors['objects']['is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(classes, dtype=tf.bool))

    decoded_tensors = {
        'source_id': source_id,
        'image': image,
        'width': tf.shape(parsed_tensors['image'])[0],
        'height': tf.shape(parsed_tensors['image'])[1],
        'groundtruth_classes': classes,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    #tf.print(decoded_tensors)
    return decoded_tensors

