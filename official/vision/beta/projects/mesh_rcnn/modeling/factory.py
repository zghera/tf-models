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
"""Contains factory functions for Mesh R-CNN networks."""

from typing import Optional
from requests import head

import tensorflow as tf  # type: ignore

from official.vision.modeling import backbones
from official.vision.modeling import decoders

from official.vision.beta.projects.mesh_rcnn.configs.mesh_rcnn import MeshRCNN as mesh_rcnn_cfg
from official.vision.beta.projects.mesh_rcnn.configs.mesh_rcnn import VoxelHead
from official.vision.beta.projects.mesh_rcnn.configs.mesh_rcnn import MeshHead
from official.vision.beta.projects.mesh_rcnn.configs.mesh_rcnn import ZHead

from official.vision.beta.projects.mesh_rcnn.modeling.heads import voxel_head
from official.vision.beta.projects.mesh_rcnn.modeling.heads import mesh_head
from official.vision.beta.projects.mesh_rcnn.modeling.heads import z_head


def build_z_head(head_config: ZHead) -> z_head.ZHead:
  """Builds Z Prediction Head.
  Args:
    head_config: Dataclass parameterization instance for z head.

  Returns:
    Z head layer instance.
  """
  return z_head.ZHead(
    num_fc=head_config.num_fc,
    fc_dim=head_config.fc_dim,
    cls_agnostic=head_config.cls_agnostic,
    num_classes=head_config.num_classes
  )


def build_mesh_head(head_config: MeshHead) -> mesh_head.MeshHead:
  """Builds Mesh Branch Prediction Head.
  Args:
    head_config: Dataclass parameterization instance for mesh head.

  Returns:
    Mesh head layer instance.
  """
  return mesh_head.MeshHead(
    num_stages=head_config.num_stages,
    stage_depth=head_config.stage_depth,
    output_dim=head_config.output_dim,
    graph_conv_init=head_config.graph_conv_init
  )

def build_voxel_head(head_config: VoxelHead,
                      kernel_regularizer:
                      Optional[tf.keras.regularizers.Regularizer],
                      bias_regularizer:
                      Optional[tf.keras.regularizers.Regularizer],
                      activity_regularizer:
                      Optional[tf.keras.regularizers.Regularizer]
                    ) -> voxel_head.VoxelHead:
  """Builds Voxel Branch Prediction Head.
  Args:
    head_config: Dataclass parameterization instance for voxel head.
    kernel_regularizer: Convolutional layer weight regularizer object.
    bias_regularizer: Convolutional layer bias regularizer object.
    activity_regularizer: Convolutional layer activation regularizer object.
  Returns:
    Voxel head layer instance.
  """
  return voxel_head.VoxelHead(
    voxel_depth=head_config.voxel_depth,
    conv_dim=head_config.conv_dim,
    num_conv=head_config.num_conv,
    use_group_norm=head_config.use_group_norm,
    predict_classes=head_config.predict_classes,
    bilinearly_upscale_input=head_config.bilinearly_upscale_input,
    class_based_voxel=head_config.class_based_voxel,
    num_classes=head_config.num_classes,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
  )
def build_mesh_rcnn(input_specs: tf.keras.layers.InputSpec,
                   model_config: mesh_rcnn_cfg,
                   l2_regularizer: Optional[
                       tf.keras.regularizers.Regularizer] = None,
                   backbone: Optional[tf.keras.Model] = None,
                   decoder: Optional[tf.keras.Model] = None) -> tf.keras.Model:
  """Builds mesh_rcnn model."""
  # TODO: do we need this? - if we do then take it from mask rcnn
  #norm_activation_config = model_config.norm_activation 

  if not backbone:
    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)
  backbone_features = backbone(tf.keras.Input(input_specs.shape[1:]))

  if not decoder:
    decoder = decoders.factory.build_decoder(
        input_specs=backbone.output_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

  return model