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
"""Mesh R-CNN configuration definition."""

import dataclasses

from official.modeling import hyperparams  # type: ignore
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.projects.mesh_rcnn import optimization
from official.vision.beta.projects.mesh_rcnn.tasks import mesh_rcnn
from official.vision.configs import common

@dataclasses.dataclass
class ZHead(hyperparams.Config):
    """Parameterization for the Mesh R-CNN Z Head."""
    num_fc: int = 2
    fc_dim: int = 1024
    cls_agnostic: bool = False
    num_classes: int = 9

@dataclasses.dataclass
class VoxelHead(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Voxel Branch Prediction Head."""
  voxel_depth: int = 28
  conv_dim: int = 256
  num_conv: int = 0
  use_group_norm: bool = False
  predict_classes: bool = False
  bilinearly_upscale_input: bool = True
  class_based_voxel: bool = False
  num_classes: int = 0

@dataclasses.dataclass
class MeshHead(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Mesh Head."""
  num_stages: int = 3
  stage_depth: int = 3
  output_dim: int = 128
  graph_conv_init: str = 'normal'

@dataclasses.dataclass
class MeshLosses(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Mesh and Voxel Losses."""
  voxel_weight: float = 0.0
  chamfer_weight: float = 1.0
  normal_weight: float = 0.0
  edge_weight: float = 0.1
  true_num_samples: int = 5000
  pred_num_samples: int = 5000

@exp_factory.register_config_factory('mesh_training')
def mesh_training() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv3 and v4."""
  train_batch_size = 256
  eval_batch_size = 8
  train_epochs = 300
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  validation_interval = 5

  max_num_instances = 200
  config = cfg.ExperimentConfig(
      trainer=cfg.TrainerConfig(
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema': {
                  'average_decay': 0.9998,
                  'trainable_weights_only': False,
                  'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'adam',
                  'sgd_torch': {
                      'learning_rate' : 0.001, 
                      'beta_1' : 0.9, 
                      'beta_2' : 0.999, 
                      'epsilon' : 1e-07
                  }
              },
              'learning_rate': {},
              'warmup': {}
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config