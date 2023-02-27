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

from official.vision.configs import common
from official.modeling import hyperparams  # type: ignore
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.projects.mesh_rcnn import optimization
#from official.vision.beta.projects.mesh_rcnn.tasks import mesh_rcnn
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones
from official.core import config_definitions as cfg

#Parser and dataconfig from Mask-RCNN(Subject to change)
@dataclasses.dataclass
class Parser(hyperparams.Config):
  num_channels: int = 3
  match_threshold: float = 0.5
  unmatched_threshold: float = 0.5
  rpn_match_threshold: float = 0.7, 
  rpn_unmatched_threshold: float = 0.3, 
  rpn_batch_size_per_im: int = 256, 
  rpn_fg_fraction: float = 0.5, 
  aug_rand_hflip: bool = False, 
  aug_scale_min: float = 1, 
  aug_scale_max: int = 1, 
  skip_crowd_during_training: bool = True, 
  max_num_instances: int =100, 
  max_num_verts: int = 108416, 
  max_num_faces: int = 126748, 
  max_num_voxels: int = 2097152, 
  include_mask: bool = True, 
  mask_crop_size: int = 112

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: common.DataDecoder = common.DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True
  # Number of examples in the data set, it's used to create the annotation file.
  num_examples: int = -1

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
  
@dataclasses.dataclass
class MeshRCNN(hyperparams.Config):
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(
      type='fpn', fpn=decoders.FPN())
      
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
                  'adam': {
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

