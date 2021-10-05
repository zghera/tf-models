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
"""Image classification with darknet configs."""

import dataclasses
from typing import List, Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.configs import common
from official.vision.beta.configs import image_classification as imc
from official.vision.beta.projects.swin.configs import backbones

from official.vision.beta.configs import common


@dataclasses.dataclass
class ImageClassificationModel(hyperparams.Config):
  num_classes: int = 1000
  input_size: List[int] = dataclasses.field(default_factory=lambda:[224, 224])
  backbone: backbones.Backbone = backbones.Backbone(type='swin', swin=backbones.Swin())
  dropout_rate: float = 0.0
  norm_activation: common.NormActivation = common.NormActivation()
  # Adds a Batch Normalization layer pre-GlobalAveragePooling in classification.
  add_head_batch_norm: bool = False
  kernel_initializer: str = "TruncatedNormal"


@dataclasses.dataclass
class Losses(hyperparams.Config):
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0

@exp_factory.register_config_factory('swin_classification')
def swin_classification() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=imc.ImageClassificationTask(
        model=ImageClassificationModel(),
        train_data = imc.DataConfig(
            is_training=True, 
            random_erasing=common.RandomErasing(),
            mixup_and_cutmix=common.MixupAndCutmix(),
            aug_type=common.Augmentation(type = 'randaug',randaug = common.RandAugment()))
            ),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


