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

"""Backbones configurations."""
import dataclasses
from typing import Optional
from official.modeling import hyperparams
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class ResNet(hyperparams.Config):
  """ResNet config."""
  model_id: int = 50
  depth_multiplier: float = 1.0
  stem_type: str = 'v0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0
  scale_stem: bool = True
  resnetd_shortcut: bool = False
  replace_stem_max_pool: bool = False
  bn_trainable: bool = True


@dataclasses.dataclass
class Darknet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = 'cspdarknet53'
  width_scale: float = 1.0
  depth_scale: float = 1.0
  dilate: bool = False
  min_level: int = 3
  max_level: int = 5
  use_separable_conv: bool = False
  use_reorg_input: bool = False


@dataclasses.dataclass
class SpineNet(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  min_level: int = 3
  max_level: int = 7


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  type: Optional[str] = None
  resnet: ResNet = ResNet()
  spinenet: SpineNet = SpineNet()
  darknet: Darknet = Darknet()