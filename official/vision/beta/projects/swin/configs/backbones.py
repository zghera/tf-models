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
from typing import Optional, List
from official.modeling import hyperparams
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class Swin(hyperparams.Config):
  min_level: Optional[int] = None
  max_level: Optional[int] = None
  embed_dims: int = 96
  depths: List[int] = dataclasses.field(default_factory=lambda:[2, 2, 6, 2])
  num_heads: List[int] = dataclasses.field(default_factory=lambda:[3, 6, 12, 24]) 
  window_size: List[int] = dataclasses.field(default_factory=lambda:[7, 7, 7, 7]) 
  patch_size: int = 4
  mlp_ratio: float = 4
  qkv_bias: bool = True
  qk_scale: bool = None
  dropout: float = 0.0
  attention_dropout: float = 0.0
  drop_path: float = 0.1
  absolute_positional_embed: bool = False
  normalize_endpoints: bool = True
  norm_layer: str = 'layer_norm'

@dataclasses.dataclass
class Backbone(backbones.Backbone):
  swin: Swin = Swin()
