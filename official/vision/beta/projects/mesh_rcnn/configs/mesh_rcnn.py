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

from official.modeling import hyperparams # type: ignore


@dataclasses.dataclass
class ZHead(hyperparams.Config):
    """Parameterization for the Mesh R-CNN Z Head."""
    name: str = "fastRCNNFCHead"
    num_fc: int = 2
    fc_dim: int = 1024
    cls_agnostic: bool = False
    num_classes: int = 9
    pooler_resolution: int = 7
    pooler_sampling_ratio: int = 2
    pooler_type: str = "roi_align"
    z_reg_weight: float = 5.0
    smooth_l1_beta: float = 0.0 
@dataclasses.dataclass
class VoxelHead(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Voxel Branch Prediction Head."""
  name: str = 'VoxelRCNNConvUpsampleHead'
  voxel_depth: int = 28
  conv_dim: int = 256
  num_conv: int = 0
  use_group_norm: bool = False
  predict_classes: bool = False
  bilinearly_upscale_input: bool = True
  class_based_voxel: bool = False
  num_classes: int = 0
  cubify_thresh: float = 0.0
  cubify_loss_weight: float = 1.0
  norm : str = ''
  cls_agnostic_voxel: bool = False
  pooler_type: str = 'roi_align'
  pooler_sampling_ratio : int = 0
  pooler_resolution : int = 14

@dataclasses.dataclass
class MeshHead(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Mesh Head."""
  name: str = 'MeshRCNNGraphConvHead'
  pooler_resolution: int = 14
  pooler_sampling_ratio: int = 0
  pooler_type: str = 'roi_align'
  num_stages: int = 3 #why this one is different from the original implementation?
  num_graph_conv: int = 3 
  graph_conv_dim: int = 256
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
  

