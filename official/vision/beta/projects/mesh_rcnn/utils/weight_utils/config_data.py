from dataclasses import dataclass, field
from typing import Dict

from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_classes import (
    batchNormCFG, bottleneckBlockCFG, conv2dCFG)


@dataclass
class BackboneConfigData():
  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "resnet50":
      return [
        conv2dCFG(weights_dict=self.weights_dict['stem']['conv1']),
        batchNormCFG(weights_dict=self.weights_dict['stem']['conv1']['norm']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res2']['0']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res2']['1']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res2']['2']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res3']['0']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res3']['1']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res3']['2']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res3']['3']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res4']['0']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res4']['1']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res4']['2']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res4']['3']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res4']['4']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res4']['5']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res5']['0']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res5']['1']),
        bottleneckBlockCFG(weights_dict=self.weights_dict['res5']['2'])
      ]


@dataclass
class HeadConfigData():
  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "rpn":
      return [  
      ]