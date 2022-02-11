"""Configs for model components."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict

from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_classes import (
    BatchNormCFG, BottleneckBlockCFG, Conv2dCFG)


class ConfigData(ABC):
  """Weight config list for components."""

  def get_cfg_list(self, name):
    "Gets component configuration list from specified name."
    print(name)

@dataclass
class BackboneConfigData(ConfigData):
  """Weight config for MaskRCNN resnet50 backbone"""

  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "resnet50":
      return [
        Conv2dCFG(weights_dict=self.weights_dict["stem"]["conv1"]),
        BatchNormCFG(weights_dict=self.weights_dict["stem"]["conv1"]["norm"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res2"]["0"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res2"]["1"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res2"]["2"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res3"]["0"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res3"]["1"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res3"]["2"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res3"]["3"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res4"]["0"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res4"]["1"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res4"]["2"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res4"]["3"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res4"]["4"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res4"]["5"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res5"]["0"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res5"]["1"]),
        BottleneckBlockCFG(weights_dict=self.weights_dict["res5"]["2"])
      ]
    return []


@dataclass
class DecoderConfigData(ConfigData):
  """Weight config for MaskRCNN FPN decoder"""

  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "fpn":
      return [
        Conv2dCFG(weights_dict=self.weights_dict["fpn_lateral5"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_lateral4"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_lateral3"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_lateral2"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_output2"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_output3"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_output4"]),
        Conv2dCFG(weights_dict=self.weights_dict["fpn_output5"])
      ]
    return []
