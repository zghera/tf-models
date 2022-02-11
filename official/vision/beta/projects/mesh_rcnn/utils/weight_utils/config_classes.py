"""Config classes for keras layer weight loading from Pytorch Dict"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import tensorflow as tf


class Config(ABC):
  """Base class for layer configs"""

  def get_weights(self):
    """
    This function generates the weights needed to be loaded into the layer.
    """
    return []

  def load_weights(self, layer: tf.keras.layers.Layer) -> int:
    """
    Given a layer, this function retrives the weights for that layer in an
    appropriate format, and loads them into the layer.

    Args:
      layer: keras layer to load weights into.

    Returns:
      Number of weights loaded.
    """

    weights = self.get_weights()
    layer.set_weights(weights)
    n_weights = 0

    for w in weights:
      n_weights += np.size(w)
    return n_weights


@dataclass
class Conv2dCFG(Config):
  """Config class for keras.Conv2D layer."""

  weights_dict: Dict = field(repr=False, default=None)
  weights: List = field(repr=False, default=None)

  def __post_init__(self):
    self.weights = [self.weights_dict['weight']]
    if 'bias' in self.weights_dict:
      self.weights.append(self.weights_dict['bias'])

  def get_weights(self):
    return self.weights


@dataclass
class BatchNormCFG(Config):
  """Config class for keras.BatchNormalization layer."""

  weights_dict: Dict = field(repr=False, default=None)
  weights: List = field(repr=False, default=None)

  def __post_init__(self):
    self.weights = [self.weights_dict['weight'],
                    self.weights_dict['bias'],
                    self.weights_dict['running_mean'],
                    self.weights_dict['running_var']]

  def get_weights(self):
    return self.weights


@dataclass
class BottleneckBlockCFG(Config):
  """Config class for BottleneckBlock layer."""

  weights_dict: Dict = field(repr=False, default=None)
  weights: List = field(repr=False, default=None)
  bn_vars: List = field(repr=False, default=None)

  def __post_init__(self):
    if 'shortcut' in self.weights_dict:
      self.weights = [self.weights_dict['shortcut']['weight'],
                      self.weights_dict['shortcut']['norm']['weight'],
                      self.weights_dict['shortcut']['norm']['bias']]
      self.bn_vars = [self.weights_dict['shortcut']['norm']['running_mean'],
                      self.weights_dict['shortcut']['norm']['running_var']]
    else:
      self.weights = []
      self.bn_vars = []

    for j in range(1,4):
      self.weights.append(self.weights_dict['conv'+str(j)]['weight'])
      self.weights.append(self.weights_dict['conv'+str(j)]['norm']['weight'])
      self.weights.append(self.weights_dict['conv'+str(j)]['norm']['bias'])

      self.bn_vars.append(
        self.weights_dict['conv'+str(j)]['norm']['running_mean'])
      self.bn_vars.append(
        self.weights_dict['conv'+str(j)]['norm']['running_var'])

  def get_weights(self):
    return self.weights + self.bn_vars


@dataclass
class VoxelHeadCFG(Config):
  """Config class for VoxelHead layer."""

  weights_dict: Dict = field(repr=False, default=None)
  weights: List = field(repr=False, default=None)

  def __post_init__(self):
    self.weights = [self.weights_dict['voxel_fcn1']['weight'],
                    self.weights_dict['voxel_fcn1']['bias'],
                    self.weights_dict['voxel_fcn2']['weight'],
                    self.weights_dict['voxel_fcn2']['bias'],
                    self.weights_dict['voxel_fcn3']['weight'],
                    self.weights_dict['voxel_fcn3']['bias'],
                    self.weights_dict['voxel_fcn4']['weight'],
                    self.weights_dict['voxel_fcn4']['bias'],
                    self.weights_dict['deconv']['weight'],
                    self.weights_dict['deconv']['bias'],
                    self.weights_dict['predictor']['weight'],
                    self.weights_dict['predictor']['bias']]

  def get_weights(self):
    return self.weights


@dataclass
class RPNHeadCFG(Config):
  """Config class for RPN layer."""

  weights_dict: Dict = field(repr=False, default=None)
  weights: List = field(repr=False, default=None)

  def __post_init__(self):
    self.weights = [self.weights_dict['conv']['weight'],
                    self.weights_dict['conv']['bias'],
                    self.weights_dict['objectness_logits']['weight'],
                    self.weights_dict['objectness_logits']['bias'],
                    self.weights_dict['anchor_deltas']['weight'],
                    self.weights_dict['anchor_deltas']['bias']]

  def get_weights(self):
    return self.weights
