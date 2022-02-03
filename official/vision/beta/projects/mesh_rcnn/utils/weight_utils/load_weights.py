"""
This file contains functions used to load the Pytorch Mesh R-CNN checkpoint 
weights into the Tensorflow model. 
"""

import numpy as np
import tensorflow as tf
from torch import load

from official.vision.beta.modeling.layers.nn_blocks import BottleneckBlock
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_data import \
    BackboneConfigData


def pth_to_dict(pth_path):
  """ Converts a TF checkpoint into a nested dictionary of weights.
  Args:
  pth_path: String, indicating filepath of the Pytorch checkpoint
  Returns:
  Dictionary where the checkpoint weights are stored
  Number of weights read
  """
  print("\nConverting model checkpoint from {} to weights dictionary\n".format(pth_path))
  pth_dict = load(pth_path)
  weights_dict = {}
  n_read = 0

  for key in pth_dict["model"]:
    keys = key.split('.')
    curr_dict = weights_dict
    for newKey in keys[:-1]:
      if newKey not in curr_dict:
        curr_dict[newKey] = {}
      curr_dict = curr_dict[newKey]
  
    np_tensor = pth_dict["model"][key].cpu().numpy()
    # transpose from Pytorch to use in TF
    if len(np_tensor.shape) == 4:
      np_tensor = np.transpose(np_tensor, [2, 3, 1, 0])
    
    tf_tensor = tf.convert_to_tensor(np_tensor)
    curr_dict[keys[-1]] = tf_tensor

    n_read += np.prod(np_tensor.shape)
  
  print("Successfully read {} checkpoint weights\n".format(n_read))
  return weights_dict, n_read

def get_backbone_layer_cfgs(weights_dict, backbone_name):
  """ Fetches the config classes for the backbone.

  This function generates a list of config classes corresponding to
  each building block in the backbone.

  Args:
    weights_dict: Dictionary that stores the backbone model weights
    backbone_name: String, indicating the desired backbone configuration
  Returns:
    A list containing the config classe of the backbone building block
  """

  print("Fetching backbone config classes for {}\n".format(backbone_name))
  cfgs = BackboneConfigData(weights_dict=weights_dict).get_cfg_list(backbone_name)
  return cfgs

def load_weights_backbone(backbone, weights_dict, backbone_name):
  """ Loads the weights defined in the weights_dict into the backbone.

  This function loads the backbone weights by first fetching the necesary
  config classes for the backbone, then loads them in one by one for 
  each layer that has weights associated with it.

  Args:
    backbone: keras.Model backbone
    weights_dict: Dictionary that stores the backbone model weights
    backbone_name: String, indicating the desired backbone configuration
  Returns:
    Number of weights loaded in 
  """

  print("Loading backbone weights\n")

  cfgs = get_backbone_layer_cfgs(weights_dict, backbone_name)
  n_weights_total = 0
  loaded_layers = 0

  i = 0
  for layer in backbone.layers:
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization, BottleneckBlock)):
      n_weights = cfgs[i].load_weights(layer)
      print("Weights loaded for {}: {}".format(layer.name, n_weights))
      n_weights_total += n_weights
      loaded_layers += 1
      i += 1
              
  print("{} Weights have been loaded for {} / {} layers\n".format(n_weights_total, loaded_layers, len(backbone.layers)))
  return n_weights_total