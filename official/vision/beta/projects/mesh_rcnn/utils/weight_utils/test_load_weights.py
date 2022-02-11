'''Test for weight loading'''

import tensorflow as tf

from official.vision.beta.modeling.backbones import resnet
from official.vision.beta.projects.mesh_rcnn.configs import \
    mesh_rcnn as mesh_rcnn_config
from official.vision.beta.projects.mesh_rcnn.modeling import factory
from official.vision.beta.projects.mesh_rcnn.modeling.decoders import fpn
from official.vision.beta.projects.mesh_rcnn.modeling.heads import rpn
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_classes import (
    RPNHeadCFG, VoxelHeadCFG)
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.load_weights import (
    load_weights_backbone, load_weights_decoder, pth_to_dict)

PTH_PATH = "C:\\Users\\allen\\Downloads\\meshrcnn_R50.pth"


def test_load_voxel_head():
  "Test for loading voxel head weights."

  weights_dict, _ = pth_to_dict(PTH_PATH)

  input_layer = tf.keras.layers.Input(shape=(12, 12, 256))

  head_cfg = mesh_rcnn_config.VoxelHead(voxel_depth=24,
                                        conv_dim=256,
                                        num_conv=4,
                                        use_group_norm=False,
                                        predict_classes=False,
                                        bilinearly_upscale_input=True,
                                        class_based_voxel=False,
                                        num_classes=0)
  head = factory.build_voxel_head(head_cfg,
                                  kernel_regularizer=None,
                                  bias_regularizer=None,
                                  activity_regularizer=None)(input_layer)

  weight_cfg = VoxelHeadCFG(weights_dict =
                                weights_dict["roi_heads"]["voxel_head"])

  model = tf.keras.Model(inputs=[input_layer], outputs=[head])

  for layer in model.layers:
    if layer.name == "voxel_head":
      n_weights = weight_cfg.load_weights(layer)

  print("TOTAL WEIGHTS LOADED FOR VOXEL HEAD: ", end="")
  tf.print(n_weights)

def test_load_backbone():
  "Test for loading resnet50 backbone weights."

  weights_dict, _ = pth_to_dict(PTH_PATH)
  backbone = resnet.ResNet(model_id=50)

  x = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  backbone.build(x.shape)

  n_weights = load_weights_backbone(backbone,
                                    weights_dict["backbone"]["bottom_up"],
                                    "resnet50")
  print("TOTAL WEIGHTS LOADED FOR BACKBONE: ", end="")
  tf.print(n_weights)


def test_load_decoder():
  "Test for loading FPN decoder weights."

  weights_dict, _ = pth_to_dict(PTH_PATH)
  min_level = 2
  max_level = 5

  backbone = resnet.ResNet(model_id=50)
  decoder = fpn.FPN(
    input_specs=backbone.output_specs,
    min_level=min_level,
    max_level=max_level,
    use_separable_conv=False)

  n_weights = load_weights_decoder(decoder,
                                   weights_dict["backbone"],
                                   "fpn")
  print("TOTAL WEIGHTS LOADED FOR DECODER: ", end="")
  tf.print(n_weights)

  print(decoder.output_specs)


def test_load_rpn_head():
  "Test for loading RPN head weights."

  weights_dict, _ = pth_to_dict(PTH_PATH)
  min_level = 2
  max_level = 5

  features = {
      "2": tf.keras.layers.Input(shape=(256, 256, 256)),
      "3": tf.keras.layers.Input(shape=(128, 128, 256)),
      "4": tf.keras.layers.Input(shape=(64, 64, 256)),
      "5": tf.keras.layers.Input(shape=(32, 32, 256))
  }

  rpn_head = rpn.RPNHead(
            min_level=min_level,
            max_level=max_level,
            num_anchors_per_location=3,
            num_convs=1)(features)

  weight_cfg = RPNHeadCFG(weights_dict =
                            weights_dict["proposal_generator"]["rpn_head"])

  model = tf.keras.Model(inputs=[features], outputs=[rpn_head])

  for layer in model.layers:
    if layer.name == "rpn_head":
      n_weights = weight_cfg.load_weights(layer)

  print("TOTAL WEIGHTS LOADED FOR VOXEL HEAD: ", end="")
  tf.print(n_weights)

if __name__ == "__main__":
  test_load_rpn_head()
