import tensorflow as tf

from official.vision.beta.modeling.backbones import resnet
from official.vision.beta.modeling.decoders import fpn
from official.vision.beta.modeling.heads import dense_prediction_heads
from official.vision.beta.projects.mesh_rcnn.configs import \
    mesh_rcnn as mesh_rcnn_config
from official.vision.beta.projects.mesh_rcnn.modeling import factory
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_classes import \
    voxelHeadCFG
from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.load_weights import (
    load_weights_backbone, pth_to_dict)

PTH_PATH = ""

def test_load_voxel_head():
  weights_dict, n_read = pth_to_dict(PTH_PATH)

  input = tf.keras.layers.Input(shape=(12, 12, 256))
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
                                  activity_regularizer=None)(input)

  cfg = voxelHeadCFG(weights_dict = weights_dict['roi_heads']['voxel_head'])
  
  model = tf.keras.Model(inputs=[input], outputs=[head])
  model.summary()
  for layer in model.layers:
    if layer.name == "voxel_head":
      cfg.load_weights(layer)


def test_load_backbone():
  weights_dict, n_read = pth_to_dict(PTH_PATH)
  backbone = resnet.ResNet(model_id=50)
  x = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  backbone.build(x.shape)
  
  n_weights = load_weights_backbone(backbone, weights_dict['backbone']['bottom_up'], "resnet50")
  print("TOTAL WEIGHTS LOADED FOR BACKBONE: ", end="")
  tf.print(n_weights)


def test_load_decoder():
  min_level = 2
  max_level = 5

  backbone = resnet.ResNet(model_id=50)
  decoder = fpn.FPN(
    input_specs=backbone.output_specs,
    min_level=min_level,
    max_level=max_level,
    use_separable_conv=False)

  for layer in decoder.layers:  
    print(layer.name)
    names = [weight.name for weight in layer.weights]
    for name, weight in zip(names, layer.get_weights()):
      print(name, weight.shape)
    

def test_load_rpn_head():
  min_level = 2
  max_level = 5
  num_scales = 5
  aspect_ratios = [0.5, 1.0, 2.0]
  num_anchors_per_location = num_scales * len(aspect_ratios)
  rpn_head = dense_prediction_heads.RPNHead(
            min_level=min_level,
            max_level=max_level,
            num_anchors_per_location=num_anchors_per_location,
            num_convs=1)
  rpn_head.summary()
    

if __name__ == "__main__":
    test_load_rpn_head()
