"""Tests for mesh_rcnn_model.py."""

import os
# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from typing import Optional

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.projects.mesh_rcnn.modeling import mesh_rcnn_model

from official.vision.beta.projects.mesh_rcnn.modeling.heads import mesh_head
from official.vision.beta.projects.mesh_rcnn.modeling.heads import voxel_head
from official.vision.beta.projects.mesh_rcnn.modeling.heads import z_head

from official.vision.ops import anchor

from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import fpn
from official.vision.modeling.heads import dense_prediction_heads
from official.vision.modeling.layers import roi_aligner
from official.vision.modeling.layers import roi_generator

from official.vision.beta.projects.mesh_rcnn.modeling.layers import experimental_roi_align


class MeshRCNNModelTest(parameterized.TestCase, tf.test.TestCase):

    @combinations.generate(
        combinations.combine(
            include_mesh=[True, False], # includes voxel and mesh head
            use_separable_conv=[True, False],
            build_anchor_boxes=[True, False],
            is_training=[True, False]))
    def test_build_model(self, include_mesh, use_separable_conv, build_anchor_boxes, is_training):
        
        voxel_depth=24
        conv_dim=256
        num_conv=2
        use_group_norm=False
        predict_classes=False # should be False
        bilinearly_upscale_input= not predict_classes 
        class_based_voxel=True 
        num_classes = 1
        
        min_level = 3
        max_level = 7
        num_scales = 3
        aspect_ratios = [1.0]
        anchor_size = 3
        num_anchors_per_location = num_scales * len(aspect_ratios)
        image_size = 256
        images = np.random.rand(2, image_size, image_size, 3)
        image_shape = np.array([[image_size, image_size], [image_size, image_size]])
        
        if build_anchor_boxes:
            anchor_boxes = anchor.Anchor(
                min_level=min_level,
                max_level=max_level,
                num_scales=num_scales,
                aspect_ratios=aspect_ratios,
                anchor_size=anchor_size,
                image_size=(image_size, image_size)).multilevel_boxes
            for l in anchor_boxes:
                anchor_boxes[l] = tf.tile(
                    tf.expand_dims(anchor_boxes[l], axis=0), [2, 1, 1, 1])
        else:
            anchor_boxes = None
            
        backbone = resnet.ResNet(model_id=50)
        decoder = fpn.FPN(
            input_specs=backbone.output_specs,
            min_level=min_level,
            max_level=max_level,
            use_separable_conv=use_separable_conv)
        rpn_head = dense_prediction_heads.RPNHead(
            min_level=min_level,
            max_level=max_level,
            num_anchors_per_location=num_anchors_per_location,
            num_convs=1)
        roi_generator_obj = roi_generator.MultilevelROIGenerator()
        roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=12)
    
        if include_mesh: #if not necessary to test without voxel/mesh head remove include_mesh in model
            voxel_head_obj = voxel_head.VoxelHead(
                voxel_depth=voxel_depth,
                conv_dim=conv_dim,
                num_conv=num_conv,
                use_group_norm=use_group_norm,
                predict_classes=predict_classes,
                bilinearly_upscale_input=bilinearly_upscale_input,
                class_based_voxel=class_based_voxel,
                num_classes=num_classes
            )
            mesh_head_obj = mesh_head.MeshHead()
        else:
            voxel_head_obj = None
            mesh_head_obj = None
        model = mesh_rcnn_model.MeshRCNNModel( 
            backbone,
            decoder,
            rpn_head,
            roi_generator_obj,
            roi_aligner_obj,
            voxel_head_obj,
            mesh_head_obj,
            
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_size)

        # Results
        results = model(
            images,
            image_shape,
            anchor_boxes,
            training=is_training)
        
        self.assertIn('backbone_features', results)
        self.assertIn('decoder_features', results)
        self.assertIn('rpn_boxes', results)
        self.assertIn('rpn_scores', results)
        self.assertIn('feature_map', results)
        if include_mesh:
            self.assertIn('verts', results)
            self.assertIn('faces', results)
            self.assertIn('verts_mask', results)
            self.assertIn('faces_mask', results)
                        
    """  
    @combinations.generate(
        combinations.combine(
            strategy=[
                strategy_combinations.cloud_tpu_strategy,
                strategy_combinations.one_device_strategy_gpu,
                ],
            include_mesh=[True, False],
            use_separable_conv=[True, False],
            build_anchor_boxes=[True, False],
            is_training=[True, False],
            predict_classes=[True, False],
            use_group_norm=[True,False],
            class_based_voxel=[True,False]
        ))

    def test_forward(self, strategy, include_mesh, use_separable_conv, build_anchor_boxes, is_training, 
                     predict_classes, use_group_norm, class_based_voxel):
        
        voxel_depth=24
        conv_dim=256
        num_conv=2
        use_group_norm=use_group_norm
        predict_classes=predict_classes  # True for Pix3D
        bilinearly_upscale_input= not predict_classes 
        class_based_voxel=class_based_voxel  #If `predict_classes` is True but `class_based_voxel` is False, we will only predict 1 class. 
        num_classes = 1
        
        min_level = 3
        max_level = 4
        num_scales = 3
        aspect_ratios = [1.0]
        anchor_size = 3
        image_size = (256, 256)
        images = np.random.rand(2, image_size[0], image_size[1], 3)
        image_shape = np.array([[224, 100], [100, 224]])
        with strategy.scope():
            if build_anchor_boxes:
                anchor_boxes = anchor.Anchor(
                    min_level=min_level,
                    max_level=max_level,
                    num_scales=num_scales,
                    aspect_ratios=aspect_ratios,
                    anchor_size=anchor_size,
                    image_size=image_size).multilevel_boxes
            else:
                anchor_boxes = None
            num_anchors_per_location = len(aspect_ratios) * num_scales

            input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, 3])
            backbone = resnet.ResNet(model_id=50, input_specs=input_specs)
            decoder = fpn.FPN(
                min_level=min_level,
                max_level=max_level,
                use_separable_conv=use_separable_conv,
                input_specs=backbone.output_specs)
            rpn_head = dense_prediction_heads.RPNHead(
                min_level=min_level,
                max_level=max_level,
                num_anchors_per_location=num_anchors_per_location)
            roi_generator_obj = roi_generator.MultilevelROIGenerator()
            roi_aligner_obj = roi_aligner.MultilevelROIAligner()
            if include_mesh:
                voxel_head_obj = voxel_head.VoxelHead(
                    voxel_depth=voxel_depth,
                    conv_dim=conv_dim,
                    num_conv=num_conv,
                    use_group_norm=use_group_norm,
                    predict_classes=predict_classes,
                    bilinearly_upscale_input=bilinearly_upscale_input,
                    class_based_voxel=class_based_voxel,
                    num_classes=num_classes
                )
                mesh_head_obj = mesh_head.MeshHead()
            else:
                voxel_head_obj = None
                mesh_head_obj = None
            model = mesh_rcnn_model.MeshRCNNModel(
                backbone,
                decoder,
                rpn_head,
                roi_generator_obj,
                roi_aligner_obj,
                voxel_head_obj,
                mesh_head_obj,
                
                min_level=min_level,
                max_level=max_level,
                num_scales=num_scales,
                aspect_ratios=aspect_ratios,
                anchor_size=anchor_size)

            results = model(
                images,
                image_shape,
                anchor_boxes,
                training=is_training)

        self.assertIn('backbone_features', results)
        self.assertIn('decoder_features', results)
        self.assertIn('rpn_boxes', results)
        self.assertIn('rpn_scores', results)
        self.assertIn('feature_map', results)
        if include_mesh:
            self.assertIn('verts', results)
            self.assertIn('faces', results)
            self.assertIn('verts_mask', results)
            self.assertIn('faces_mask', results)
        """

if __name__ == '__main__':
  tf.test.main()