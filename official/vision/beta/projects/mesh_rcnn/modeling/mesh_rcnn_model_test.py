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

from official.vision.beta.projects.mesh_rcnn.modeling.layers.nn_blocks import MeshRefinementStage
from official.vision.beta.projects.mesh_rcnn.ops import cubify
from official.vision.ops import anchor

from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import fpn
from official.vision.modeling.heads import dense_prediction_heads
from official.vision.modeling.heads import instance_heads
from official.vision.modeling.layers import detection_generator
from official.vision.modeling.layers import mask_sampler
from official.vision.modeling.layers import roi_aligner
from official.vision.modeling.layers import roi_generator
from official.vision.modeling.layers import roi_sampler


class MeshRCNNModelTest(parameterized.TestCase, tf.test.TestCase):
    
    @combinations.generate(
        combinations.combine(
            include_mesh=[True, False],
            use_separable_conv=[True, False],
            build_anchor_boxes=[True, False],
            is_training=[True, False]))
    def test_build_model(self, include_mesh, use_separable_conv,
                       build_anchor_boxes, is_training):
        num_classes = 3
        min_level = 3
        max_level = 7
        num_scales = 3
        aspect_ratios = [1.0]
        anchor_size = 3
        resnet_model_id = 50    
        num_anchors_per_location = num_scales * len(aspect_ratios)
        image_size = 384
        images = np.random.rand(2, image_size, image_size, 3)
        image_shape = np.array([[image_size, image_size], [image_size, image_size]])

        if build_anchor_boxes:
            anchor_boxes = anchor.Anchor(
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=3,
            image_size=(image_size, image_size)).multilevel_boxes
        for l in anchor_boxes:
            anchor_boxes[l] = tf.tile(
                tf.expand_dims(anchor_boxes[l], axis=0), [2, 1, 1, 1])
        else:
            anchor_boxes = None

        backbone = resnet.ResNet(model_id=resnet_model_id)
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
        detection_head = instance_heads.DetectionHead(num_classes=num_classes)
        roi_generator_obj = roi_generator.MultilevelROIGenerator()
        roi_aligner_obj = roi_aligner.MultilevelROIAligner()
    
        if include_mesh:
            voxel_head_obj = voxel_head.VoxelHead(
                num_classes=num_classes,
            )
            """
                Need to include the following in voxel head:
                'voxel_depth', 'conv_dim', 'num_conv', 'use_group_norm', 
                'predict_classes', 'bilinearly_upscale_input', 'class_based_voxel',  'num_classes'
            """
            mesh_head_obj = mesh_head.MeshHead()
        else:
            voxel_head_obj = None
            mesh_head_obj = None
        model = mesh_rcnn_model.MeshRCNNModel( 
            backbone,
            decoder,
            rpn_head,
            detection_head,
            roi_generator_obj,
            roi_aligner_obj,
            voxel_head_obj,
            mesh_head_obj,

            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_size)

        gt_boxes = np.array(
            [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
             [[100, 100, 150, 150], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
            dtype=np.float32)
        gt_classes = np.array([[2, 1, -1], [1, -1, -1]], dtype=np.int32)
        if include_mesh:
            gt_masks = np.ones((2, 3, 100, 100))
        else:
            gt_masks = None

        # Results will be checked in test_forward.
        _ = model(
            images,
            image_shape,
            anchor_boxes,
            gt_boxes,
            gt_classes,
            gt_masks,
            training=is_training)

    @combinations.generate(
        combinations.combine(
            strategy=[
                strategy_combinations.cloud_tpu_strategy,
                strategy_combinations.one_device_strategy_gpu,
            ],
            include_mesh=[True, False],
            build_anchor_boxes=[True, False],
            use_cascade_heads=[True, False],
            training=[True, False],
        ))
    
    def test_forward(self, strategy, include_mesh, build_anchor_boxes, training,
                   use_cascade_heads):
        num_classes = 3
        min_level = 3
        max_level = 4
        num_scales = 3
        aspect_ratios = [1.0]
        anchor_size = 3
        if use_cascade_heads:
            cascade_iou_thresholds = [0.6]
            class_agnostic_bbox_pred = True
            cascade_class_ensemble = True
        else:
            cascade_iou_thresholds = None
            class_agnostic_bbox_pred = False
            cascade_class_ensemble = False

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
            input_specs=backbone.output_specs)
        rpn_head = dense_prediction_heads.RPNHead(
            min_level=min_level,
            max_level=max_level,
            num_anchors_per_location=num_anchors_per_location)
        detection_head = instance_heads.DetectionHead(
            num_classes=num_classes,
            class_agnostic_bbox_pred=class_agnostic_bbox_pred)
        roi_generator_obj = roi_generator.MultilevelROIGenerator()

        roi_sampler_cascade = []
        roi_sampler_obj = roi_sampler.ROISampler()
        roi_sampler_cascade.append(roi_sampler_obj)
        if cascade_iou_thresholds:
            for iou in cascade_iou_thresholds:
                roi_sampler_obj = roi_sampler.ROISampler(
                    mix_gt_boxes=False,
                    foreground_iou_threshold=iou,
                    background_iou_high_threshold=iou,
                    background_iou_low_threshold=0.0,
                    skip_subsampling=True)
            roi_sampler_cascade.append(roi_sampler_obj)
        roi_aligner_obj = roi_aligner.MultilevelROIAligner()
        if include_mesh:
            voxel_head_obj = voxel_head.VoxelHead(
                num_classes=num_classes,
            )
            """
                Need to include the following in voxel head:
                'voxel_depth', 'conv_dim', 'num_conv', 'use_group_norm', 
                'predict_classes', 'bilinearly_upscale_input', 'class_based_voxel',  'num_classes'
            """
            mesh_head_obj = mesh_head.MeshHead()
        else:
            voxel_head_obj = None
            mesh_head_obj = None
        model = mesh_rcnn_model.MeshRCNNModel(
            backbone,
            decoder,
            rpn_head,
            detection_head,
            roi_generator_obj,
            roi_aligner_obj,
            voxel_head_obj,
            mesh_head_obj,

            class_agnostic_bbox_pred=class_agnostic_bbox_pred,
            cascade_class_ensemble=cascade_class_ensemble,
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_size)

        gt_boxes = np.array(
            [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
             [[100, 100, 150, 150], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
            dtype=np.float32)
        gt_classes = np.array([[2, 1, -1], [1, -1, -1]], dtype=np.int32)
        if include_mesh:
            gt_masks = np.ones((2, 3, 100, 100))
        else:
            gt_masks = None

        results = model(
            images,
            image_shape,
            anchor_boxes,
            gt_boxes,
            gt_classes,
            gt_masks,
            training=training)

        self.assertIn('rpn_boxes', results)
        self.assertIn('rpn_scores', results)
        if training:
            self.assertIn('class_targets', results)
            self.assertIn('box_targets', results)
            self.assertIn('class_outputs', results)
            self.assertIn('box_outputs', results)
            if include_mesh:
                self.assertIn('mask_outputs', results)
        else:
            self.assertIn('detection_boxes', results)
            self.assertIn('detection_scores', results)
            self.assertIn('detection_classes', results)
            self.assertIn('num_detections', results)
            if include_mesh:
                self.assertIn('detection_masks', results)

if __name__ == '__main__':
  tf.test.main()