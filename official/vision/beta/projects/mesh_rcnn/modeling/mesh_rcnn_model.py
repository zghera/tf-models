"""Mesh R-CNN models."""
from statistics import mode
import tensorflow as tf

from typing import Optional, List, Mapping
from official.vision.ops import anchor
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify

class MeshRCNNModel(tf.keras.Model):
    """Mesh R-CNN Model definition"""
    def __init__(self, 
                 backbone: tf.keras.Model,
                 decoder: tf.keras.Model,
                 rpn_head: tf.keras.layers.Layer,
                 roi_generator: tf.keras.layers.Layer,
                 roi_aligner: tf.keras.layers.Layer,
                 voxel_head: Optional[tf.keras.layers.Layer] = None,
                 mesh_head: Optional[tf.keras.layers.Layer] = None,
                 min_level: Optional[int] = None,
                 max_level: Optional[int] = None,
                 num_scales: Optional[int] = None,
                 aspect_ratios: Optional[List[float]] = None,
                 anchor_size: Optional[float] = None,
                 **kwargs):

        """Initializes the Mesh R-CNN model.
        Args:
        backbone: `tf.keras.Model`, the backbone network.
        decoder: `tf.keras.Model`, the decoder network.
        rpn_head: the RPN head.
        roi_generator: the ROI generator.
        roi_aligner: the ROI aligner.
        voxel_head: the voxel head
        mesh_head: the mesh head
        min_level: Minimum level in output feature maps.
        max_level: Maximum level in output feature maps.
        num_scales: A number representing intermediate scales added on each level.
            For instances, num_scales=2 adds one additional intermediate anchor
            scales [2^0, 2^0.5] on each level.
        aspect_ratios: A list representing the aspect raito anchors added on each
            level. The number indicates the ratio of width to height. For instances,
            aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each scale level.
        anchor_size: A number representing the scale of size of the base anchor to
            the feature stride 2^level.
        """
        super(MeshRCNNModel, self).__init__(**kwargs)
        self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'rpn_head': rpn_head,
        'roi_generator': roi_generator,
        'roi_aligner': roi_aligner,
        'min_level': min_level,
        'max_level': max_level,
        'num_scales': num_scales,
        'aspect_ratios': aspect_ratios,
        'anchor_size': anchor_size,
        }

        self.backbone = backbone
        self.decoder = decoder
        self.rpn_head = rpn_head
        self.roi_generator = roi_generator
        self.roi_aligner = roi_aligner
        self.voxel_head = voxel_head
        self.mesh_head = mesh_head
        self._include_mesh = mesh_head and voxel_head is not None

    def call(self,
            images: tf.Tensor,
            image_shape: tf.Tensor,
            anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
            training: Optional[bool] = None
            ) -> Mapping[str, tf.Tensor]:
        
        model_outputs = {}

        # Feature extraction.
        (backbone_features,
        decoder_features) = self._get_backbone_and_decoder_features(images)

        # Region proposal network.
        rpn_scores, rpn_boxes = self.rpn_head(decoder_features)

        model_outputs.update({
        'backbone_features': backbone_features,
        'decoder_features': decoder_features,
        'rpn_boxes': rpn_boxes,
        'rpn_scores': rpn_scores
        })

        # Generate anchor boxes for this batch if not provided.
        if anchor_boxes is None:
            _, image_height, image_width, _ = images.get_shape().as_list()
            anchor_boxes = anchor.Anchor(
                min_level=self._config_dict['min_level'],
                max_level=self._config_dict['max_level'],
                num_scales=self._config_dict['num_scales'],
                aspect_ratios=self._config_dict['aspect_ratios'],
                anchor_size=self._config_dict['anchor_size'],
                image_size=(image_height, image_width)).multilevel_boxes
            for l in anchor_boxes:
                anchor_boxes[l] = tf.tile(
                        tf.expand_dims(anchor_boxes[l], axis=0),
                        [tf.shape(images)[0], 1, 1, 1])

        # Generate RoIs.
        current_rois, _ = self.roi_generator(rpn_boxes, rpn_scores, anchor_boxes,
                                            image_shape, training)

        # Get roi features.
        roi_features = self.roi_aligner(model_outputs['decoder_features'], current_rois)
        model_outputs.update({'feature_map': roi_features})

        # check if include mesh is false
        if not self._include_mesh:
            return model_outputs
        
        # get voxels 
        voxels = self.voxel_head(roi_features) 

        # get cubified mesh
        mesh = cubify(voxels=voxels, thresh=0.5)
        model_outputs.update(mesh)

        # mesh refinement stage
        model_outputs = self.mesh_head(inputs=model_outputs)

        return model_outputs

    def _get_backbone_and_decoder_features(self, images):

        backbone_features = self.backbone(images)
        if self.decoder:
            features = self.decoder(backbone_features)
        else:
            features = backbone_features
        return backbone_features, features