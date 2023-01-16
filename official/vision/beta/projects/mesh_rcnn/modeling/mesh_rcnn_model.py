"""Mesh RCNN models."""
import tensorflow as tf

class MeshRCNNModel(tf.keras.Model):
    """Mesh RCNN Model definition"""
    def __init__(self, 
                 backbone: tf.keras.Model,
                 decoder: tf.keras.Model,
                 **kwargs):

        """Initializes the Mesh R-CNN model.
        Args:
        backbone: `tf.keras.Model`, the backbone network.
        decoder: `tf.keras.Model`, the decoder network.
        """
        super(MeshRCNNModel, self).__init__(**kwargs)
        
        
        self.backbone = backbone
        self.decoder = decoder



    def _get_backbone_and_decoder_features(self, images):

        backbone_features = self.backbone(images)
        if self.decoder:
            features = self.decoder(backbone_features)
        else:
            features = backbone_features
        return backbone_features, features
