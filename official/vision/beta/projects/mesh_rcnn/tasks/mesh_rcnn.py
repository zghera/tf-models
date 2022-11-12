"""MeshRCNN task definition."""

from official.core import base_task
from official.core import task_factory
from official.vision.beta.projects.mesh_rcnn.configs import mesh_rcnn as exp_cfg
from official.vision.beta.projects.mesh_rcnn.modeling import factory
from official.vision.beta.projects.mesh_rcnn.losses import mesh_losses

@task_factory.register_task_cls(exp_cfg.MeshRCNNTask)

class MeshRCNNTask(base_task.Task):
    def __init__(self, params, logging_dir: Optional[str] = None):
        super().__init__(params, logging_dir)
        return

    def build_model(self):
        """Build Mesh R-CNN model."""
        model, losses = factory.build_mesh_rcnn(
            input_specs, model_base_cfg, l2_regularizer)
        return

    def build_inputs(self, params, input_context=None):
        """Build input dataset."""
        return
    
    def build_metrics(self, training=True):
        """Build detection metrics."""
        return

    def build_losses(self):
        """Build Mesh R-CNN losses."""
        loss_config = exp_cfg.MeshLosses
        mesh_loss = mesh_losses.MeshLoss(loss_config.voxel_weight,
                         loss_config.chamfer_weight,
                         loss_config.normal_weight,
                         loss_config.edge_weight,
                         loss_config.true_num_samples,
                         loss_config.pred_num_samples)

        total_loss, voxel_loss, chamfer_loss_, normal_loss_, edge_loss_ = \
        mesh_loss(
            voxels_true,
            voxels_pred,
            meshes_true,
            meshes_pred,
            edges_pred,
            edges_mask_pred,
        )

        losses = {
            'total_loss' : total_loss,
            'voxel_loss' : voxel_loss,
            'chamfer_loss' : chamfer_loss_,
            'normal_loss' : normal_loss_,
            'edge_loss' : edge_loss_,
        }
        return losses
        

    def initialize(self, model: tf.keras.Model):
        """Loading pretrained checkpoint."""
        return

    def train_step(self, inputs, model, optimizer, metrics=None):
        """Train Step.

        Forward step and backwards propagate the model.

        Args:
        inputs: a dictionary of input tensors.
        model: the model, forward pass definition.
        optimizer: the optimizer for this training step.
        metrics: a nested structure of metrics objects.

        Returns:
        A dictionary of logs.
        """
        return

    def validation_step(self, inputs, model, metrics=None):
        """Validatation step.

        Args:
        inputs: a dictionary of input tensors.
        model: the keras.Model.
        metrics: a nested structure of metrics objects.

        Returns:
        A dictionary of logs.
        """
        return
    
    def aggregate_logs(self, state=None, step_outputs=None):
        """Get Metric Results."""
        return

    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Reduce logs and remove unneeded items. Update with COCO results."""
        return

    def create_optimizer(self,
                       optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
        """Creates an TF optimizer from configurations.

        Args:
        optimizer_config: the parameters of the Optimization settings.
        runtime_config: the parameters of the runtime.

        Returns:
        A tf.optimizers.Optimizer object.
        """
        return





    




