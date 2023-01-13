import collections

import tensorflow as tf

from official.modeling import hyperparams
from official.projects.yolo.modeling.backbones.darknet import LayerBuilder
from official.projects.yolo.modeling.layers import nn_blocks
from official.vision.modeling.backbones import factory


class ELANBlockConfig:
  """Class to store configs of layers in ELAN to make code more readable."""

  def __init__(self, layer, stack, convs_per_split, total_split_convs, reps,
               filters, kernel_size, strides, padding, activation, downsample,
               downsample_filter_scale, route, output_name, is_output):
    self.layer = layer
    self.stack = stack
    self.repetitions = reps
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.activation = activation
    self.route = route
    self.output_name = output_name
    self.is_output = is_output

    self.convs_per_split = convs_per_split
    self.total_split_convs = total_split_convs
    self.downsample = downsample
    self.downsample_filter_scale = downsample_filter_scale

def build_elan_block_specs(config):
  specs = []
  for layer in config:
    specs.append(ELANBlockConfig(*layer))
  return specs

ELAN_LISTNAMES = [
    'default_layer_name', 'level_type', 'convs_per_split', 'total_split_convs',
    'number_of_layers_in_level', 'filters', 'kernal_size', 'strides',
    'padding', 'default_activation', 'downsample', 'downsample_filter_scale',
    'route', 'level/output_name', 'is_output'
]
ELAN_REGULAR = {
    'list_names':
    ELAN_LISTNAMES,
    'splits': {
        'backbone_split': -1
    },
    'backbone': [
        [
            'ConvBN', None, None, None, 1, 32, 3, 1, 'same', 'swish', None,
            None, -1, 1, False
        ],
        [
            'ConvBN', None, None, None, 1, 64, 3, 2, 'same', 'swish', None,
            None, -1, 1, True
        ],
        [
            'ConvBN', None, None, None, 1, 64, 3, 1, 'same', 'swish', None,
            None, -1, 2, False
        ],
        [
            'ConvBN', None, None, None, 1, 128, 3, 2, 'same', 'swish', None,
            None, -1, 2, True
        ],
        [
            'ELANBlock', None, 2, 4, 1, 256, None, None, None, 'swish',
            False, 4, -1, 3, False
        ],
        [
            'ELANBlock', None, 2, 4, 1, 512, None, None, None, 'swish', True,
            4, -1, 3, True
        ],
        [
            'ELANBlock', None, 2, 4, 1, 1024, None, None, None, 'swish',
            True, 4, -1, 4, True
        ],
        [
            'ELANBlock', None, 2, 4, 1, 1024, None, None, None, 'swish',
            True, 2, -1, 5, True
        ],
    ]
}

BACKBONES = {
  'elan_regular': ELAN_REGULAR,
}

class ELAN(tf.keras.Model):
  """The ELAN backbone architecture."""

  def __init__(
      self,
      model_id='elan_regular',
      input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      min_level=None,
      max_level=5,
      width_scale=1.0,
      depth_scale=1.0,
      use_reorg_input=False,
      csp_level_mod=(),
      activation=None,
      use_sync_bn=False,
      use_separable_conv=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      dilate=False,
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_regularizer=None,
      **kwargs):

    layer_specs, splits = ELAN.get_model_config(model_id)

    self._model_name = model_id
    self._splits = splits
    self._input_shape = input_specs
    self._registry = LayerBuilder()

    # default layer look up
    self._min_size = min_level
    self._max_size = max_level
    self._output_specs = None
    self._csp_level_mod = set(csp_level_mod)

    self._kernel_initializer = kernel_initializer
    self._bias_regularizer = bias_regularizer
    self._norm_momentum = norm_momentum
    self._norm_epislon = norm_epsilon
    self._use_sync_bn = use_sync_bn
    self._use_separable_conv = use_separable_conv
    self._activation = activation
    self._kernel_regularizer = kernel_regularizer
    self._dilate = dilate
    self._width_scale = width_scale
    self._depth_scale = depth_scale
    self._use_reorg_input = use_reorg_input

    self._default_dict = {
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epislon,
        'use_sync_bn': self._use_sync_bn,
        'activation': self._activation,
        'use_separable_conv': self._use_separable_conv,
        'dilation_rate': 1,
        'name': None
    }

    inputs = tf.keras.layers.Input(shape=self._input_shape.shape[1:])
    output = self._build_struct(layer_specs, inputs)
    super().__init__(inputs=inputs, outputs=output, name=self._model_name)

  @property
  def input_specs(self):
    return self._input_shape

  @property
  def output_specs(self):
    return self._output_specs

  @property
  def splits(self):
    return self._splits

  def _build_struct(self, net, inputs):
    if self._use_reorg_input:
      inputs = nn_blocks.Reorg()(inputs)
      net[0].filters = net[1].filters
      net[0].output_name = net[1].output_name
      del net[1]

    endpoints = collections.OrderedDict()
    stack_outputs = [inputs]
    for i, config in enumerate(net):
      if config.output_name > self._max_size:
        break

      config.filters = int(config.filters * self._width_scale)
      config.repetitions = int(config.repetitions * self._depth_scale)

      if config.layer == "ELANBlock":
        x, x_route = self._build_elan_block(stack_outputs[config.route],
                                            config,
                                            name=f'{config.layer}_{i}')
        stack_outputs.append(x_route)
      else:
        x = self._build_block(stack_outputs[config.route],
                              config,
                              name=f'{config.layer}_{i}')
        stack_outputs.append(x)

      if (config.is_output and self._min_size is None):
        endpoints[str(config.output_name)] = x
      elif (self._min_size is not None and config.output_name >= self._min_size
            and config.output_name <= self._max_size):
        endpoints[str(config.output_name)] = x

    self._output_specs = {
        l: endpoints[l].get_shape()
        for l in endpoints.keys()
    }

    return endpoints

  def _get_activation(self, activation):
    if self._activation is None:
      return activation
    return self._activation

  def _build_block(self, inputs, config, name):
    x = inputs
    i = 0
    self._default_dict['activation'] = self._get_activation(config.activation)
    while i < config.repetitions:
      self._default_dict['name'] = f'{name}_{i}'
      layer = self._registry(config, self._default_dict)
      x = layer(x)
      i += 1
    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None
    return x

  def _build_elan_block(self, inputs, config, name):
    self._default_dict['activation'] = self._get_activation(config.activation)
    self._default_dict['name'] = name

    x, x_route = nn_blocks.ELANBlock(
        filters=config.filters,
        total_split_convs=config.total_split_convs,
        convs_per_split=config.convs_per_split,
        downsample=config.downsample,
        downsample_filter_scale=config.downsample_filter_scale,
        **self._default_dict)(inputs)

    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None

    return x, x_route

  @staticmethod
  def get_model_config(name):
    name = name.lower()
    backbone = BACKBONES[name]['backbone']
    splits = BACKBONES[name]['splits']
    return build_elan_block_specs(backbone), splits

  @property
  def model_id(self):
    return self._model_name

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def get_config(self):
    layer_config = {
        'model_id': self._model_name,
        'min_level': self._min_size,
        'max_level': self._max_size,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epislon,
        'use_sync_bn': self._use_sync_bn,
        'activation': self._activation,
    }
    return layer_config


@factory.register_backbone_builder('ELAN')
def build_ELAN(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None
) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds ELAN."""

  backbone_config = backbone_config.get()
  model = ELAN(model_id=backbone_config.model_id,
               min_level=backbone_config.min_level,
               max_level=backbone_config.max_level,
               input_specs=input_specs,
               dilate=backbone_config.dilate,
               width_scale=backbone_config.width_scale,
               depth_scale=backbone_config.depth_scale,
               use_reorg_input=backbone_config.use_reorg_input,
               activation=norm_activation_config.activation,
               use_sync_bn=norm_activation_config.use_sync_bn,
               use_separable_conv=backbone_config.use_separable_conv,
               norm_momentum=norm_activation_config.norm_momentum,
               norm_epsilon=norm_activation_config.norm_epsilon,
               kernel_regularizer=l2_regularizer)
  return model
