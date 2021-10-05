from official.vision.beta.tasks import image_classification as imc
from official.vision.beta.projects.swin.configs import swin_classification as dcfg


import matplotlib.pyplot as plt

import tensorflow as tf



# prep_gpu()

def test_classification_input():
  with tf.device('/CPU:0'):
    config = dcfg.swin_classification().task
    task = imc.ImageClassificationTask(config)

    config.train_data.global_batch_size = 1
    config.validation_data.global_batch_size = 1
    config.train_data.dtype = 'float32'
    config.validation_data.dtype = 'float32'
    config.train_data.tfds_name = 'imagenet2012'
    config.validation_data.tfds_name = 'imagenet2012'
    config.train_data.tfds_split = 'train'
    config.validation_data.tfds_split = 'validation'
    config.train_data.tfds_data_dir = '/media/vbanna/DATA_SHARE/CV/datasets/tensorflow'
    config.validation_data.tfds_data_dir = '/media/vbanna/DATA_SHARE/CV/datasets/tensorflow'

    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data


def test_classification_pipeline():
  dataset, dsp = test_classification_input()
  for l, (i, j) in enumerate(dataset):
    plt.imshow(i[0].numpy())
    plt.show()
    if l > 30:
      break
  return


if __name__ == '__main__':
  # time_pipeline(num=100)
  test_classification_pipeline()
