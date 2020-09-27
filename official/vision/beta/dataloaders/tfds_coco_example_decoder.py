import tensorflow_datasets as tfds 
import tensorflow as tf
from official.vision.beta.dataloaders import decoder

import matplotlib.pyplot as plt
import cv2


class MSCOCODecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""
  def __init__(self,
               include_mask=False,
               regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id

  def decode(self, sample):
    """Decode the serialized example"""
    decoded_tensors = {
        'source_id': sample['image/id'],
        'image': sample['image'],
        'height': tf.shape(sample['image'])[0],
        'width':  tf.shape(sample['image'])[1],
        'groundtruth_classes': sample['objects']['label'],
        'groundtruth_is_crowd': sample['objects']['is_crowd'],
        'groundtruth_area': sample['objects']['area'],
        'groundtruth_boxes': sample['objects']['bbox'],
    }
    return decoded_tensors
coco, info = tfds.load('coco', split = 'train', with_info= True)

decoder = MSCOCODecoder()
dataset = coco.map(decoder.decode)

for i in dataset.take(1):
    print(i)
    cv2.imshow('sp', cv2.cvtColor(i['image'].numpy(), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

