import json
import logging
import os

import tensorflow as tf
from absl import app  # pylint:disable=unused-import
from absl import flags

from official.vision.beta.data import tfrecord_lib
from official.vision.beta.data.tfrecord_lib import convert_to_feature

flags.DEFINE_multi_string('shapenet_dir', '', 'Directory containing '
                                                                                   'ShapeNet.')
flags.DEFINE_string('output_file_prefix', '', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def parse_obj_file(file):
    """
    Parses relevant data out of a .obj file. This contains all of the model information.
    Args:
        file: file path to .obj file
    Return:
        vertices: vertices of object
        faces: faces of object
    """
    vertices = []
    faces = []

    obj_file = open(file, 'r')
    lines = obj_file.readlines()

    for line in lines:
        lineID = line[0:2]

        if lineID == "v ":
            vertex = line[2:].split(" ")

            for i, v in enumerate(vertex):
                vertex[i] = float(v)

            vertices.append(vertex)

        if lineID == "f ":

            face = line[2:].split(" ")

            for i, f in enumerate(face):
                face[i] = [int(x) - 1 for x in f.split("/")]

            faces.append(face)

    return vertices, faces


def create_tf_example(image):
    model_id = image["model_id"]
    label = image["label"]

    temp_file_dir = os.join(image["shapenet_dir"], image["synset_id"])
    model_vertices, model_faces = parse_obj_file(os.join(temp_file_dir, image["model_id"]))

    feature_dict = {"model_id": convert_to_feature(model_id),
                    "label": convert_to_feature(label),
                    "vertices": convert_to_feature(model_vertices),
                    "faces": convert_to_feature(model_faces)}

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))

    return example, 0


def generate_annotations(images, shapenet_dir):
    for image in images:
        yield {"shapenet_dir": shapenet_dir,
               "label": image["label"],
               "model_id": image["model_id"],
               "synset_id": image["synset_id"]}


def _create_tf_record_from_shapenet_dir(shapenet_dir,
                                        output_path,
                                        num_shards):
    """Loads Shapenet json files and converts to tf.Record format.
    Args:
      images_info_file: shapenet_dir download directory
      output_path: Path to output tf.Record file.
      num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)

    # create synset ID to label mapping dictionary
    with open('C:/Users/Ethan/PycharmProjects/tf-models/official/vision/beta/projects/mesh_rcnn/data'
              '/shapenet_synset_dict.json', "r") as dict_file:
        synset_dict = json.load(dict_file)

    # images list
    images = []

    for _, synset_directories, _ in os.walk(shapenet_dir[0]):
        for synset_directory in synset_directories:
            for _, object_directories, _ in os.walk(os.path.join(shapenet_dir[0], synset_directory)):
                for object_directory in object_directories:
                    image = {"model_id": object_directory,
                             "label": synset_dict[synset_directory],
                             "shapenet_dir": shapenet_dir,
                             "synset_id": synset_directory}
                    images.append(image)

    shapenet_annotations_iter = generate_annotations(
        images=images, shapenet_dir=shapenet_dir)

    num_skipped = tfrecord_lib.write_tf_record_dataset(
        output_path, shapenet_annotations_iter, create_tf_example, num_shards)

    logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
    assert FLAGS.shapenet_dir, '`shapenet_dir` missing.'

    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)

    _create_tf_record_from_shapenet_dir('shapenet_dir', 'tmp', 32)


if __name__ == '__main__':
    app.run(main)
