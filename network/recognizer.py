import cv2
import os

import glob
from config import *
import numpy as np
import network.utils as utils

FLAGS = utils.FLAGS


"""
singleton for loading model in memory
"""
class ModelLoader:
    model = None
    classes = []
    __instance = None
    graph = None
    session = None
    decode_maps = None
    input = None
    output = None
    frozen_graph_name = ""

    @staticmethod
    def get_instance():
        if not ModelLoader.__instance:
            ModelLoader()
        return ModelLoader.__instance

    def __init__(self):
        self.decode_maps = self.build_decode_maps()
        self.frozen_graph_name = os.path.join(pb_path, "frozen.pb")
        self.graph = self.load_graph()
        session_config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=session_config, graph=self.graph)
        self.input = self.graph.get_tensor_by_name("prefix/{}:0".format(input_tensor_name))
        self.output = self.graph.get_tensor_by_name("prefix/{}:0".format(output_tensor_name))
        ModelLoader.__instance = self

    def build_decode_maps(self):
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        decode_maps = {}
        for i, char in enumerate(charset, 1):
            decode_maps[i] = char

        SPACE_INDEX = 0
        SPACE_TOKEN = ''
        decode_maps[SPACE_INDEX] = SPACE_TOKEN

        return decode_maps

    def load_graph(self):
        with tf.gfile.GFile(self.frozen_graph_name, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph


def recognize(img_path: str) -> list:
    """
    Read images to np array, loads model, makes recognize withoun pre-segmentation
    :param img_path: path to folder with images
    :return:
    list with recognized characters
    """
    img_list = []
    for filename in glob.iglob("{}{}".format(img_path, "/*.jpeg"), recursive=True):
        im = cv2.imread(filename, 1).astype(np.float32) / 255.
        im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
        im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        img_list.append(im)
    try:
        model = ModelLoader.get_instance()

    except Exception as e:
        print(e)
        return []
    decoded_expression = []

    x = model.input
    y = model.output
    with tf.device('/gpu:0'):
        for img in img_list:
            y_out = model.session.run(y, feed_dict={
                x: np.asanyarray([img] * FLAGS.batch_size)
            })

            expression = ''
            for letter in list(y_out[0]):
                if letter == -1:
                    expression += ''
                else:
                    expression += model.decode_maps[letter]

            decoded_expression.append(expression)
        return decoded_expression
