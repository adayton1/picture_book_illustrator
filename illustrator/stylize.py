# Adapted from: https://github.com/hzy46/fast-neural-style-tensorflow
from __future__ import print_function
import random
import os

import tensorflow as tf

import utils
utils.extend_syspath([
 'deps/fast-neural-style-tensorflow',
])  # yapf: disable=
from preprocessing import preprocessing_factory as preprocessing
import reader
import model
import time


class Stylizer(object):
    def __init__(self, model_file_path=None):
        self.image_preprocessing_fn, _ = preprocessing.get_preprocessing(
            'vgg_16', is_training=False)

        self.model_file_path = model_file_path
        if model_file_path is None:
            model_dir = os.path.join(
                os.path.dirname(__file__), '../deps/fast-neural-style-models')
            model_files = os.listdir(model_dir)
            self.model_file_path = os.path.join(model_dir,
                                                random.choice(model_files))

        self.model_path = os.path.abspath(self.model_file_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.sess.close()
        pass

    def stylize_image(self, image_path, image_size=None):
        print('Stylizing image...')
        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                if image_size:
                    height, width = image_size
                else:
                    with open(image_path, 'rb') as img:
                        with tf.Session().as_default() as sess:
                            if image_path.lower().endswith('png'):
                                image = sess.run(
                                    tf.image.decode_png(img.read()))
                            else:
                                image = sess.run(
                                    tf.image.decode_jpeg(img.read()))
                            height = image.shape[0]
                            width = image.shape[1]

                    image = reader.get_image(image_path, height, width,
                                             self.image_preprocessing_fn)

                    # Add batch dimension
                    image = tf.expand_dims(image, 0)

                    generated = model.net(image, training=False)
                    generated = tf.cast(generated, tf.uint8)

                    # Remove batch dimension
                    generated = tf.squeeze(generated, [0])

                    # Restore model variables.
                    saver = tf.train.Saver(
                        tf.global_variables(),
                        write_version=tf.train.SaverDef.V1)
                    sess.run([
                        tf.global_variables_initializer(),
                        tf.local_variables_initializer()
                    ])
                    saver.restore(sess, self.model_path)

                return sess.run(tf.image.encode_jpeg(generated))


if __name__ == '__main__':
    import sys
    import numpy as np
    import cv2

    with Stylizer() as stylizer:
        image = stylizer.stylize_image(sys.argv[1])
        with open('out.jpg', 'wb') as f:
            f.write(image)
