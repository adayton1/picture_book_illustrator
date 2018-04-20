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
        print("Loading stylizer model...")
        self.image_preprocessing_fn, _ = preprocessing.get_preprocessing(
            'vgg_16', is_training=False)

        self.model_file_path = model_file_path
        if model_file_path is None:
            model_files = os.listdir(
                os.path.join(
                    os.path.dirname(__file__),
                    '../deps/fast-neural-style-models'))
            self.model_file_path = random.choice(model_files)

        self.model_path = os.path.abspath(self.model_file_path)
        with tf.Graph().as_default() as g:
            with tf.Session().as_default() as sess:
                self.sess = sess
                self.graph = g
                saver = tf.train.Saver(
                    tf.global_variables(), write_version=tf.train.SaverDef.V1)

                saver.restore(self.sess, self.model_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def stylize_images(self, image_paths, image_size=None):
        print('Stylizing image(s)...')
        height, width = image_size

        self.sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])

        outputs = []
        for image_path in image_paths:
            if not height or not width:
                with open(image_path, 'rb') as img:
                    with tf.Session().as_default() as sess:
                        if image_path.lower().endswith('png'):
                            image = sess.run(tf.image.decode_png(img.read()))
                        else:
                            image = sess.run(tf.image.decode_jpeg(img.read()))
                        height = image.shape[0]
                        width = image.shape[1]

            image = reader.get_image(image_path, height, width,
                                     self.image_preprocessing_fn)
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)
            # Remove batch dimension
            generated = tf.squeeze(generated, [0])
            outputs.append(self.sess.run(tf.image.encode_jpeg(generated)))

        return outputs


if __name__ == '__main__':
    import cv2
    import sys

    with Stylizer() as stylizer:
        for image in stylizer.stylize_images(sys.argv[1:]):
            cv2.imshow('image', image)
