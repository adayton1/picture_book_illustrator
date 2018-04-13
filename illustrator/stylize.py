# Adapted from https://github.com/ghwatson/faststyle/blob/master/stylize_image.py

import cv2
import numpy as np
import tensorflow as tf

from deps.faststyle.im_transf_net import create_net
import deps.faststyle.utils as utils

standard_image_size = (512, 512)


class Stylizer(object):
    def __init__(self, model_file_path, image_size=standard_image_size):
        self.model_path = model_file_path

        with tf.variable_scope('img_t_net'):
            self.inputs = tf.placeholder(tf.float32, shape=(1, image_size[0], image_size[1], 3), name='input')
            self.net = create_net(self.inputs, "resize")

    def __enter__(self):
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, self.model_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with tf.variable_scope('img_t_net'):
            self.sess.close()

    def stylize_image(self, img, content_target_resize=1.0):
        print('Stylizing image...')

        # Preprocess input image.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = utils.imresize(img, content_target_resize)
        orig_dim = img.shape
        img = cv2.resize(img, standard_image_size)
        img_4d = img[np.newaxis, :]

        print('Evaluating...')
        img_out = self.sess.run(self.net, feed_dict={self.inputs: img_4d})

        # Postprocess + save the output image.
        print('Saving image...')
        img_out = np.squeeze(img_out)

        # Original dimensions are (height, width, channels)
        # The resize function expects (width, height)
        new_dim = (orig_dim[1], orig_dim[0])
        img_out = cv2.resize(img_out, new_dim)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        print('Done stylizing image.')

        return img_out
