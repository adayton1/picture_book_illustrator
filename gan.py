from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import cv2
import sys

batch_size = 1
image_height = 28
image_width = 28


def conv(x, filter_size=8, stride=2, num_filters=64, is_output=False, name="conv"):
    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]
    out_channels = num_filters

    with tf.variable_scope(name):
        W = tf.get_variable("W",
                            shape=[filter_height, filter_width, in_channels, out_channels],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b",
                            shape=[out_channels],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        conv = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding="SAME")
        out = tf.nn.bias_add(conv, b)
        if is_output:
            return out
        return tf.nn.relu(out)
    # return tf.contrib.layers.batch_norm(tf.nn.relu(out))


def convt(x, out_shape, filter_size=8, stride=2, is_output=False, name="convt"):
    filter_height, filter_width = filter_size, filter_size
    in_channels = x.get_shape().as_list()[-1]

    with tf.variable_scope(name):
        W = tf.get_variable("W",
                            shape=[filter_height, filter_width, out_shape[-1], in_channels],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b",
                            shape=[out_shape[-1]],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        conv = tf.nn.conv2d_transpose(x, W, out_shape, [1, stride, stride, 1], padding="SAME")
        out = tf.nn.bias_add(conv, b)
        if is_output:
            return out
        return tf.nn.relu(out)
    # return tf.contrib.layers.batch_norm(tf.nn.relu(out))


def fc(x, out_size=50, is_output=False, name="fc"):
    in_size = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[in_size, out_size],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", shape=[out_size],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        out = tf.matmul(x, W) + b
        if is_output:
            return out
        return tf.nn.relu(out)


def gen_model(seed_imgs):
    with tf.variable_scope("generator"):
        # fc1 = fc(seed_imgs, out_size=7*7*128, name="g_fc1")
        # fc1_reshaped = tf.reshape(fc1, [batch_size, 7, 7, 128])
        # convt1 = convt(seed_imgs, [batch_size, 7, 7, 128], filter_size=5, stride=2, name="g_convt1")
        conv1 = conv(seed_imgs, filter_size=5, stride=2, num_filters=64, name="g_convt1")
        # print("convt1", conv1.get_shape())
        convt2 = convt(conv1, [batch_size, 28, 28, 1], filter_size=5, stride=2, is_output=True, name="g_convt2")
        # print("convt2", convt2.get_shape())
        return tf.sigmoid(convt2)


def disc_model(imgs, reuse):
    with tf.variable_scope("discriminator", reuse=reuse):
        # imgs_reshaped = tf.reshape(imgs, [batch_size, 28, 28, 1])
        conv1 = conv(imgs, filter_size=5, stride=2, num_filters=32, name="d_conv1")
        # print("conv1", conv1.get_shape())
        conv2 = conv(conv1, filter_size=5, stride=2, num_filters=64, name="d_conv2")
        # print("conv2", conv2.get_shape())
        conv2_reshaped = tf.reshape(conv2, [batch_size, 7 * 7 * 64])
        # print("conv2_reshaped", conv2_reshaped.get_shape())
        fc1 = fc(conv2_reshaped, out_size=1024, name="d_fc1")
        # print("fc1", fc1.get_shape())
        fc_out = fc(fc1, out_size=1, is_output=True, name="d_fc_out")
        # print("fc_out", fc_out.get_shape())
        return tf.sigmoid(fc_out)  # probability


with tf.name_scope('gan'):
    gen_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name="gen_input")
    disc_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name="disc_input")  # true images

    gen_output = gen_model(gen_input)

    true_probs = disc_model(disc_input, reuse=False)
    gen_probs = disc_model(gen_output, reuse=True)

    disc_loss = tf.reduce_mean(tf.log(true_probs + 0.4) + tf.log(0.6 - gen_probs))
    gen_loss = tf.reduce_mean(tf.log(gen_probs))
    disc_acc = (tf.reduce_mean(true_probs) + tf.reduce_mean(1.0 - gen_probs)) / 2.0  # TODO: images or probs?

    t_vars = tf.trainable_variables()

    disc_vars = [var for var in t_vars if "d_" in var.name]
    disc_optim = tf.train.AdamOptimizer(0.0000001, beta1=0.01).minimize(-disc_loss, var_list=disc_vars)

    gen_vars = [var for var in t_vars if "g_" in var.name]
    gen_optim = tf.train.AdamOptimizer(0.00000001, beta1=0.01).minimize(-gen_loss, var_list=gen_vars)


# TODO: Train discriminator on true images 1 time for every 3 times on generated images

def load_data(path='/mnt/pccfs/not_backed_up/data/quickdraw'):
    for filename in os.listdir(path):
        for image in np.load(os.path.join(path, filename)):
            yield np.reshape(image / 255.0, (batch_size, image_height, image_width, 1))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(100000):
        i = 0

        seed_image, true_image = None, None
        for image in load_data():
            if seed_image is None:
                seed_image = image
                continue
            if true_image is None:
                true_image = image

            # train the GAN on a real image
            sess.run(disc_optim, feed_dict={gen_input: seed_image, disc_input: true_image})

            # train the GAN on 3 generated images
            for j in range(3):
                sess.run(gen_optim, feed_dict={gen_input: seed_image})

            if i % 10 == 0:
                cv2.imshow("seed image", np.reshape(seed_image, (image_height, image_width)))
                cv2.imshow("true image", np.reshape(true_image, (image_height, image_width)))

                disc_acc_val, disc_loss_val, gen_loss_val, gen_output_val, true_probs_val, gen_probs_val = sess.run(
                    [disc_acc, disc_loss, gen_loss, gen_output, true_probs, gen_probs],
                    feed_dict={gen_input: seed_image, disc_input: true_image})
                print(i, disc_loss_val, gen_loss_val, disc_acc_val, true_probs_val, gen_probs_val, sep='\t')

                cv2.imshow("generated", gen_output_val[0])

                cv2.waitKey(10)

            seed_image = None
            true_image = None
            i += 1
