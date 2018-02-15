from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import cv2
import sys


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


def load_data(path='/mnt/pccfs/not_backed_up/data/quickdraw'):
	for filename in os.listdir(path):
		for image in np.load(os.path.join(path, filename)):
			yield image


batch_size = 1
channels = 1
vector_cols = 345  # TODO
image_height = 28
image_width = 28

input_shape = [batch_size, vector_cols]
output_shape = [batch_size, image_height, image_width, channels]

x = tf.placeholder(tf.float32, input_shape, name="x")
y_ = tf.placeholder(tf.float32, output_shape, name="y_")

# CONVOLUTIONAL VERSION
fc1 = fc(x, out_size=7 * 7 * 32, name="fc1")
fc1_reshaped = tf.reshape(fc1, [batch_size, 7, 7, 32])
convt1 = convt(fc1_reshaped, [batch_size, 14, 14, 16], filter_size=3, stride=2, name="convt1")
y = convt(convt1, output_shape, filter_size=3, stride=2, is_output=True)

# FULLY CONNECTED VERSION
# fc1 = fc(x, out_size=500, name="fc1")
# fc2 = fc(fc1, out_size=150, name="fc2")
# fc3 = fc(fc2, out_size=500, name="fc3")
# fc4 = fc(fc3, out_size=784, is_output=True, name="fc4")
# y = tf.reshape(fc4, [batch_size, image_height, image_width])

loss = tf.reduce_mean((y_ - y) ** 2)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(100000):
		i = 0
		for image in load_data():
			i += 1
			if i > 300:
				break

			input_vec = np.zeros([1, vector_cols])
			input_vec[0, i] = 1.0

			target = np.reshape(image, (batch_size, image_height, image_width, channels))

			_, loss_val, output = sess.run([optimizer, loss, y], feed_dict={x: input_vec, y_: target})

			print("LOSS:", loss_val)
			cv2.imshow("target", np.reshape(image, (image_height, image_width)))
			cv2.imshow("output", output[0])

			cv2.waitKey(10)
