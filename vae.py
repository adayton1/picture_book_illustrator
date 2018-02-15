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
		#return tf.contrib.layers.batch_norm(tf.nn.relu(out))

def convt(x, out_shape, filter_size=8, stride=2, is_output=False, name="convt"):
	filter_height, filter_width = filter_size, filter_size
	in_channels = x.get_shape().as_list()[-1]

	with tf.variable_scope(name):
		W = tf.get_variable("W",
			shape=[filter_height, filter_width, out_shape[-1], in_channels],
			initializer = tf.contrib.layers.variance_scaling_initializer())
		b = tf.get_variable("b",
			shape=[out_shape[-1]],
			initializer = tf.contrib.layers.variance_scaling_initializer())
		conv = tf.nn.conv2d_transpose(x, W, out_shape, [1, stride, stride, 1], padding="SAME")
		out = tf.nn.bias_add(conv, b)
		if is_output:
			return out
		return tf.nn.relu(out)
		#return tf.contrib.layers.batch_norm(tf.nn.relu(out))

def fc(x, out_size=50, is_output=False, name="fc"):
	in_size = x.get_shape().as_list()[-1]
	with tf.variable_scope(name):
		W = tf.get_variable("W", shape=[in_size, out_size],
			initializer = tf.contrib.layers.variance_scaling_initializer())
		b = tf.get_variable("b", shape=[out_size],
			initializer = tf.contrib.layers.variance_scaling_initializer())
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
vector_cols = 345 # TODO
image_height = 28
image_width = 28

input_shape = [batch_size, vector_cols]
output_shape = [batch_size, image_height, image_width, channels]

x = tf.placeholder(tf.float32, input_shape, name="x")
y_ = tf.placeholder(tf.float32, output_shape, name="y_")



# ENCODER
fc1 = fc(x, out_size=150, name="fc1")
fc2 = fc(fc1, out_size=100, name="fc2")
n_z = 5
z_mean = fc(fc2, out_size=n_z, is_output=True, name="z_mean")
z_stddev = fc(fc2, out_size=n_z, is_output=True, name="z_stddev")

# the variational part in the middle
true_sample = tf.random_normal([batch_size, n_z])
#sampled_z = z_mean + (z_stddev * true_sample)
sampled_z = z_mean + tf.exp(z_stddev / 2.0) * true_sample

# DECODER
fc3 = fc(sampled_z, out_size=7*7*32, name="fc3")
fc3_reshaped = tf.reshape(fc3, [batch_size, 7, 7, 32])
convt1 = convt(fc3_reshaped, [batch_size, 14, 14, 16], filter_size=3, stride=2, name="convt1")
y = convt(convt1, output_shape, filter_size=3, stride=2, is_output=True, name="y")

# LOSS FUNCTION
generation_loss = tf.reduce_mean((y_-y)**2.0)
latent_loss = 0.5 * tf.reduce_sum(z_mean**2.0 + z_stddev**2.0 - tf.log(z_stddev**2.0+1e10) - 1.0)
loss = tf.reduce_mean(generation_loss + latent_loss)

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	for epoch in range(100000):
		i = 0
		for image in load_data():
			i += 1
			if i > 300:
				break

			input_vec = np.zeros([1, vector_cols])
			input_vec[0,i] = 1.0

			target = np.reshape(image, (batch_size, image_height, image_width, channels))

			_, loss_val, output, gen_loss_val, lat_loss_val, z_mean_val, z_stddev_val = sess.run(
				[optimizer, loss, y, generation_loss, latent_loss, z_mean, z_stddev],
				feed_dict={x: input_vec, y_: target})

			print("LOSS:", loss_val, "GEN LOSS:", gen_loss_val, "LATENT LOSS:", lat_loss_val) #, z_mean_val, z_stddev_val)
			cv2.imshow("target", np.reshape(image, (image_height, image_width)))
			cv2.imshow("output", output[0])

			cv2.waitKey(10)

		if epoch % 10 == 0:
			print(" ** Saving weights **")
			saver.save(sess, "vae_autosave.ckpt")


