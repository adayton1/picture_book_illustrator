import glob
import math
import multiprocessing
import time

import tensorflow as tf


def train_input(file_paths, num_epochs, batch_size, image_dims, noise_dims, num_label_classes, num_take=-1):
	with tf.device('/cpu:0'):
		dataset = tf.data.TFRecordDataset(file_paths)

		def parser(record):
			keys_to_features = {
			    "feature/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
			    "label/encoded": tf.FixedLenFeature(shape=(), dtype=tf.int64)
			}
			parsed = tf.parse_single_example(record, keys_to_features)
			image = tf.image.decode_png(parsed["feature/encoded"], channels=image_dims[-1], dtype=tf.uint16)
			image = tf.image.convert_image_dtype(image, tf.float32)
			image = tf.reshape(image, image_dims)
			return image, tf.stack(tf.one_hot(parsed["label/encoded"], num_label_classes))

		dataset = dataset.shuffle(buffer_size=100000)
		dataset = dataset.take(num_take)
		dataset = dataset.map(parser, num_parallel_calls=multiprocessing.cpu_count() - 1)
		dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
		dataset = dataset.repeat(num_epochs)
		dataset = dataset.prefetch(buffer_size=round(min(num_take, 5e6) / batch_size))

		image, one_hot_label = dataset.make_one_shot_iterator().get_next()
		noise = tf.random_normal([batch_size, noise_dims])
	return (noise, one_hot_label), image


def predict_input(batch_size, noise_dims):
	noise = tf.random_normal([batch_size, noise_dims])
	return noise


def generator(inputs, activation_fn=tf.nn.relu, weight_decay=2.5e-5):
	"""Conditional generator to produce drawn images.

	Args:
		inputs: A 2-tuple of Tensors (noise, one_hot_labels).
		activation_fn: The activation fn.
		weight_decay: The value of the l2 weight decay.

	Returns:
		A generated image in the range [-1, 1].
	"""
	with tf.contrib.slim.arg_scope(
	    [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d_transpose],
	    activation_fn=activation_fn,
	    normalizer_fn=tf.contrib.layers.batch_norm,
	    weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
		net = tf.contrib.layers.fully_connected(inputs[0], 2048)
		net = tf.contrib.gan.features.condition_tensor_from_onehot(net, inputs[1])
		net = tf.contrib.layers.fully_connected(net, 7 * 7 * 256)
		net = tf.reshape(net, [-1, 7, 7, 256])
		net = tf.contrib.layers.conv2d_transpose(net, 128, [4, 4], stride=2)
		net = tf.contrib.layers.conv2d_transpose(net, 64, [4, 4], stride=2)
		# Make sure that generator output is in the same range as `inputs` ie [-1, 1].
		net = tf.contrib.layers.conv2d(net, 1, 4, activation_fn=tf.tanh, normalizer_fn=None)
		return net


def discriminator(image, conditioning, activation_fn=lambda net: tf.nn.leaky_relu(net, alpha=0.01),
                  weight_decay=2.5e-5):
	"""Discriminator network on images.

	Args:
		image: Real or generated image. Should be in the range [-1, 1].
		conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
		activation_fn: The activation function.
		weight_decay: The L2 weight decay.

	Returns:
		Logits for the probability that the image is real.
	"""
	with tf.contrib.slim.arg_scope(
	    [tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
	    activation_fn=activation_fn,
	    normalizer_fn=None,
	    weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
	    biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
		# FIXME: originally 64, changed because of noise_dims
		net = tf.contrib.layers.conv2d(image, 128, [4, 4], stride=2)
		# FIXME: originally 128, changed because of noise_dims
		net = tf.contrib.layers.conv2d(net, 256, [4, 4], stride=2)
		net = tf.contrib.layers.flatten(net)
		net = tf.contrib.gan.features.condition_tensor_from_onehot(net, conditioning[1])
		# FIXME: changed because of noise_dims
		net = tf.contrib.layers.fully_connected(net, 2048, normalizer_fn=tf.contrib.layers.layer_norm)
		return tf.contrib.layers.linear(net, 1)


def main():
	num_epochs = 800000
	batch_size = 64
	files = glob.glob('/mnt/pccfs/not_backed_up/data/quickdraw/*.tfrecords')
	image_dims = (28, 28, 1)  # height, width, channels
	noise_dims = 256  # FIXME: originally 64, cannot tune this hyperparameter without breaking everything
	num_label_classes = 385
	logdir = 'logdir/{}'.format(time.strftime('%Y%m%d-%H%M%S'))

	print("starting run", logdir)
	gan_estimator = tf.contrib.gan.estimator.GANEstimator(
		logdir,
		generator_fn=generator,
		discriminator_fn=discriminator,
		generator_loss_fn=tf.contrib.gan.losses.wasserstein_generator_loss,
		discriminator_loss_fn=tf.contrib.gan.losses.wasserstein_discriminator_loss,
		generator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
		discriminator_optimizer=tf.train.AdamOptimizer(0.00001, 0.5),
		add_summaries=tf.contrib.gan.estimator.SummaryType.IMAGES,
		config=tf.estimator.RunConfig(session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

	gan_estimator.train(
	    lambda: train_input(files, num_epochs, batch_size, image_dims, noise_dims, num_label_classes, num_take=3000000),
	    max_steps=num_epochs)


if __name__ == '__main__':
	main()
