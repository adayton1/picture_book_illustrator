import tensorflow as tf


def train_input_fn(file, num_epochs, batch_size, image_dims, noise_dims):
	dataset = tf.data.TFRecordDataset([file])

	def parser(record):
		keys_to_features = {
			"png": tf.FixedLenFeature((), tf.string, default_value=""),
			# "label": tf.FixedLenFeature((), tf.int64),
		}
		parsed = tf.parse_single_example(record, keys_to_features)
		image = tf.image.decode_png(parsed["png"], channels=image_dims[-1], dtype=tf.uint16)
		image = tf.image.convert_image_dtype(image, tf.float32)
		image = tf.reshape(image, image_dims)
		# image = (tf.to_float(image) - 128.0) / 128.0
		return image

	dataset = dataset.map(parser)
	dataset = dataset.shuffle(buffer_size=1000)
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
	dataset = dataset.repeat(num_epochs)
	iterator = dataset.make_one_shot_iterator()

	images = iterator.get_next()
	noise = tf.random_normal([batch_size, noise_dims])
	return noise, images


def predict_input_fn(batch_size, noise_dims):
	noise = tf.random_normal([batch_size, noise_dims])
	return noise


def generator_fn(noise, activation_fn=tf.nn.relu, weight_decay=2.5e-5):
	"""Simple generator to produce images.

	Args:
		noise: A single Tensor representing noise.
		activation_fn: The activation fn.
		weight_decay: The value of the l2 weight decay.

	Returns:
		A generated image in the range [-1, 1].
	"""
	with tf.contrib.slim.arg_scope(
			[tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d_transpose],
			activation_fn=activation_fn, normalizer_fn=tf.contrib.layers.batch_norm,
			weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
		net = tf.contrib.layers.fully_connected(noise, 1024)
		net = tf.contrib.layers.fully_connected(net, 7 * 7 * 128)
		net = tf.reshape(net, [-1, 7, 7, 128])
		net = tf.contrib.layers.conv2d_transpose(net, 64, [4, 4], stride=2)
		net = tf.contrib.layers.conv2d_transpose(net, 32, [4, 4], stride=2)
		# Make sure that generator output is in the same range as `inputs` ie [-1, 1].
		net = tf.contrib.layers.conv2d(net, 1, 4, activation_fn=tf.tanh, normalizer_fn=None)
		return net


def discriminator_fn(img, unused_conditioning=None, activation_fn=lambda net: tf.nn.leaky_relu(net, alpha=0.01),
					 weight_decay=2.5e-5):
	"""Discriminator network on images.

	Args:
		img: Real or generated image. Should be in the range [-1, 1].
		unused_conditioning: The TFGAN API can help with conditional GANs, which
			would require extra `condition` information to both the generator and the
			discriminator. This argument is not used because this is an unconditional GAN.
		activation_fn: The activation fn.
		weight_decay: The L2 weight decay.

	Returns:
		Logits for the probability that the image is real.
	"""
	with tf.contrib.slim.arg_scope(
			[tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
			activation_fn=activation_fn, normalizer_fn=None,
			weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
			biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
		net = tf.contrib.layers.conv2d(img, 128, [4, 4], stride=2)  # FIXME: originally 64, changed because of noise_dims
		net = tf.contrib.layers.conv2d(net, 256, [4, 4], stride=2)  # FIXME: originally 128, changed because of noise_d
		net = tf.contrib.layers.flatten(net)
		net = tf.contrib.layers.fully_connected(net, 1024, normalizer_fn=tf.contrib.layers.layer_norm)
		return tf.contrib.layers.linear(net, 1)


def main():
	num_epochs = 300000
	batch_size = 16
	train_file = '/mnt/pccfs/not_backed_up/data/quickdraw_tf/slim.tfrecords'
	image_dims = (28, 28, 1)  # height, width, channels
	noise_dims = 128  # FIXME: originally 64, cannot tune this hyperparameter without breaking everything

	gan_estimator = tf.contrib.gan.estimator.GANEstimator(
		'logdir',
		generator_fn=generator_fn,
		discriminator_fn=discriminator_fn,
		generator_loss_fn=tf.contrib.gan.losses.wasserstein_generator_loss,
		discriminator_loss_fn=tf.contrib.gan.losses.wasserstein_discriminator_loss,
		generator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
		discriminator_optimizer=tf.train.AdamOptimizer(0.00001, 0.5),
		add_summaries=tf.contrib.gan.estimator.SummaryType.IMAGES)

	gan_estimator.train(lambda: train_input_fn(train_file, num_epochs, batch_size, image_dims, noise_dims),
						max_steps=num_epochs)

# gan_estimator.evaluate(eval_input_fn)

# predictions = np.array([x for x in gan_estimator.predict(lambda: predict_input_fn(batch_size, image_dims))])

# image_rows = [np.concatenate(predictions[i:i + 6], axis=0) for i in range(0, 36, 6)]
# tiled_image = np.concatenate(image_rows, axis=1)
# print(tiled_image)


if __name__ == '__main__':
	main()
