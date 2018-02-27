import os

import tensorflow as tf

BATCH_SIZE = 32
NOISE_DIMS = 64
LABELS_FILENAME = 'labels.txt'
DATASET_DIR = '/mnt/pccfs/not_backed_up/data/mnist'


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
	"""Reads the labels file and returns a mapping from ID to class name.
	Args:
		dataset_dir: The directory in which the labels file is found.
		filename: The filename where the class names are written.
	Returns:
		A map from a label (integer) to class name.
	"""
	labels_filename = os.path.join(dataset_dir, filename)
	with tf.gfile.Open(labels_filename, 'rb') as f:
		lines = f.read().decode()
	lines = lines.split('\n')
	lines = filter(None, lines)

	labels_to_class_names = {}
	for line in lines:
		index = line.index(':')
		labels_to_class_names[int(line[:index])] = line[index + 1:]
	return labels_to_class_names


def get_split(split_name, dataset_dir, file_pattern='mnist_%s.tfrecord', reader=tf.TFRecordReader):
	"""Gets a dataset tuple with instructions for reading MNIST.
	Args:
		split_name: A train/test split name.
		dataset_dir: The base directory of the dataset sources.
		file_pattern: The file pattern to use when matching the dataset sources.
			It is assumed that the pattern contains a '%s' string so that the split
			name can be inserted.
		reader: The TensorFlow reader type.
	Returns:
		A `Dataset` namedtuple.
	Raises:
		ValueError: if `split_name` is not a valid train/test split.
	"""
	file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

	keys_to_features = {
		'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
		'image/class/label': tf.FixedLenFeature(
			[1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
	}

	items_to_handlers = {
		'image': tf.contrib.slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
		'label': tf.contrib.slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
	}

	decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
		keys_to_features, items_to_handlers)

	labels_to_names = None
	if tf.gfile.Exists(os.path.join(dataset_dir, LABELS_FILENAME)):
		labels_to_names = read_label_file(dataset_dir)

	return tf.contrib.slim.dataset.Dataset(
		data_sources=file_pattern,
		reader=reader,
		decoder=decoder,
		num_samples={'train': 60000, 'test': 10000}[split_name],
		num_classes=10,
		items_to_descriptions={
			'image': 'A [28 x 28 x 1] grayscale image.',
			'label': 'A single integer between 0 and 9',
		},
		labels_to_names=labels_to_names)


def provide_data(split_name, batch_size=BATCH_SIZE, dataset_dir=DATASET_DIR, num_readers=1, num_threads=1):
	"""Provides batches of MNIST digits.
	Args:
		split_name: Either 'train' or 'test'.
		batch_size: The number of images in each batch.
		dataset_dir: The directory where the MNIST data can be found.
		num_readers: Number of dataset readers.
		num_threads: Number of prefetching threads.
	Returns:
		images: A `Tensor` of size [batch_size, 28, 28, 1]
		one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
			each row has a single element set to one and the rest set to zeros.
			num_samples: The number of total samples in the dataset.
	Raises:
		ValueError: If `split_name` is not either 'train' or 'test'.
	"""
	dataset = get_split(split_name, dataset_dir=dataset_dir)
	provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
		dataset,
		num_readers=num_readers,
		common_queue_capacity=2 * batch_size,
		common_queue_min=batch_size,
		shuffle=(split_name == 'train'))
	[image, label] = provider.get(['image', 'label'])

	# Preprocess the images.
	image = (tf.to_float(image) - 128.0) / 128.0

	# Creates a QueueRunner for the pre-fetching operation.
	images, labels = tf.train.batch(
		[image, label],
		batch_size=batch_size,
		num_threads=num_threads,
		capacity=5 * batch_size)

	one_hot_labels = tf.one_hot(labels, dataset.num_classes)
	return images, one_hot_labels, dataset.num_samples


def train_input_fn():
	with tf.device('/cpu:0'):
		images, _, _ = provide_data('train', BATCH_SIZE, DATASET_DIR, num_threads=12)
	noise = tf.random_normal([BATCH_SIZE, NOISE_DIMS])
	return noise, images


def predict_input_fn():
	noise = tf.random_normal([BATCH_SIZE, NOISE_DIMS])
	return noise


def generator_fn(noise, activation_fn=tf.nn.relu, weight_decay=2.5e-5):
	"""Simple generator to produce MNIST images.

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
	"""Discriminator network on MNIST digits.

	Args:
		img: Real or generated MNIST digits. Should be in the range [-1, 1].
		unused_conditioning: The TFGAN API can help with conditional GANs, which
			would require extra `condition` information to both the generator and the
			discriminator. Since this example is not conditional, we do not use this
			argument.
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
		net = tf.contrib.layers.conv2d(img, 64, [4, 4], stride=2)
		net = tf.contrib.layers.conv2d(net, 128, [4, 4], stride=2)
		net = tf.contrib.layers.flatten(net)
		net = tf.contrib.layers.fully_connected(net, 1024, normalizer_fn=tf.contrib.layers.layer_norm)
		return tf.contrib.layers.linear(net, 1)


def main():
	gan_estimator = tf.contrib.gan.estimator.GANEstimator(
		'./logdir',
		generator_fn=generator_fn,
		discriminator_fn=discriminator_fn,
		generator_loss_fn=tf.contrib.gan.losses.wasserstein_generator_loss,
		discriminator_loss_fn=tf.contrib.gan.losses.wasserstein_discriminator_loss,
		generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
		discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
		add_summaries=tf.contrib.gan.estimator.SummaryType.IMAGES)

	gan_estimator.train(train_input_fn, max_steps=20000)


# gan_estimator.evaluate(eval_input_fn)

# predictions = np.array([x for x in gan_estimator.predict(predict_input_fn)])

# image_rows = [np.concatenate(predictions[i:i + 6], axis=0) for i in range(0, 36, 6)]
# tiled_image = np.concatenate(image_rows, axis=1)
# print(tiled_image)


if __name__ == '__main__':
	main()
