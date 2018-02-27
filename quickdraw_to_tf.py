import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

data_root = '/mnt/pccfs/not_backed_up/data'
limit = 0
image_height = 28
image_width = 28
image_format = b'PNG'
image_num_channels = 1


def int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def load_data_tensorflow(src_path=data_root + '/quickdraw'):
	dest_path = '{}/quickdraw_tf/train{}.tfrecords'.format(data_root, '_{}'.format(limit) if limit else '')
	writer = tf.python_io.TFRecordWriter(dest_path)

	with tf.Graph().as_default():
		image_placeholder = tf.placeholder(dtype=tf.uint16)
		encoded_image = tf.image.encode_png(image_placeholder, compression=3)

		config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
		# config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 10}, allow_soft_placement=True,
		# 						log_device_placement=True)
		with tf.Session('', config=config) as sess:
			for label_id, filename in enumerate(tqdm(os.listdir(src_path))):
				raw = np.load(os.path.join(src_path, filename))
				sample = raw[np.random.choice(raw.shape[0], limit, replace=False), :] if limit else raw
				for image in sample:
					image = np.reshape(image, (image_height, image_width, image_num_channels))
					img = sess.run(encoded_image, feed_dict={image_placeholder: image})

					feature = {
						'feature/encoded': bytes_feature(img),
						'feature/format': bytes_feature(image_format),
						'label/encoded': int64_feature(label_id),
						'label/human': bytes_feature(tf.compat.as_bytes(os.path.splitext(filename)[0])),
						'feature/height': int64_feature(image_height),
						'feature/width': int64_feature(image_width),
					}

					example = tf.train.Example(features=tf.train.Features(feature=feature))
					writer.write(example.SerializeToString())


def load_data(src_path=data_root + '/quickdraw'):
	dest_path = '{}/quickdraw_tf/train{}.tfrecords'.format(data_root, '_{}'.format(limit) if limit else '')
	writer = tf.python_io.TFRecordWriter(dest_path)

	for label_id, filename in enumerate(tqdm(os.listdir(src_path))):
		raw = np.load(os.path.join(src_path, filename))
		sample = raw[np.random.choice(raw.shape[0], limit, replace=False), :] if limit else raw
		for image in sample:
			img = BytesIO()
			Image.fromarray(np.atleast_2d(image)).save(img, 'PNG')
			img = img.getvalue()

			feature = {
				'feature/encoded': bytes_feature(img),
				'feature/format': bytes_feature(image_format),
				'label/encoded': int64_feature(label_id),
				'label/human': bytes_feature(tf.compat.as_bytes(os.path.splitext(filename)[0])),
				'feature/height': int64_feature(image_height),
				'feature/width': int64_feature(image_width),
			}
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())


if __name__ == '__main__':
	load_data()
