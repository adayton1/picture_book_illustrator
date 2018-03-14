import multiprocessing
import glob
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image


def int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def write_tfrecord(label_id, src_path, limit=0, image_dims=(28, 28, 1), image_format=b'png', data_dir=None):
	if not data_dir:
		data_dir = os.path.dirname(src_path)
	label_human = os.path.splitext(os.path.basename(src_path))[0].lower()
	dest_path = os.path.join(data_dir, '{}{}.tfrecords'.format(label_human, '_{}'.format(limit) if limit else ''))

	record_writer = tf.python_io.TFRecordWriter(dest_path)

	raw = np.load(src_path)
	sample = raw[np.random.choice(raw.shape[0], limit, replace=False), :] if limit else raw
	for image in sample:
		img = BytesIO()
		Image.fromarray(np.reshape(image, image_dims)).save(img, 'PNG')
		img = img.getvalue()

		feature = {
			'feature/encoded': bytes_feature(img),
			'feature/format': bytes_feature(image_format),
			'feature/height': int64_feature(image_dims[0]),
			'feature/width': int64_feature(image_dims[1]),
			'feature/channels': int64_feature(image_dims[2]),
			'label/encoded': int64_feature(label_id),
			'label/human': bytes_feature(tf.compat.as_bytes(label_human)),
		}
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		record_writer.write(example.SerializeToString())


if __name__ == '__main__':
	with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
		pool.starmap(write_tfrecord, enumerate(sorted(glob.glob('/mnt/pccfs/not_backed_up/data/quickdraw/*.npy'))))
