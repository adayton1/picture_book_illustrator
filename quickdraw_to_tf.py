import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

data_root = '/mnt/pccfs/not_backed_up/data'
limit = 100


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_data(path=data_root + '/quickdraw'):
	dest = '{}/quickdraw_tf/train{}.tfrecords'.format(data_root, '_{}'.format(limit) if limit else '')
	writer = tf.python_io.TFRecordWriter(dest)

	for label, filename in enumerate(tqdm(os.listdir(path))):
		raw = np.load(os.path.join(path, filename))
		sample = raw[np.random.choice(raw.shape[0], limit, replace=False), :] if limit else raw
		for image in sample:
			img = BytesIO()
			Image.fromarray(np.atleast_2d(image)).save(img, 'PNG')
			img = img.getvalue()
			# img = image.astype(np.float32).tostring()

			feature = {
				'label/one_hot': _int64_feature(label),
				'label/human': _bytes_feature(tf.compat.as_bytes(os.path.splitext(filename)[0])),
				'feature/png': _bytes_feature(tf.compat.as_bytes(img))
			}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())


if __name__ == '__main__':
	load_data()
