import glob
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def record_writer(path, limit, class_types):
	name = 'train'
	if class_types:
		name = '{}_{}'.format(name, '_'.join(class_types))
	if limit:
		'{}_{}'.format(name, limit)

	dest_path = '{}/{}.tfrecords'.format(path, name)
	return tf.python_io.TFRecordWriter(dest_path)


def load_data(path, limit, class_types=(), image_dims=(28, 28, 1), image_format=b'png'):
	writer = record_writer(path, limit, class_types)

	for label_id, filename in enumerate(tqdm(sorted(os.listdir(path)))):
		label_human = os.path.splitext(filename)[0]
		if class_types and label_human not in class_types:
			continue

		raw = np.load(os.path.join(path, filename))
		sample = raw[np.random.choice(raw.shape[0], limit, replace=False), :] if limit else raw
		for image in tqdm(sample):
			img = BytesIO()
			Image.fromarray(np.atleast_2d(image)).save(img, 'PNG')
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

			writer.write(example.SerializeToString())


if __name__ == '__main__':
	load_data(path='/mnt/pccfs/not_backed_up/data/quickdraw', limit=0, class_types=('cat',))
