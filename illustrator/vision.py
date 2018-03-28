import tensorflow as tf
import tensornets as nets
from tensornets.datasets import voc


class ObjectDetector(object):
	def __init__(self, inputs=tf.placeholder(tf.float32, [None, 512, 512, 3]), classnames=voc.classnames):
		self.inputs = inputs
		self.model = nets.YOLOv2(self.inputs, nets.Darknet19)
		self.classnames = classnames

	def __enter__(self):
		self.sess = tf.Session()
		self.sess.run(self.model.pretrained())
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.sess.close()

	def compute_bounding_boxes(self, image):
		preds = self.sess.run(self.model, {self.inputs: self.model.preprocess(image)})
		boxes = self.model.get_boxes(preds, image.shape[1:3])
		return {self.classnames[i]: g for i, g in enumerate(boxes)}

	def load_image(self, *args, **kwargs):
		if kwargs.get('target_size') is None and kwargs.get('crop_size') is None:
			kwargs['target_size'] = self.inputs.get_shape().as_list()[1:3]
		return nets.utils.load_img(*args, **kwargs)


if __name__ == '__main__':
	import random
	import sys
	import numpy as np
	import matplotlib
	import matplotlib.pyplot as plt

	colors = list(matplotlib.colors.cnames.keys())
	used_colors = []

	def get_unused_color():
		color = random.choice(colors)
		while color in used_colors:
			color = random.choice(colors)
		used_colors.append(color)
		return color

	with ObjectDetector() as detector:
		for path in sys.argv[1:]:
			image = detector.load_image(path)
			boxes = detector.compute_bounding_boxes(image)

			plt.imshow(image[0].astype(np.uint8))
			for label, box_group in boxes.items():
				color = get_unused_color()
				for box in box_group:
					plt.gca().add_patch(
					    plt.Rectangle(
					        (box[0], box[1]),
					        box[2] - box[0],
					        box[3] - box[1],
					        color=color,
					        label=label,
					        fill=False,
					        linewidth=2))
					label = None
			plt.legend()
			plt.show()
			used_colors = []
