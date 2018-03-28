import tensorflow as tf
import tensornets as nets


class ObjectDetector(object):
	def __init__(self, inputs=tf.placeholder(tf.float32, [None, 512, 512, 3])):
		self.inputs = inputs
		self.model = nets.YOLOv2(self.inputs, nets.Darknet19)

	def __enter__(self):
		self.sess = tf.Session()
		self.sess.run(self.model.pretrained())
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.sess.close()

	def compute_bounding_boxes(self, image):
		preds = self.sess.run(self.model, {self.inputs: self.model.preprocess(image)})
		return self.model.get_boxes(preds, image.shape[1:3])

	def load_image(self, *args, **kwargs):
		if kwargs.get('target_size') is None and kwargs.get('crop_size') is None:
			kwargs['target_size'] = self.inputs.get_shape().as_list()[1:3]
		return nets.utils.load_img(*args, **kwargs)


if __name__ == '__main__':
	from tensornets.datasets import voc
	import numpy as np
	import matplotlib.pyplot as plt

	with ObjectDetector() as detector:
		image = detector.load_image('data/cat.png')
		boxes = detector.compute_bounding_boxes(image)

		print("%s: %s" % (voc.classnames[7], boxes[7][0]))  # 7 is cat

		box = boxes[7][0]
		plt.imshow(image[0].astype(np.uint8))
		plt.gca().add_patch(
		    plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='r', linewidth=2))
		plt.show()
