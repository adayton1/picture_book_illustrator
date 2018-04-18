import pathlib
import sys

import tensorflow as tf
import tensornets as nets
from tensornets.datasets import voc

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

tf.logging.set_verbosity(tf.logging.WARN)


class ImageCaptioner(object):
	"""Generate captions for images using default beam search parameters."""

	def __init__(
	    self,
	    checkpoint_path=None,
	    vocab_file=None,
	    model_dir=pathlib.Path(__file__).parent.parent.joinpath('deps/Pretrained-Show-and-Tell-model').resolve()):
		if checkpoint_path is None:
			checkpoint_path = str(model_dir.joinpath('model.ckpt-2000000'))

		if vocab_file is None:
			vocab_file = str(model_dir.joinpath('word_counts.txt'))
		# Build the inference graph.
		g = tf.Graph()
		with g.as_default():
			model = inference_wrapper.InferenceWrapper()
			restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)
		g.finalize()

		# Create the vocabulary.
		self.vocab = vocabulary.Vocabulary(vocab_file)

		# TODO: add GPU support by fixing: Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms)
		self.sess = tf.Session(graph=g, config=tf.ConfigProto(device_count={'GPU': 0}))

		# Load the model from checkpoint.
		restore_fn(self.sess)

		# Prepare the caption generator. Here we are implicitly using the default
		# beam search parameters. See caption_generator.py for a description of the
		# available beam search parameters.
		self.generator = caption_generator.CaptionGenerator(model, self.vocab)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.sess.close()

	def generate(self, image_files):
		sentences = []
		for image_file in image_files:
			captions = self.generator.beam_search(self.sess, tf.gfile.GFile(image_file, "rb").read())
			# Ignore begin and end words.
			sentence = [self.vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
			sentences.append(" ".join(sentence))
		return sentences


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
