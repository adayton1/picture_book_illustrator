import os
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf
import cv2

# HACK
import utils
utils.extend_syspath([
 'deps/tensorflow_models/research',
 'deps/tensorflow_models/research/im2txt',
 'deps/tensorflow_models/research/slim',
 'deps/tensorflow_models/research/object_detection'
])  # yapf: disable

from im2txt import configuration, inference_wrapper
from im2txt.inference_utils import caption_generator, vocabulary
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

tf.logging.set_verbosity(tf.logging.WARN)


class ImageCaptioner(object):
    """Generate captions for images using default beam search parameters."""

    def __init__(
            self,
            checkpoint_path=None,
            vocab_file=None,
            model_dir=os.path.join(utils.project_root,
                                   'deps/Pretrained-Show-and-Tell-model')):
        print('Loading ImageCaptioner model...')

        if checkpoint_path is None:
            checkpoint_path = os.path.join(model_dir, 'model.ckpt-2000000')

        if vocab_file is None:
            vocab_file = os.path.join(model_dir, 'word_counts.txt')
        # Build the inference graph.
        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(
                configuration.ModelConfig(), checkpoint_path)
        g.finalize()

        # Create the vocabulary.
        self.vocab = vocabulary.Vocabulary(vocab_file)

        self.sess = tf.Session(graph=g)

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
            try:
                encoded_image = tf.gfile.GFile(image_file, "rb").read()
                captions = self.generator.beam_search(self.sess, encoded_image)
            except:
                sentences.append("")
                continue

            # Ignore begin and end words.
            sentence = [
                self.vocab.id_to_word(w) for w in captions[0].sentence[1:-1]
            ]
            sentences.append(" ".join(sentence))
        return sentences


class ObjectDetector(object):
    """Detects object in an image.
    Adapted from: https://github.com/tensorflow/models/blob/6ff0a53f/research/object_detection/object_detection_tutorial.ipynb
    """

    def __init__(
            self,
            max_num_classes=90,
            checkpoint_path=os.path.join(
                utils.project_root,
                'deps/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
            ),
            label_map=os.path.join(
                utils.project_root,
                'deps/tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt'
            )):
        print('Loading ObjectDetector model...')
        self.graph = tf.Graph()
        self.sess = tf.Session()

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(label_map)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=max_num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.sess.close()
        pass

    def compute_bounding_boxes(self, images):
        if not isinstance(images, list):
            images = [images]
        outputs = []
        with self.graph.as_default():
            with tf.Session() as sess:
                get_tensor_by_name = tf.get_default_graph().get_tensor_by_name
                for image in images:
                    if isinstance(image, str):
                        image = self.load_image(image)
                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {
                        output.name
                        for op in ops for output in op.outputs
                    }
                    tensor_dict = {}
                    for key in [
                            'num_detections',
                            'detection_boxes',
                            'detection_scores',
                            'detection_classes',
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = get_tensor_by_name(tensor_name)
                    image_tensor = get_tensor_by_name('image_tensor:0')

                    # Run inference
                    model_output = sess.run(
                        tensor_dict,
                        feed_dict={
                            image_tensor: np.expand_dims(image, axis=0)
                        })

                    output = defaultdict(list)
                    classes = model_output['detection_classes'][0].astype(int)
                    for i in range(int(model_output['num_detections'][0])):
                        box = model_output['detection_boxes'][0][i]
                        # convert to normalized (pixel) coordinates
                        # ymin, xmin, ymax, xmax = box
                        box[0] *= image.shape[0]
                        box[1] *= image.shape[1]
                        box[2] *= image.shape[0]
                        box[3] *= image.shape[1]
                        key = self.category_index[classes[i]]['name']
                        output[key].append(box.astype(int))
                    outputs.append(output)
        return outputs

    def load_image(self, path):
        return cv2.imread(path)


if __name__ == '__main__':
    import random
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image

    colors = list(matplotlib.colors.cnames.keys())
    used_colors = []

    def get_unused_color():
        color = random.choice(colors)
        while color in used_colors:
            color = random.choice(colors)
        used_colors.append(color)
        return color

    with ObjectDetector() as detector:
        boxes = detector.compute_bounding_boxes(sys.argv[1:])
        for i, box_output in enumerate(boxes):
            image = Image.open(sys.argv[i + 1])  # HACK

            plt.imshow(image)
            for label, box_group in box_output.items():
                color = get_unused_color()
                for box in box_group:
                    ymin, xmin, ymax, xmax = box
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (xmin, ymin),
                            width=xmax - xmin,
                            height=ymax - ymin,
                            color=color,
                            label=label,
                            fill=False,
                            linewidth=2))
                    label = None
            plt.legend()
            plt.show()
            used_colors = []
