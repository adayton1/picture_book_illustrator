import os
import pathlib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

import numpy as np
import tensorflow as tf
from PIL import Image

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
        if checkpoint_path is None:
            checkpoint_path = str(model_dir.joinpath('model.ckpt-2000000'))

        if vocab_file is None:
            vocab_file = str(model_dir.joinpath('word_counts.txt'))
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
            captions = self.generator.beam_search(self.sess,
                                                  tf.gfile.GFile(
                                                      image_file, "rb").read())
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
        outputs = []
        with self.graph.as_default():
            with tf.Session() as sess:
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
                            'num_detections', 'detection_boxes',
                            'detection_scores', 'detection_classes',
                            'detection_masks'
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph(
                            ).get_tensor_by_name(tensor_name)
                    if 'detection_masks' in tensor_dict:
                        # The following processing is only for single image
                        detection_boxes = tf.squeeze(
                            tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(
                            tensor_dict['detection_masks'], [0])
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        real_num_detection = tf.cast(
                            tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0],
                                                   [real_num_detection, -1])
                        detection_masks = tf.slice(
                            detection_masks, [0, 0, 0],
                            [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image.shape[0],
                            image.shape[1])
                        detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5),
                            tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
                    image_tensor = tf.get_default_graph().get_tensor_by_name(
                        'image_tensor:0')

                    # Run inference
                    output_dict = sess.run(
                        tensor_dict,
                        feed_dict={
                            image_tensor: np.expand_dims(image, axis=0)
                        })

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(
                        output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict[
                        'detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict[
                        'detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict[
                            'detection_masks'][0]
                    outputs.append(output_dict)
        return outputs

    def load_image(self, path):
        image = Image.open(path)
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width,
                                                  3)).astype(np.uint8)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from object_detection.utils import visualization_utils as vis_util

    with ObjectDetector() as detector:
        for box in detector.compute_bounding_boxes(sys.argv[1:]):
            print(box)
            img = detector.load_image(sys.argv[1])
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                box['detection_boxes'],
                box['detection_classes'],
                box['detection_scores'],
                detector.category_index,
                instance_masks=box.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
