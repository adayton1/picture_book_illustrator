from __future__ import unicode_literals
import codecs
import os
import subprocess
import spacy

import tensorflow as tf
import numpy as np
from deps.faststyle.im_transf_net import create_net
import deps.faststyle.utils as utils

# Load English model
nlp = spacy.load('en_core_web_lg')


def read_file(input_file):

    with codecs.open(input_file, "r", "utf-8") as f:
        text = f.read()

    pages = text.split("\n\n")

    return text, pages


# Adapted from https://github.com/ghwatson/faststyle/blob/master/stylize_image.py
def stylize_image(input_img_path, output_img_path, model_path, upsample_method='resize', content_target_resize=1.0):
    print('Stylizing image...')

    # Read + preprocess input image.
    img = utils.imread(input_img_path)
    img = utils.imresize(img, content_target_resize)
    img_4d = img[np.newaxis, :]

    # Create the graph.
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=img_4d.shape, name='input')
        Y = create_net(X, upsample_method)

    # Saver used to restore the model to the session.
    saver = tf.train.Saver()

    # Filter the input image.
    with tf.Session() as sess:
        print('Loading up model...')
        saver.restore(sess, model_path)
        print('Evaluating...')
        img_out = sess.run(Y, feed_dict={X: img_4d})

    # Postprocess + save the output image.
    print('Saving image...')
    img_out = np.squeeze(img_out)
    utils.imwrite(output_img_path, img_out)

    print('Done stylizing image.')


def illustrate(input_file, output_dir, style_model):

    print("Reading input file...")
    text, pages = read_file(input_file)

    # Process the whole doc
    # doc = nlp(text)

    # Iterate through each page
    for i, page in enumerate(pages):
        print("\n\nIllustrating page {0}...".format(i))

        print("Natural language processing...")
        page_doc = nlp(page)

        nouns = []

        for chunk in page_doc.noun_chunks:
            noun = chunk.root

            if noun.lemma_ != "-PRON-":
                nouns.append(noun.lemma_)

        # Download image
        print("Downloading image...")
        keywords = "{0}".format(" ".join(nouns))
        subprocess.call(["googleimagesdownload", "-k", keywords, "-l", "1", "-o", output_dir, "-f", "jpg"])

        # Move image to destination
        temp_dir = os.path.join(output_dir, keywords)
        temp = subprocess.check_output(["ls", temp_dir])[:-1]   # Throw away the \n character returned by ls
        temp = temp.decode("utf-8")
        filename, file_extension = os.path.splitext(temp)
        full_file_path = os.path.join(temp_dir, temp)
        destination = os.path.join(output_dir, "{0}".format(i) + file_extension)
        subprocess.call(["mv", full_file_path, destination])

        # Clear out the temporary folder
        subprocess.call(["rm", "-r", temp_dir])

        # Stylize image
        stylize_image(destination, destination, style_model)
        #subprocess.call(["python", "stylize_image.py", "--input_img_path", destination, "--output_img_path",
        #                "./out.jpg", "--model_path", "./models/starry_final.ckpt"])


if __name__ == "__main__":
    # Imports
    import argparse

    # Get command line arguments
    parser = argparse.ArgumentParser(description='Produces illustrations for the given text.')
    parser.add_argument('--input-file', type=str, required=False, help='Path to the text file.',
                        default=os.path.join(os.path.dirname(__file__), '../data/peter_rabbit.txt'))
    parser.add_argument('--output-dir', type=str, required=False, help='Path to the output directory.',
                        default=os.path.join(os.path.dirname(__file__), '../illustrated_books'))
    parser.add_argument('--style-model', type=str, required=False, help='Path to the style transfer model.',
                        default=os.path.join(os.path.dirname(__file__), '../deps/faststyle/models/starry_final.ckpt'))
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)
    style_model = os.path.abspath(args.style_model)

    illustrate(input_file, output_dir, style_model)
