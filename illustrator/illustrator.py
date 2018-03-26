from __future__ import unicode_literals
import codecs
import os
import subprocess
import spacy
import cv2

import tensorflow as tf
import numpy as np
#from deps.faststyle.im_transf_net import create_net
import deps.faststyle.im_transf_net as im_transf_net
import deps.faststyle.utils as utils

# Load English model
nlp = spacy.load('en_core_web_lg')

standard_img_shape = (500, 500)

# Create the graph.
with tf.variable_scope('img_t_net'):
    X = tf.placeholder(tf.float32, shape=(1, standard_img_shape[0], standard_img_shape[1], 3), name='input')
    Y = im_transf_net.create_net(X, 'resize') # resize or deconv


def read_file(input_file):

    with codecs.open(input_file, "r", "utf-8") as f:
        text = f.read()

    pages = text.split("\n\n")

    return text, pages


def google_image_search(keywords, output_file_name, output_dir):
    # Download the first image corresponding to the keyword search
    subprocess.call(["googleimagesdownload", "-k", keywords, "-l", "1", "-o", output_dir, "-f", "jpg"])

    # Get the path to where the downloaded image was saved
    download_dir = os.path.join(output_dir, keywords)
    downloaded_file_name = subprocess.check_output(["ls", download_dir])[:-1]  # Throw away the \n character returned by ls
    downloaded_file_name = downloaded_file_name.decode("utf-8")
    downloaded_file_path = os.path.join(download_dir, downloaded_file_name)

    # Get the filename and extension
    filename, file_extension = os.path.splitext(downloaded_file_path)

    # Move to the destination
    destination = os.path.join(output_dir, output_file_name + file_extension)
    subprocess.call(["mv", downloaded_file_path, destination])

    # Clear out the temporary folder
    subprocess.call(["rm", "-r", download_dir])

    # Return the path to the saved file
    return destination


def one_google_image_per_page(page_doc, page_number, output_dir):
    nouns = []

    for chunk in page_doc.noun_chunks:
        noun = chunk.root

        if noun.lemma_ != "-PRON-":
            nouns.append(noun.lemma_)

    # Download image
    print("Downloading image...")
    keywords = " ".join(nouns)
    image_path = google_image_search(keywords, "{0}".format(page_number), output_dir)

    # Return image path
    return image_path


def multiple_google_images_per_page(page_doc, page_number, output_dir):
    nouns = {}

    for i, chunk in enumerate(page_doc.noun_chunks):
        # Get noun
        noun = chunk.root
        noun_text = noun.lemma_

        if noun_text == "-PRON-":
            continue

        # Download image
        print("Downloading image...")
        keywords = chunk.text
        output_file_name = "{0}_{1}".format(page_number, i)
        image_path = google_image_search(keywords, output_file_name, output_dir)

        # Save noun and the path to the image
        nouns[noun_text] = image_path

    # Return the nouns and associated images for a page
    return nouns


# Adapted from https://github.com/ghwatson/faststyle/blob/master/stylize_image.py
def stylize_image(input_img_path, output_img_path, sess, content_target_resize=1.0):
    print('Stylizing image...')

    # Read + preprocess input image.
    img = utils.imread(input_img_path)
    #img = utils.imresize(img, content_target_resize)
    orig_dim = img.shape
    img = cv2.resize(img, standard_img_shape)
    img_4d = img[np.newaxis, :]

    # Saver used to restore the model to the session.
    print('Evaluating...')
    img_out = sess.run(Y, feed_dict={X: img_4d})

    # Postprocess + save the output image.
    print('sSaving image...')
    img_out = np.squeeze(img_out)
    img_out = cv2.resize(img_out, orig_dim[:2])
    utils.imwrite(output_img_path, img_out)

    print('Done stylizing image.')


def illustrate(input_file, output_dir, sess):

    print("Reading input file...")
    text, pages = read_file(input_file)

    # Process the whole doc
    # doc = nlp(text)

    # Iterate through each page
    for i, page in enumerate(pages):
        print("\n\nIllustrating page {0}...".format(i))

        print("Natural language processing...")
        page_doc = nlp(page)

        image_path = one_google_image_per_page(page_doc, i, output_dir)

        #nounToImageMap = multiple_google_images_per_page(page_doc, i, os.path.join(output_dir, "page{0}".format(i)))

        # Stylize image
        stylize_image(image_path, image_path, sess)
        #subprocess.call(["python", "stylize_image.py", "--input_img_path", image_path, "--output_img_path",
        #                image_path, "--model_path", style_model])


if __name__ == "__main__":
    # Imports
    import argparse

    # Get command line arguments
    parser = argparse.ArgumentParser(description='Produces illustrations for the given text.')
    parser.add_argument('--input-file', type=str, required=False, help='Path to the text file.',
                        default=os.path.join(os.path.dirname(__file__), '../data/peter_rabbit.txt'))
    parser.add_argument('--output-dir', type=str, required=False, help='Path to the output directory.',
                        default=os.path.join(os.path.dirname(__file__), '../illustrated_books/peter_rabbit'))
    parser.add_argument('--style-model', type=str, required=False, help='Path to the style transfer model.',
                        default=os.path.join(os.path.dirname(__file__), 'deps/faststyle/models/starry_final.ckpt'))
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)
    model_path = os.path.abspath(args.style_model)

    # Filter the input image.
    with tf.Session() as sess:
        print('Loading up model...')
        tf.train.Saver().restore(sess, model_path)
        illustrate(input_file, output_dir, sess)
