from __future__ import unicode_literals
import codecs
import cv2
import glob
import img2pdf
from natsort import natsorted
import os
from PIL import Image, ImageDraw, ImageFont
import shutil
import subprocess
import spacy

import deps.scrapeImages as scrape_images

import tensorflow as tf
import numpy as np
from deps.faststyle.im_transf_net import create_net
import deps.faststyle.utils as utils

# Load English model
nlp = spacy.load('en_core_web_lg')

standard_img_shape = (500, 500)

# Create the graph.
with tf.variable_scope('img_t_net'):
    X = tf.placeholder(tf.float32, shape=(1, standard_img_shape[0], standard_img_shape[1], 3), name='input')
    Y = create_net(X, 'resize') # resize or deconv


def read_file(input_file):

    with codecs.open(input_file, "r", "utf-8") as f:
        text = f.read()

    pages = text.split("\n\n")

    return text, pages


def google_image_search(keywords, output_file_name, output_dir):
    # Download the first image corresponding to the keyword search
    subprocess.call(["googleimagesdownload",
                     "-k", keywords,
                     "-l", "1",
                     "-o", output_dir,
                     "-f", "jpg",
                     #"-r", "labled-for-noncommercial-reuse-with-modification",
                     "-s", "medium",
                     #"-a", "wide",
                     #"-t", "clip-art",
                     "-m"])

    # Get the path to where the downloaded image was saved
    download_dir = os.path.join(output_dir, keywords)
    downloaded_file_name = os.listdir(download_dir)[0]
    # downloaded_file_name = downloaded_file_name.decode("utf-8")
    downloaded_file_path = os.path.join(download_dir, downloaded_file_name)

    # Get the filename and extension
    filename, file_extension = os.path.splitext(downloaded_file_path)

    # Move to the destination
    destination = os.path.join(output_dir, output_file_name + file_extension)
    shutil.move(downloaded_file_path, destination)

    # Clear out the temporary folder
    os.rmdir(download_dir)

    # Return the path to the saved file
    return destination


def scrape_google_images(keywords, output_file_name, output_dir, num_images=1):
    # Download the first image corresponding to the keyword search
    scrape_images.run(keywords, output_dir, num_images)

    # Get the path to where the downloaded image was saved
    downloaded_file_name = os.listdir(output_dir)[0]
    # downloaded_file_name = downloaded_file_name.decode("utf-8")
    downloaded_file_path = os.path.join(output_dir, downloaded_file_name)

    # Get the filename and extension
    filename, file_extension = os.path.splitext(downloaded_file_path)

    # Move to the destination
    destination = os.path.join(output_dir, output_file_name + file_extension)
    shutil.move(downloaded_file_path, destination)

    # Return the path to the saved file
    return destination


def one_google_image_per_page(page_doc, page_number, output_dir):
    nouns = []

    for chunk in page_doc.noun_chunks:
        noun = chunk.root

        if noun.lemma_ != "-PRON-":
            if noun.ent_type_ == "PERSON":
                nouns.append(noun.ent_type_.lower())
            else:
                nouns.append(noun.lemma_)

    # Download image
    print("Downloading image...")
    keywords = " ".join(nouns)
    image_path = google_image_search(keywords, "{0}".format(page_number), output_dir)

    # Return image path
    return image_path


def multiple_google_images_per_page(noun_to_image_map, page_doc, page_number, output_dir):

    for i, chunk in enumerate(page_doc.noun_chunks):
        # Get noun
        noun = chunk.root
        noun_text = noun.lemma_

        if noun_text == "-PRON-":
            continue

        if noun_text not in noun_to_image_map:
            # Download image
            print("Downloading image...")
            keywords = chunk.text
            output_file_name = "{0}_{1}".format(page_number, i)

            image_path = scrape_google_images(keywords, output_file_name, output_dir, 1)

            #image_path = google_image_search(keywords, output_file_name, output_dir)

            # Save noun and the path to the image
            noun_to_image_map[noun_text] = image_path
        else:
            print("Reusing noun")

    # Return the nouns and associated images for a page
    return noun_to_image_map


def pad_bottom_of_image(image_path, percentage=0.15):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    bottom_padding = int(percentage * height)
    img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(image_path, img)


# Adapted from https://github.com/ghwatson/faststyle/blob/master/stylize_image.py
def stylize_image(input_img_path, output_img_path, sess, content_target_resize=1.0):
    print('Stylizing image...')

    # Read + preprocess input image.
    img = utils.imread(input_img_path)
    img = utils.imresize(img, content_target_resize)
    orig_dim = img.shape
    img = cv2.resize(img, standard_img_shape)
    img_4d = img[np.newaxis, :]

    print('Evaluating...')
    img_out = sess.run(Y, feed_dict={X: img_4d})

    # Postprocess + save the output image.
    print('Saving image...')
    img_out = np.squeeze(img_out)
    img_out = cv2.resize(img_out, orig_dim[:2])
    utils.imwrite(output_img_path, img_out)

    print('Done stylizing image.')


def split_text(text, max_width, font):
    width = font.getsize(text)[0]

    if width > max_width:
        multiline_text = list(text)

        start_of_new_line = 0
        last_space_index = 0

        for i, char in enumerate(text):
            if char == ' ':
                current_width = font.getsize(text[start_of_new_line:i])[0]

                if current_width > max_width:
                    multiline_text[last_space_index] = '\n'
                    start_of_new_line = last_space_index + 1

                    remaining_width = font.getsize(text[start_of_new_line:])[0]

                    if remaining_width <= max_width:
                        return ''.join(multiline_text)
                else:
                    last_space_index = i

        return ''.join(multiline_text)
    else:
        return text


def add_text_to_image(input_img_path, text, percentage=0.15):
    img = Image.open(input_img_path)
    width, height = img.size

    text_box_start_height = height / (1.0 + percentage)
    text_box_height = height - text_box_start_height

    text_start_width = int(0.05 * width)
    text_start_height = int(text_box_start_height + (0.1 * text_box_height))

    #fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
    font = ImageFont.load_default()
    multiline_text = split_text(text, 0.9 * width, font)

    d = ImageDraw.Draw(img)
    d.multiline_text((text_start_width, text_start_height), multiline_text, font=font, fill="black")

    img.save(input_img_path)


def convert_images_to_pdf(output_dir):
    # https://stackoverflow.com/questions/4568580/python-glob-multiple-filetypes
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif')  # the tuple of file types
    image_paths = []

    for extension in extensions:
        image_paths.extend(glob.glob(os.path.join(output_dir, extension)))

    image_paths = natsorted(image_paths)

    # multiple inputs (variant 2)
    with open(os.path.join(output_dir, "book.pdf"), "wb") as f:
        f.write(img2pdf.convert(image_paths))


def illustrate(input_file, output_dir, sess):

    print("Reading input file...")
    text, pages = read_file(input_file)

    # Process the whole doc
    # doc = nlp(text)

    noun_to_image_map = {}

    # Iterate through each page
    for i, page in enumerate(pages):
        print("\n\nIllustrating page {0}...".format(i))

        print("Natural language processing...")
        page_doc = nlp(page)

        image_path = one_google_image_per_page(page_doc, i, output_dir)

        # noun_to_image_map = multiple_google_images_per_page(noun_to_image_map, page_doc, i,
        #                                                     os.path.join(output_dir, "page{0}".format(i)))

        # Pad the image so there is room for text
        pad_bottom_of_image(image_path)

        # Stylize image
        stylize_image(image_path, image_path, sess)

        # Add the text of the page to the image
        add_text_to_image(image_path, page)

    # Convert all the images to a pdf
    convert_images_to_pdf(output_dir)


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
                        default=os.path.join(os.path.dirname(__file__), '../deps/faststyle/models/starry_final.ckpt'))
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)
    model_path = os.path.abspath(args.style_model)

    # Filter the input image.
    with tf.Session() as sess:
        print('Loading up model...')
        tf.train.Saver().restore(sess, model_path)
        illustrate(input_file, output_dir, sess)
