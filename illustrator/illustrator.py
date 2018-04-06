from __future__ import unicode_literals
import codecs
import cv2
import glob
import img2pdf
import matplotlib.font_manager
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


def get_font(font_name=None, font_size=16):
    # print(sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist])))

    font_path = None

    all_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    for font_file in all_fonts:
        font_file_name = os.path.basename(font_file)
        root, extension = os.path.splitext(font_file_name)

        if extension != ".ttf":
            continue
        else:
            if font_name == root:
                font_path = font_file
                break

    if not font_path:
        font_manager = matplotlib.font_manager.FontManager(size=16, weight='normal')
        font_properties = matplotlib.font_manager.FontProperties(family=None, style=None, variant=None,
                                                                 weight=None, stretch=None, size=font_size,
                                                                 fname=None, _init=None)
        font_path = font_manager.findfont(font_properties, fontext='ttf', directory=None,
                                          fallback_to_default=True, rebuild_if_missing=True)

    return ImageFont.truetype(font=font_path, size=font_size)


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


def wrap_text(text, max_width, font):
    # TODO: Make this a binary search

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

        current_width = font.getsize(text[start_of_new_line:])[0]

        if current_width > max_width:
            multiline_text[last_space_index] = '\n'

        return ''.join(multiline_text)
    else:
        return text





def pad_bottom_of_image(img, min_padding, percentage=0.15):
    height, width = img.shape[:2]
    bottom_padding = max(min_padding, int(percentage * height))
    return cv2.copyMakeBorder(img, 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))


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

    # Original dimensions are (height, width, channels)
    # The resize function expects (width, height)
    new_dim = (orig_dim[1], orig_dim[0])
    img_out = cv2.resize(img_out, new_dim)
    utils.imwrite(output_img_path, img_out)

    print('Done stylizing image.')


def compute_text_position(height, text_height, full_height, width):
    text_box_height = full_height - height
    start_height = int(height + ((text_box_height - text_height) / 2.0))
    start_width = int(0.05 * width)
    return start_width, start_height


def add_text_to_image(image_path, multiline_text, position, font):
    img = Image.open(image_path)

    d = ImageDraw.Draw(img)
    d.multiline_text(position, multiline_text, font=font, fill="black")

    img.save(image_path)


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


def illustrate(input_file, output_dir, sess, font):

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

        # Read image
        img = Image.open(image_path)
        width, height = img.size

        # Wrap text
        d = ImageDraw.Draw(img)
        multiline_text = wrap_text(page, int(0.9 * width), font)
        text_width, text_height = d.textsize(multiline_text, font=font)

        # Pad the image so there is room for text
        img = cv2.imread(image_path)
        img = pad_bottom_of_image(img, int(1.15 * text_height))
        full_height = img.shape[0]
        cv2.imwrite(image_path, img)

        # Stylize image
        stylize_image(image_path, image_path, sess)

        # Add the text of the page to the image
        text_position = compute_text_position(height, text_height, full_height, width)
        add_text_to_image(image_path, multiline_text, text_position, font)

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
    parser.add_argument('--font', type=str, required=False, help='Font name.',
                        default='Times New Roman')
    parser.add_argument('--font-size', type=int, required=False, help='Font size.',
                        default=16)
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)
    model_path = os.path.abspath(args.style_model)
    font = get_font(args.font, args.font_size)

    # Filter the input image.
    with tf.Session() as sess:
        print('Loading up model...')
        tf.train.Saver().restore(sess, model_path)
        illustrate(input_file, output_dir, sess, font)
