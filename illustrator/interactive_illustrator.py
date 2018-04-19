from __future__ import unicode_literals
import codecs
import cv2
import glob
from google_images_download import google_images_download
import img2pdf
import math
import matplotlib.font_manager
from natsort import natsorted
import os
from PIL import Image, ImageDraw, ImageFont
import shutil
import spacy

import vision
import deps.scrapeImages as scrape_images

import tensorflow as tf
import numpy as np
from deps.faststyle.im_transf_net import create_net
import deps.faststyle.utils as utils

# Module variables
standard_img_shape = (500, 500)
image_downloader = google_images_download.googleimagesdownload()

# Load English model
print('Loading nlp model...')
nlp = spacy.load('en_core_web_lg')

# Create the graph.
print('Creating style transfer network...')
with tf.variable_scope('img_t_net'):
    X = tf.placeholder(
        tf.float32,
        shape=(1, standard_img_shape[0], standard_img_shape[1], 3),
        name='input')
    Y = create_net(X, 'resize')  # resize or deconv


def get_font(font_name=None, font_size=16):
    # print(sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist])))

    font_path = None

    all_fonts = matplotlib.font_manager.findSystemFonts(
        fontpaths=None, fontext='ttf')

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
        font_manager = matplotlib.font_manager.FontManager(
            size=16, weight='normal')
        font_properties = matplotlib.font_manager.FontProperties(
            family=None,
            style=None,
            variant=None,
            weight=None,
            stretch=None,
            size=font_size,
            fname=None,
            _init=None)
        font_path = font_manager.findfont(
            font_properties,
            fontext='ttf',
            directory=None,
            fallback_to_default=True,
            rebuild_if_missing=True)

    return ImageFont.truetype(font=font_path, size=font_size)


def read_file(input_file):

    with codecs.open(input_file, "r", "utf-8") as f:
        text = f.read()

    pages = text.split("\n\n")

    return text, pages


# Adapted from https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent
def make_white_transparent(image, threshold=235):
    image = image.convert("RGBA")

    pixdata = image.load()

    width, height = image.size
    for y in range(height):
        for x in range(width):
            pixel = pixdata[x, y]
            average = sum(pixel[:3]) / float(len(pixel[:3]))

            if average > threshold:
                pixdata[x, y] = (255, 255, 255, 0)

    return image


def google_image_search(keywords,
                        output_file_name,
                        output_dir,
                        type="line-drawing"):
    image_downloader_arguments = {
        "keywords": keywords,
        "output_directory": output_dir,
        "limit": 1,
        "format": "jpg",
        "size": "medium",
        #"aspect_ratio": "wide",
        "type": type,
        #"usage_rights": "labled-for-noncommercial-reuse-with-modification",
        "metadata": True
    }

    # Download the first image corresponding to the keyword search
    image_downloader.download(image_downloader_arguments)

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
    image_path = google_image_search(keywords, "{0}".format(page_number),
                                     output_dir)

    # Return image path
    return image_path


def create_page_image(page_doc, noun_to_image_map, detector, page_number,
                      output_dir):

    nouns_dir = os.path.join(output_dir, 'nouns')

    keywords = []
    nouns = []

    for i, chunk in enumerate(page_doc.noun_chunks):
        # Get noun
        noun_token = chunk.root
        noun = noun_token.lemma_

        if noun == "-PRON-":
            continue

        if noun_token.ent_type_ == "PERSON":
            keywords.append(noun_token.ent_type_.lower())
        else:
            keywords.append(noun)

        nouns.append(noun)

        if noun not in noun_to_image_map:
            # Download image
            print("Downloading image...")
            keyword_search = chunk.text
            image_path = google_image_search(keyword_search, noun, nouns_dir)

            # Save noun and the path to the image
            noun_to_image_map[noun] = image_path
        else:
            print("Reusing noun")

    new_image = combine_images(keywords, nouns, noun_to_image_map, detector,
                               output_dir)
    image_path = os.path.join(output_dir, '{0}.jpg'.format(page_number))
    new_image.save(image_path)

    # Return the nouns and associated images for a page
    return image_path, noun_to_image_map


def combine_images(keywords, nouns, noun_to_image_map, detector, output_dir):
    keyword_string = ' '.join(keywords)
    image_path = google_image_search(
        keyword_string, "template", output_dir, type="photo")
    image = detector.load_image(image_path)
    boxes = detector.compute_bounding_boxes(image)

    reference_image = Image.open(image_path)

    width, height = reference_image.size
    width_ratio = width / 512
    height_ratio = height / 512

    if width < height:
        new_width = height
        new_height = height
    else:
        new_width = width
        new_height = width

    new_image = Image.new('RGB', (new_width, new_height), color='white')

    x_offset = int((new_width - width) / 2.0)
    y_offset = int((new_height - height) / 2.0)

    for noun in nouns:
        try:
            noun_image = Image.open(noun_to_image_map[noun])
        except:
            continue

        if noun in boxes and boxes[noun].size:
            box = boxes[noun][0]

            if box.size:
                box[0] *= width_ratio
                box[2] *= width_ratio
                box[1] *= height_ratio
                box[3] *= height_ratio

                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                box_area = box_width * box_height
                resized_image = resize_preserve_aspect_ratio_PIL(
                    noun_image, box_area)
                resized_image = make_white_transparent(resized_image)
                noun_image_width, noun_image_height = resized_image.size
                additional_x_offset = int((box_width - noun_image_width) / 2.0)
                additional_y_offset = int(
                    (box_height - noun_image_height) / 2.0)

                upper_left_x = int(box[0] + x_offset + additional_x_offset)
                upper_left_y = int(box[1] + y_offset + additional_y_offset)

                new_image.paste(
                    resized_image,
                    box=(upper_left_x, upper_left_y),
                    mask=resized_image)

    final_image = new_image.resize((768, 768))

    os.remove(image_path)
    return final_image


def expand_to_aspect_ratio(width, height, target_ratio=0.8):
    ratio = width / height

    if abs(target_ratio - ratio) < 1e-12:
        return width, height
    elif ratio < target_ratio:
        width = height * target_ratio
    else:
        height = width * target_ratio

    return int(width), int(height)


# https://stackoverflow.com/questions/33701929/how-to-resize-an-image-in-python-while-retaining-aspect-ratio-given-a-target-s/33702454
def resize_preserve_aspect_ratio_openCV(image, target_area):
    current_height, current_width = image.shape[:2]
    aspect_ratio = current_width / current_height

    new_height = math.sqrt(target_area / aspect_ratio)
    new_width = new_height * aspect_ratio

    new_image = cv2.resize(image, (new_width, new_height))
    return new_image


# https://stackoverflow.com/questions/33701929/how-to-resize-an-image-in-python-while-retaining-aspect-ratio-given-a-target-s/33702454
def resize_preserve_aspect_ratio_PIL(image, target_area):
    current_width, current_height = image.size
    aspect_ratio = current_width / current_height

    new_height = math.sqrt(target_area / aspect_ratio)
    new_width = new_height * aspect_ratio

    new_image = image.resize((int(new_width), int(new_height)))
    return new_image


def wrap_text(text, max_width, font):
    if font.getsize(text)[0] < max_width:
        return text
    else:
        multiline_text = list(text)

        start_of_new_line = 0

        while font.getsize(text[start_of_new_line:])[0] > max_width:
            a = start_of_new_line
            b = len(text) - 1

            while (b - a) > 1:
                c = (a + b) // 2
                current_width = font.getsize(
                    text[start_of_new_line:(c + 1)])[0]

                if current_width <= max_width:
                    a = c

                    if current_width == max_width:
                        break
                else:
                    b = c

            while text[a] != ' ':
                a -= 1

            multiline_text[a] = '\n'
            start_of_new_line = a + 1

        return ''.join(multiline_text)


def pad_bottom_of_image(img, min_padding, percentage=0.25):
    height, width = img.shape[:2]
    bottom_padding = max(min_padding, int(percentage * height))
    return cv2.copyMakeBorder(
        img,
        0,
        bottom_padding,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255))


# Adapted from https://github.com/ghwatson/faststyle/blob/master/stylize_image.py
def stylize_image(img, sess, content_target_resize=1.0):
    print('Stylizing image...')

    # Preprocess input image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    print('Done stylizing image.')

    return img_out


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
    extensions = ('*.jpg', '*.jpeg', '*.png',
                  '*.gif')  # the tuple of file types
    image_paths = []

    for extension in extensions:
        image_paths.extend(glob.glob(os.path.join(output_dir, extension)))

    image_paths = natsorted(image_paths)

    # multiple inputs (variant 2)
    with open(os.path.join(output_dir, "book.pdf"), "wb") as f:
        f.write(img2pdf.convert(image_paths))


def illustrate(input_file, output_dir, sess, detector, font):

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

        #image_path = one_google_image_per_page(page_doc, i, output_dir)

        image_path, noun_to_image_map = create_page_image(
            page_doc, noun_to_image_map, detector, i, output_dir)

        # Read image with PIL
        img = Image.open(image_path)
        width, height = img.size

        # Wrap text using PIL
        d = ImageDraw.Draw(img)
        multiline_text = wrap_text(page, int(0.9 * width), font)
        text_width, text_height = d.textsize(multiline_text, font=font)

        # Read the image with OpenCV
        img = cv2.imread(image_path)

        # Pad the image with OpenCV so there is room for text
        img = pad_bottom_of_image(img, int(1.25 * text_height))
        full_height = img.shape[0]

        # Stylize image with OpenCV
        img = stylize_image(img, sess)

        # Save the image with OpenCV
        cv2.imwrite(image_path, img)

        # Add the text of the page to the image using PIL
        text_position = compute_text_position(height, text_height, full_height,
                                              width)
        add_text_to_image(image_path, multiline_text, text_position, font)

    # Convert all the images to a pdf
    nouns_dir = os.path.join(output_dir, 'nouns')
    shutil.rmtree(nouns_dir, ignore_errors=True)
    convert_images_to_pdf(output_dir)


if __name__ == "__main__":
    # Imports
    import argparse

    # Get command line arguments
    parser = argparse.ArgumentParser(
        description='Produces illustrations for the given text.')
    parser.add_argument(
        '--input-file',
        type=str,
        required=False,
        help='Path to the text file.',
        default=os.path.join(
            os.path.dirname(__file__), '../data/peter_rabbit.txt'))
    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        help='Path to the output directory.',
        default=os.path.join(
            os.path.dirname(__file__), '../illustrated_books/peter_rabbit'))
    parser.add_argument(
        '--style-model',
        type=str,
        required=False,
        help='Path to the style transfer model.',
        default=os.path.join(
            os.path.dirname(__file__),
            '../deps/faststyle/models/candy_final.ckpt'))
    parser.add_argument(
        '--font',
        type=str,
        required=False,
        help='Font name.',
        default='Times New Roman')
    parser.add_argument(
        '--font-size', type=int, required=False, help='Font size.', default=18)
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)
    model_path = os.path.abspath(args.style_model)
    font = get_font(args.font, args.font_size)

    # Filter the input image.
    with tf.Session() as sess:
        print('Loading up style transfer model...')
        tf.train.Saver().restore(sess, model_path)

        print('Loading object dection model...')
        with vision.ObjectDetector() as detector:
            illustrate(input_file, output_dir, sess, detector, font)
