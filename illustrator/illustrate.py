from __future__ import unicode_literals
import codecs
import cv2
import errno
import glob
import img2pdf
import math
import matplotlib.font_manager
from natsort import natsorted
import os
from PIL import Image, ImageDraw, ImageFont
import random
import shutil
import spacy

import utils
utils.extend_syspath(['./'])  # HACK

from deps import google_images_download
import stylize
import vision

# Module variables
image_downloader = google_images_download.googleimagesdownload()
page_width = 480   # 5 inches
page_height = 672  # 7 inches

image_width = 480  # 5 inches
image_height = 480 # 5 inches


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

    pages = text.split("\n")

    return text, pages


def google_image_search(keywords,
                        output_file_name,
                        output_dir,
                        limit=1,
                        image_size="large",
                        type="line-drawing"):
    image_downloader_arguments = {
        "keywords": keywords,
        "output_directory": output_dir,
        "limit": limit,
        "format": "jpg",
        "size": image_size,
        "type": type,
        #"aspect_ratio": "wide",
        #"usage_rights": "labled-for-noncommercial-reuse-with-modification",
        # "metadata": False
    }

    # Download the images corresponding to the keyword search
    image_downloader.download(image_downloader_arguments)

    # Get the path to where the downloaded image was saved
    download_dir = os.path.join(output_dir, keywords)

    if limit == 1:
        downloaded_file_name = os.listdir(download_dir)[0]
        # downloaded_file_name = downloaded_file_name.decode("utf-8")
        downloaded_file_path = os.path.join(download_dir, downloaded_file_name)

        # Get the filename and extension
        filename, file_extension = os.path.splitext(downloaded_file_path)

        # Move to the destination
        destination = os.path.join(output_dir,
                                   output_file_name + file_extension)
        shutil.move(downloaded_file_path, destination)

        # Clear out the temporary folder
        os.rmdir(download_dir)

        # Return the path to the saved file
        return destination
    else:
        parent_dir, child_dir = os.path.split(download_dir)
        downloaded_files = os.listdir(download_dir)

        file_paths = []

        for file_name in downloaded_files:
            file_path = os.path.join(download_dir, file_name)
            destination = os.path.join(parent_dir, file_name)
            shutil.move(file_path, destination)
            file_paths.append(destination)

        # Clear out the temporary folder
        os.rmdir(download_dir)

        # Return the paths to the saved files
        return file_paths


def find_noun_images(page_doc, output_dir):
    images = {}
    entities = {}

    # Find images corresponding to the remainder of the nouns
    print("Downloading noun images...")
    for chunk in page_doc.noun_chunks:
        # Get noun
        noun_token = chunk.root
        noun = noun_token.lemma_

        # Ignore pronouns
        if noun == "-PRON-":
            continue

        # Save entities
        if noun_token.ent_type_:
            entities[noun] = noun_token.ent_type_.lower()

        # Ignore nouns that have already been found
        if noun in images:
            continue

        # Download image
        keyword_search = chunk.text
        image_path = google_image_search(keyword_search, noun, output_dir)

        # Save noun and the path to the image
        images[noun] = image_path

    return images, entities


def find_template_images(page_doc, output_dir, num_images=5):
    nouns = []

    for i, chunk in enumerate(page_doc.noun_chunks):
        # Get noun
        noun_token = chunk.root
        noun = noun_token.lemma_

        if noun == "-PRON-":
            continue

        nouns.append(noun)

    # keyword_string = ' '.join(keywords)
    keyword_string = page_doc.text
    file_paths = google_image_search(
        keyword_string,
        "template",
        output_dir,
        limit=num_images,
        image_size="large",
        type="photo")

    return nouns, file_paths


def find_best_image(original_text, images, nlp, captioner):
    original_text_doc = nlp(original_text)
    captions = captioner.generate(images)

    highest_similarity = -1
    best_images = images

    for i, caption in enumerate(captions):
        similarity = original_text_doc.similarity(nlp(caption))

        if similarity < highest_similarity:
            continue
        elif similarity == highest_similarity:
            best_images.append(images[i])
        else:
            highest_similarity = similarity
            best_images = [images[i]]

    return random.choice(best_images)


def find_images_for_page(text, noun_to_image_map, output_dir, nlp=None, captioner=None):
    # Load English model
    if nlp is None:
        print("Loading nlp model...")
        nlp = spacy.load("en_core_web_lg")

    # Natural language processing
    page_doc = nlp(text)

    nouns = []
    entities = {}

    # Find images corresponding to the remainder of the nouns
    print("Downloading noun images...")
    for chunk in page_doc.noun_chunks:
        # Get noun
        noun_token = chunk.root
        noun = noun_token.lemma_

        # Ignore pronouns
        if noun == "-PRON-":
            continue

        # Save entities
        if noun not in entities and noun_token.ent_type_:
            entities[noun] = noun_token.ent_type_.lower()

        nouns.append(noun)

        # Ignore nouns that have already been found
        if noun in noun_to_image_map:
            continue

        # Download image
        keyword_search = chunk.text
        image_path = google_image_search(keyword_search, noun, output_dir)

        # Save noun and the path to the image
        noun_to_image_map[noun] = image_path

    # Loading image caption module
    if captioner is None:
        captioner = vision.ImageCaptioner()

    keyword_string = page_doc.text
    possible_template_images = google_image_search(
        keyword_string,
        "template",
        os.path.join(output_dir, "templates"),
        limit=10,
        image_size="large",
        type="photo")

    print("Captioning template images and choosing the best...")
    best_template_path = find_best_image(text, possible_template_images,
                                         nlp, captioner)

    return nouns, entities, noun_to_image_map, best_template_path


def find_images_for_full_text(text, pages, output_dir, nlp=None, captioner=None):
    # Load English model
    if nlp is None:
        print("Loading nlp model...")
        nlp = spacy.load("en_core_web_lg")

    # Find images of entities and nouns
    images, entities = find_noun_images(nlp(text), os.path.join(output_dir, "nouns"))

    nouns = []
    template_images = []

    # Loading image caption module
    if captioner is None:
        captioner = vision.ImageCaptioner()

    # Find nouns and keywords on each page
    print("Downloading template images...")
    for i, page in enumerate(pages):
        doc = nlp(page)

        page_nouns, possible_template_images = find_template_images(
            doc,
            os.path.join(output_dir, "templates{0}".format(i)),
            num_images=10)
        nouns.append(page_nouns)

        print("Captioning template images and choosing the best...")
        best_template_path = find_best_image(page, possible_template_images,
                                             nlp, captioner)

        # TODO: Add file extension to destination
        destination = os.path.join(output_dir, "template{0}".format(i))
        shutil.copy(best_template_path, destination)

        template_images.append(destination)

    return nouns, entities, images, template_images


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

    new_image = image.resize((int(new_width), int(new_height)), Image.BICUBIC)
    return new_image


def create_image(nouns, entities, images, template_image_path, detector=None):
    if detector is None:
        detector = vision.ObjectDetector()

    template_image = detector.load_image(template_image_path)
    boxes = detector.compute_bounding_boxes(template_image)

    reference_image = Image.open(template_image_path)

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
            noun_image = Image.open(images[noun])
        except:
            print("Could not open image: {0}".format(images[noun]))
            continue

        if noun in entities:
            noun = entities[noun]

        if noun in boxes and boxes[noun].size:
            if noun in boxes:
                box = boxes[noun][0]
            else:
                box = boxes[entities[noun]].size

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
                additional_x_offset = int(
                    (box_width - noun_image_width) / 2.0)
                additional_y_offset = int(
                    (box_height - noun_image_height) / 2.0)

                upper_left_x = int(box[0] + x_offset + additional_x_offset)
                upper_left_y = int(box[1] + y_offset + additional_y_offset)

                new_image.paste(
                    resized_image,
                    box=(upper_left_x, upper_left_y),
                    mask=resized_image)

        else:
            # Choose random box
            box = [0] * 4

            box_width = random.randint(
                int(0.3 * new_width), int(0.5 * new_width))
            box_height = random.randint(
                int(0.3 * new_width), int(0.5 * new_height))
            box_area = box_width * box_height

            box[0] = random.randint(0, new_width - box_width)
            box[1] = random.randint(0, new_height - box_height)
            box[2] = box[0] + box_width
            box[3] = box[1] + box_width

            resized_image = resize_preserve_aspect_ratio_PIL(
                noun_image, box_area)
            resized_image = make_white_transparent(resized_image)
            noun_image_width, noun_image_height = resized_image.size
            additional_x_offset = int((box_width - noun_image_width) / 2.0)
            additional_y_offset = int(
                (box_height - noun_image_height) / 2.0)

            upper_left_x = int(box[0] + additional_x_offset)
            upper_left_y = int(box[1] + additional_y_offset)

            new_image.paste(
                resized_image,
                box=(upper_left_x, upper_left_y),
                mask=resized_image)

    final_image = new_image.resize((image_width, image_height))
    return final_image


def create_images(nouns, entities, images, template_images, output_dir, detector=None):
    created_images = []

    pages_dir = os.path.join(output_dir, "pages")

    try:
        os.makedirs(pages_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(pages_dir):
            pass
        else:
            raise

    # Load object dection model if necessary
    if detector is None:
        detector = vision.ObjectDetector()

    print("Creating images...")

    for i, template_path in enumerate(template_images):
        print("Creating image for page {0}...".format(i + 1))
        created_image = create_image(nouns[i], entities, images, template_path, detector)
        destination = os.path.join(pages_dir, "{0}.jpg".format(i))
        created_image.save(destination)
        created_images.append(destination)

    return created_images


def pad_bottom_of_images(image_paths, percentage=0.25):
    print("Padding images to make room for story text...")

    for image_path in image_paths:
        # Read image
        img = cv2.imread(image_path)

        # Calculate padding
        height, width = img.shape[:2]
        bottom_padding = int(percentage * height)

        # Pad image
        padded_img = cv2.copyMakeBorder(
            img,
            0,
            bottom_padding,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255))

        # Save image
        cv2.imwrite(image_path, padded_img)


def stylize_images(image_paths, stylizer=None):
    if stylizer is None:
        stylizer = stylize.Stylizer()
    print("Applying style transfer to images...")
    for i, image_path in enumerate(image_paths):
        print("Stylizing page {0}".format(i + 1))
        img_out = stylizer.stylize_image(image_path)
        with open(image_path, 'wb') as f:
            f.write(img_out)


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


def compute_text_position(full_height, full_width, text_height, image_height):
    text_box_height = full_height - image_height
    start_height = int(image_height + ((text_box_height - text_height) / 2.0))
    start_width = int(0.05 * full_width)
    return start_width, start_height


def add_text_to_images(image_paths, pages, font):
    print("Adding story text to images...")
    for i, image_path in enumerate(image_paths):
        # Read image with PIL
        img = Image.open(image_path)
        d = ImageDraw.Draw(img)

        # Wrap text using PIL
        width, height = img.size
        multiline_text = wrap_text(pages[i], int(0.9 * width), font)
        text_width, text_height = d.textsize(multiline_text, font=font)

        # Add the text of the page to the image using PIL
        text_position = compute_text_position(height, width, text_height,
                                              image_height)
        d.multiline_text(
            text_position, multiline_text, font=font, fill="white")

        # Save the image
        img.save(image_path)


def convert_images_to_pdf(input_dir, output_dir):
    print("Converting images to pdf...")
    # https://stackoverflow.com/questions/4568580/python-glob-multiple-filetypes
    extensions = ('*.jpg', '*.jpeg', '*.png',
                  '*.gif')  # the tuple of file types
    image_paths = []

    for extension in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, extension)))

    image_paths = natsorted(image_paths)

    # multiple inputs (variant 2)
    with open(os.path.join(output_dir, "book.pdf"), "wb") as f:
        f.write(img2pdf.convert(image_paths))


def illustrate(input_file,
               output_dir,
               font,
               remove_downloads=False):

    print("Reading input file...")
    text, pages = read_file(input_file)

    downloads_dir = os.path.join(output_dir, "downloads")
    nouns, entities, images, template_images = find_images_for_full_text(text, pages, downloads_dir)
    image_paths = create_images(nouns, entities, images, template_images, output_dir)
    pad_bottom_of_images(image_paths)
    stylize_images(image_paths)
    add_text_to_images(image_paths, pages, font)
    convert_images_to_pdf(os.path.join(output_dir, "pages"), output_dir)

    # Remove individual images
    if remove_downloads:
        shutil.rmtree(downloads_dir, ignore_errors=True)


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
            os.path.dirname(__file__), '../data/object_detection_story.txt'))
    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        help='Path to the output directory.',
        default=os.path.join(
            os.path.dirname(__file__),
            '../illustrated_books/object_detection'))
    parser.add_argument(
        '--font',
        type=str,
        required=False,
        help='Font name.',
        default='Times New Roman')
    parser.add_argument(
        '--font-size', type=int, required=False, help='Font size.', default=32)
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)
    font = get_font(args.font, args.font_size)

    illustrate(input_file, output_dir, font)
