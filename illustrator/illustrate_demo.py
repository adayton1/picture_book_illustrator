from illustrate import *
import cv2
import os
import spacy
import sys
import numpy as np


def show_image(title, file_path):
    img = cv2.imread(file_path)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pil_to_cv2(pil_img):
    # temp = np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)
    cv2_img = np.array(pil_img)
    return cv2_img[:, :, ::-1].copy()


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
        '--style-model',
        type=str,
        required=False,
        help='Path to the style transfer model.',
        default=os.path.join(
            os.path.dirname(__file__),
            '../deps/faststyle/models/starry_final.ckpt'))
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

    # Output directories
    downloads_dir = os.path.join(output_dir, "downloads")
    pages_dir = os.path.join(output_dir, "pages")

    # Load models
    print("Loading spacy model...")
    nlp = spacy.load("en_core_web_lg")
    captioner = vision.ImageCaptioner()
    stylizer = stylize.Stylizer()
    detector = vision.ObjectDetector()

    # Save images for each noun and entity that is seen
    nouns_to_imgs = {}

    # Input loop
    pageNum = -1
    while True:
        pageNum += 1

        # Ask for input
        prompt = "Page {}".format(pageNum) if pageNum > 0 else "Title"
        text = input("\n{}: ".format(prompt)).strip()

        # Repeat prompt if no text is entered
        if text == "":
            pageNum -= 1
            continue

        # Check for stopping criterion
        if text.lower() in ["eof", "exit", "q", "quit", "the end"]:
            break

        # text = text.replace(",", "")
        # text = text.replace(";", ".")

        # Find images for the nouns and entities on a page
        nouns, entities, nouns_to_imgs, template_img_path = find_images_for_page(
            text, nouns_to_imgs, downloads_dir, nlp, captioner)

        # Display the images for each noun and entity on the page
        for noun in nouns:
            if noun in nouns_to_imgs:
                im = cv2.imread(nouns_to_imgs[noun])

                if im is not None:
                    cv2.imshow(noun, im)
                else:
                    response = input("\nCould not read image. Continue? (y/n)")
                    if response in ["y", "Y"]:
                        continue
                    else:
                        sys.exit(1)

            else:
                response = input("\nNo image associated with noun. Continue? (y/n)")
                if response in ["y", "Y"]:
                    continue
                else:
                    sys.exit(1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create an image in PIL format
        created_image_pil = create_image(nouns, entities, nouns_to_imgs,
                                         template_img_path, detector)

        # Convert to OpenCV format
        created_image_cv2 = pil_to_cv2(created_image_pil)

        # Get the path of the image
        composed_image_path = os.path.join(pages_dir, "{}.jpg".format(pageNum))
        cv2.imwrite(composed_image_path, created_image_cv2)
        show_image("composed image", composed_image_path)

        # Pad the bottom of the image before style transfer
        pad_bottom_of_images([composed_image_path])

        # Stylize and show the image
        stylize_images([composed_image_path], stylizer)
        show_image("Image after style transfer", composed_image_path)

        # Add text to the page and show the final page
        add_text_to_images([composed_image_path], [text], font)
        show_image("Final image for page", composed_image_path)

    print("Converting to PDF...")
    convert_images_to_pdf(pages_dir, output_dir)
