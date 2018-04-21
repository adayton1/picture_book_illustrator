from illustrate import *
import cv2
import os
import spacy
import numpy as np

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

    # load models
    nlp = spacy.load("en_core_web_lg")
    captioner = vision.ImageCaptioner()
    stylizer = stylize.Stylizer()
    detector = vision.ObjectDetector()

    nouns_to_imgs = {}

    # input loop
    pageNum = -1
    while True:
        pageNum += 1
        prompt = "Page {}".format(pageNum) if pageNum > 0 else "Title"
        text = input("\n{}: ".format(prompt)).strip()
        if text == "":
            pageNum -= 1
            continue
        if text.lower() == "eof" \
         or text.lower() == "exit" \
         or text.lower() == "quit" \
         or text.lower() == "the end":
            break

        text = text.replace(",", "")
        text = text.replace(";", ".")

        print("(BEFORE) NOUNS TO IMGS:", nouns_to_imgs)
        downloads_dir = os.path.join(output_dir, "downloads")
        nouns, entities, nouns_to_imgs, template_img_path = find_images_for_page(
            text, nouns_to_imgs, downloads_dir, nlp, captioner)
        for noun, impath in nouns_to_imgs.items():
            im = cv2.imread(impath)
            cv2.imshow(noun, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("(AFTER) NOUNS TO IMGS:", nouns_to_imgs, nouns)

        created_image_pil = create_image(nouns, entities, nouns_to_imgs,
                                         template_img_path, detector)

        created_image = np.array(created_image_pil)
        created_image = created_image[:, :, ::-1].copy()

        composed_image_path = os.path.join(output_dir, "pages",
                                           "{}.jpg".format(pageNum))
        print("COMPOSED IMAGE PATH:", composed_image_path)
        cv2.imwrite(composed_image_path, created_image)
        cv2.imshow("composed image", created_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        pad_bottom_of_images([composed_image_path])

        stylize_images([composed_image_path], stylizer)

        add_text_to_images([composed_image_path], [text], font)

    print("Converting to PDF")
    convert_images_to_pdf(os.path.join(output_dir, "pages"), output_dir)
