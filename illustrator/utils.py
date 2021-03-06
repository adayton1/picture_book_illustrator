# Source: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# Some modifications made to save the images

# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import os
import sys

# HACK
project_root = os.path.abspath(os.path.join(sys.path[0], os.pardir))
sys.path.append(project_root)


# HACK
def extend_syspath(paths):
    for path in paths:
        sys.path.append(os.path.abspath(os.path.join(project_root, path)))


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities;
    image = cv2.equalizeHist(image)
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def detect_edges(image):
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny edge detection using a wide threshold, tight threshold, and automatically determined threshold
    # wide = cv2.Canny(blurred, 10, 200)
    # tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)
    return auto


def detect_edges_2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 75, 75)
    edges = auto_canny(blur)
    return edges


def detect_edges_3(image):
    #image_height = 720
    #image_width = 1280
    #resized_image = cv2.resize(image, (image_width, image_height))
    cv2.imshow("Original", image)
    dst2 = cv2.stylization(image, sigma_s=60, sigma_r=0.07)
    cv2.imshow("Testing", dst2)
    dst = cv2.edgePreservingFilter(image, flags=2, sigma_s=60, sigma_r=0.4)
    cv2.imshow("Filtered", dst)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Filtered Gray", gray)
    edges = auto_canny(gray)
    cv2.imshow("Filtered Gray Edges", edges)
    cv2.waitKey(0)
    return edges


def convert_to_sketch(image):
    grayscale_image, _ = cv2.pencilSketch(
        image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return grayscale_image


def process_image(image_file_path, destination_dir, show_image):
    # load the image
    image = cv2.imread(image_file_path)

    # convert to edges with Gaussian blurring
    edges = detect_edges(image)

    # save the edge detection image
    (head, tail) = os.path.split(image_file_path)
    destination_file_path = os.path.join(destination_dir, "edges_" + tail)
    cv2.imwrite(destination_file_path, edges)

    # convert to edges to bilateral blurring
    edges2 = detect_edges_2(image)

    # save the edge detection image
    destination_file_path = os.path.join(destination_dir, "edges2_" + tail)
    cv2.imwrite(destination_file_path, edges2)

    # convert to edges to bilateral blurring
    edges3 = detect_edges_3(image)

    # save the edge detection image
    destination_file_path = os.path.join(destination_dir, "edges3_" + tail)
    cv2.imwrite(destination_file_path, edges3)

    # convert to sketch
    sketch = convert_to_sketch(image)

    # save the sketch
    destination_file_path = os.path.join(destination_dir, "sketch_" + tail)
    cv2.imwrite(destination_file_path, sketch)

    # show the images
    if show_image:
        cv2.imshow("Original", image)
        cv2.imshow("Edges", edges)
        cv2.imshow("Edges2", edges2)
        cv2.imshow("Sketch", sketch)
        # cv2.imshow("Edges", np.hstack([wide, tight, auto]))
        cv2.waitKey(0)


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--images",
        required=True,
        help="path to input dataset of images")
    ap.add_argument(
        "-d", "--destination", required=True, help="path to save edge images")
    ap.add_argument(
        "-s", "--show", action="store_true", help="show each image")
    args = vars(ap.parse_args())

    images = args["images"]
    destination_dir = args["destination"]
    show_images = args["show"]

    if os.path.isfile(images):
        process_image(images, destination_dir, show_images)
    elif os.path.isdir(images):
        # loop over the images in the directory
        for image_path in glob.glob(images + "/*.jpg"):
            process_image(image_path, destination_dir, show_images)
    else:
        raise ValueError("{0} is not a valid file or directory".format(images))


if __name__ == "__main__":
    main()
