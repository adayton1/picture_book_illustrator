# Source: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# Some modifications made to save the images

# import the necessary packages
import numpy as np
import argparse
import glob
import cv2


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


def convert_to_sketch(image):
	grayscale_image, _ = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
	return grayscale_image


def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--images", required=True, help="path to input dataset of images")
	ap.add_argument("-d", "--destination", required=True, help="path to save edge images")
	ap.add_argument("-s", "--show", action="store_true", help="show each image")
	args = vars(ap.parse_args())

	# loop over the images
	for imagePath in glob.glob(args["images"] + "/*.jpg"):
		image = cv2.imread(imagePath)
		auto = load_image(imagePath)

		# save the image
		filename = imagePath[len(args["images"]) + 1:]
		destination_path = args["destination"] + "/" + filename
		cv2.imwrite(destination_path, auto)

		# show the images
		if args["show"]:
			cv2.imshow("Original", image)
			cv2.imshow("Grayscale v. Edges", np.hstack([gray, auto]))
			# cv2.imshow("Edges", np.hstack([wide, tight, auto]))
			cv2.waitKey(0)


if __name__ == 'main':
	main()
