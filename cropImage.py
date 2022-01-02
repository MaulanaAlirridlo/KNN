import numpy as np
import cv2
from os import listdir, walk


def croppingImage(path):
    # Read image
    img = cv2.imread(path)

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([120, 120, 120])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)
    cv2.imshow("1", thresh)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("kernel", kernel)
    cv2.imshow("2", morph)

    # invert morp image
    mask = 255 - morph
    cv2.imshow("3", mask)

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("4", result)

    # cv2.imwrite(path, result)

# for root, subdir, file in walk("./images/training/busuk"):
#     for file in file :
#         croppingImage(root+"/"+file)

    cv2.waitKey()

croppingImage("./backup/a/1.jpg")
