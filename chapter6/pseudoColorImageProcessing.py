#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np


def intensitySlicing(img):
    height, width = img.shape
    result = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i, j] > 224:
                result[i, j] = [255, 255, 255]
            elif img[i, j] > 192:
                result[i, j] = [219, 226, 255]
            elif img[i, j] > 160:
                result[i, j] = [184, 197, 255]
            elif img[i, j] > 128:
                result[i, j] = [148, 168, 255]
            elif img[i, j] > 96:
                result[i, j] = [219, 226, 255]
            elif img[i, j] > 64:
                result[i, j] = [112, 138, 255]
            elif img[i, j] > 32:
                result[i, j] = [76, 109, 255]
            else:
                result[i, j] = [41, 80, 255]

    return result


def grayToColor(img):
    height, width = img.shape
    result = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            R = np.sin(img[i, j]) * 255
            G = np.sin(img[i, j] + np.pi / 3) * 255
            B = np.sin(img[i, j] + np.pi / 5) * 255
            result[i, j] = B, G, R
    return result


def pseudocolor1():
    img = cv2.imread("../images/yiqin_gray.jpg", 0)
    print(img)
    cv2.imshow("img", img)

    colorimg = intensitySlicing(img)
    cv2.imshow("intensitySlicing", colorimg)

    cv2.waitKey(0)
    return 0
def pseudocolor2():
    img = cv2.imread("../images/color.jpg", 0)
    print(img)
    cv2.imshow("img", img)

    colorimg = grayToColor(img)
    cv2.imshow("intensitySlicing", colorimg)

    cv2.waitKey(0)
    return 0

if __name__ == "__main__":
    # pseudocolor1()
    pseudocolor2()
