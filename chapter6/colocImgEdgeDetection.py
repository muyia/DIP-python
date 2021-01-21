#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

def edgeDetection(img, sobel):

    height, width, channels = img.shape
    filter_size = len(sobel)
    n = int((filter_size - 1) / 2)
    img_edge = np.zeros((height, width), np.uint8)
    for i in range(n, height - n):
        for j in range(n, width - n):
            temp =0
            for k in range(channels):
                temp += np.sum(sobel * img[i - n:i + n + 1, j - n:j + n + 1, k])
            img_edge[i, j]=np.clip(temp, 0, 255)
    return img_edge

def edgeDetection1(img, sobel):

    height, width, channels = img.shape
    filter_size = len(sobel)
    n = int((filter_size - 1) / 2)
    img_edge = np.zeros((height, width, channels), np.uint8)
    for i in range(n, height - n):
        for j in range(n, width - n):
            for k in range(channels):
                temp = np.sum(sobel * img[i - n:i + n + 1, j - n:j + n + 1, k])
                img_edge[i, j, k]=np.clip(temp, 0, 255)
    return img_edge




lapla90 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]  # 90°增量
sobel = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]  # 45°增量

sobel1 = [[-2, -2, 0],
           [-2, 0, 2],
           [0, 2, 2]]


img = cv2.imread("../images/lena.jpg")
cv2.imshow("image", img)

imgedge = edgeDetection(img, sobel)
cv2.imshow("sobel", imgedge)

cv2.waitKey(0)