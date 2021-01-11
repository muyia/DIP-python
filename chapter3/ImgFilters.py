#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import skimage
from skimage import util

def average_filter(img, filter_size):
    img_filter = np.copy(img)
    print(img_filter[1, 1])
    filter_template = np.ones((filter_size, filter_size))
    n = int((filter_size - 1) / 2)
    height, width, channels = img_filter.shape
    for i in range(n, height - n):
        for j in range(n, width - n):
            for k in range(channels):
                print("test:",img_filter[i - n:i + n + 1, j - n:j + n + 1, k])
                img_filter[i, j, k] = np.sum(filter_template * img_filter[i - n:i + n + 1, j - n:j + n + 1, k]) / (
                filter_size ** 2)
    return img_filter


def median_filter(img, filter_size):
    img_filter = np.copy(img)
    n = int((filter_size - 1) / 2)
    mediannum = int((filter_size ** 2 - 1) / 2)
    height, width, channels = img_filter.shape
    for i in range(n, height - n):
        for j in range(n, width - n):
            for k in range(channels):
                filter_list = []
                for ii in range(filter_size):
                    for jj in range(filter_size):
                        filter_list.append(img[i - n + ii, j - n + jj, k])
                filter_list.sort()
                img_filter[i, j, k] = filter_list[mediannum]
    return img_filter


img = cv2.imread('images/123.jpg')
cv2.imshow("img", img)
img_noise=util.random_noise(img,"salt")#添加椒盐噪声
cv2.imshow("salt noise image",img_noise)

#均值滤波
img_ave_fil_3=average_filter(img_noise,3)
cv2.imshow("img aveerage filter",img_ave_fil_3)
# img_ave_fil_3=average_filter(img,3)
# img_ave_fil_5=average_filter(img,5)
# img_ave_fil_9=average_filter(img,9)
# cv2.imshow("3*3",img_ave_fil_3)
# cv2.imshow("5*5",img_ave_fil_5)
# cv2.imshow("9*9",img_ave_fil_9)

img_media_fil_3 = median_filter(img_noise, 3)
cv2.imshow("img_media_fil_3", img_media_fil_3)

cv2.waitKey(0)
