#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

def edgeDetection(img):

    pass


def laplacian(img, lapla):
    '''
    :param img: 输入图像
    :param lapla: 拉普拉斯滤波器
    :return: 输出未标定结果的拉普拉斯滤波后结果
    '''
    height, width, channels = img.shape
    filter_size = len(lapla)
    n = int((filter_size - 1) / 2)
    img_lapla = np.zeros((height, width, channels), np.uint8)
    for i in range(n, height - n):
        for j in range(n, width - n):
            for k in range(channels):
                temp = np.sum(lapla * img[i - n:i + n + 1, j - n:j + n + 1, k])
                img_lapla[i, j, k]=np.clip(temp, 0, 255)
    return img_lapla




lapla90 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]  # 90°增量
lapla45 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]  # 45°增量

sobel = [[-2, -2, 0],
           [-2, 0, 2],
           [0, 2, 2]]


img = cv2.imread("../images/color.jpg")
cv2.imshow("image", img)

imglapla90 = laplacian(img, lapla90)
cv2.imshow("imglapla90", imglapla90)


imglapla45 = laplacian(img, lapla45)
cv2.imshow("imglapla45", imglapla45)

imgsobel = laplacian(img, sobel)
cv2.imshow("sobel", imgsobel)

cv2.waitKey(0)