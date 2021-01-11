#!/user/bin/env python
# -*- coding:utf-8 -*-
# !/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np


def laplacian(img, lapla):
    '''
    :param img: 输入图像
    :param lapla: 拉普拉斯滤波器
    :return: 输出未标定结果的拉普拉斯滤波后结果
    '''
    height, width = img.shape
    filter_size = len(lapla)
    n = int((filter_size - 1) / 2)
    img_lapla = np.zeros((height, width), np.uint8)
    for i in range(n, height - n):
        for j in range(n, width - n):
            temp = np.sum(lapla * img[i - n:i + n + 1, j - n:j + n + 1])
            img_lapla[i, j] = np.clip(temp, 0, 255)
    return img_lapla


def img_sharpen_filter(img, img_lapla):
    '''
    :param img: 输入图像
    :param img_lapla: 标定结果的拉普拉斯滤波后的图像
    :return: 输出拉普拉斯锐化后的图像
    '''
    height, width = img.shape
    img_sharpen = np.zeros(img.shape, np.uint8)
    for i in range(height):
        for j in range(width):
            temp = int(img[i, j]) - int(img_lapla[i, j])    #中心值为负值，减法
            img_sharpen[i, j] = np.clip(temp, 0, 255)
    return img_sharpen

def rgbtransgray(img):
    height, width, channels = img.shape
    img_gray = np.zeros((height, width),np.uint8)
    for r in range(height):
        for l in range(width):
            img_gray[r, l] = int(1 / 3 * img[r, l, 0] + 1 / 3 * img[r, l, 1] + 1 / 3 * img[r, l, 2])
            # img_gray[r, l] = np.clip(temp, 0, 255)
    return img_gray


lapla90 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]  # 90°增量
lapla45 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]  # 45°增量
img1 = cv2.imread("images/123.jpg")
img = rgbtransgray(img1)
cv2.imshow("image", img)

imglapla90 = laplacian(img, lapla90)
cv2.imshow("imglapla90", imglapla90)
img_sharpen_fil_90 = img_sharpen_filter(img, imglapla90)
cv2.imshow("img_sharpen_fil_90", img_sharpen_fil_90)

imglapla45 = laplacian(img, lapla45)
cv2.imshow("imglapla45", imglapla45)
img_sharpen_fil_45 = img_sharpen_filter(img, imglapla45)
cv2.imshow("img_sharpen_fil_45", img_sharpen_fil_45)

cv2.waitKey(0)
