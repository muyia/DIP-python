#!/user/bin/env python
# -*- coding:utf-8 -*-
# 二维傅里叶变换
# https://www.cnblogs.com/timdyh/p/13338975.html
# 输入图像，输出频谱和log变换后的频谱

import cv2
import numpy as np
from numpy.fft import *
import math
import matplotlib as plt


def dft(x):
    n = len(x)
    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]
    G = fft(x[::2])
    H = fft(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    WH = W * H
    X = np.concatenate([G + WH, G - WH])
    return X


def dft2(img):
    height, width = img.shape
    if ((height - 1) & height) or ((width - 1) & width):
        print('Image size not a power of 2')
        return img
    res = np.zeros([height, width], 'complex128')
    for i in range(height):
        res[i, :] = dft(img[i, :])
    for j in range(width):
        res[:, j] = dft(res[:, j])
    return res





def idft(x):
    n = len(x)
    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]
    G = fft(x[::2])
    H = fft(x[1::2])
    W = np.exp(2j * np.pi * np.arange(n // 2) / n) / n
    WH = W * H
    X = np.concatenate([G + WH, G - WH])
    return X
def idft2(img):
    height, width = img.shape
    if ((height - 1) & height) or ((width - 1) & width):
        print('Image size not a power of 2')
        return img
    res = np.zeros([height, width], 'complex128')
    for i in range(height):
        res[i, :] = idft(img[i, :])
    for j in range(width):
        res[:, j] = idft(res[:, j])
    return res


def fftshift(img):
    # swap the first and third quadrants, and the second and fourth quadrants
    height, width = img.shape
    h_mid, w_mid = height // 2, width // 2
    res = np.zeros([height, width], 'complex128')
    res[:h_mid, :w_mid] = img[h_mid:, w_mid:]
    res[:h_mid, w_mid:] = img[h_mid:, :w_mid]
    res[h_mid:, :w_mid] = img[:h_mid, w_mid:]
    res[h_mid:, w_mid:] = img[:h_mid, :w_mid]
    return res


def img_trans_log(img):
    height, width = img.shape
    img_log = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            img_log[i, j] = round((math.log(1 + img[i, j])) / (math.log(256)) * 255)
    print("img_log:", img_log[1, 1])
    return img_log

#
# # Read image
# img = cv2.imread("./img/test5.png", 0)
# cv2.imshow("img", img)
# # # DFT
# img1 = fft2(img)
# G = fftshift(img1)
# #
# # # write poser spectal to image
# ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
# # print(ps[1,1])
# logps = img_trans_log(ps)
# cv2.imshow("out", ps)
# # cv2.imwrite("./img/test5_ps.jpg", ps)
# cv2.imshow("outlog", logps)
# # cv2.imwrite("./img/test5log_ps.jpg", logps)
#
# # p,q =G.shape
# # for i in range(p):
# #     for j in range(q):
# #         G[i, j] = G[i, j] * (-1) ** (i + j)
# # x = np.fft.fft2(img)
# # imgy = np.fft.ifft2(x)
# # # p,q =imgy.shape
# # # for i in range(p):
# # #     for j in range(q):
# # #         imgy[i, j] = imgy[i, j] * (-1) ** (i + j)
# # imgy = (np.abs(imgy) / np.abs(imgy).max() * 255).astype(np.uint8)
# # cv2.imshow("111",imgy)
#
# cv2.waitKey(0)
