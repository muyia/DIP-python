#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def img_trans_log(img):
    height, width, channels = img.shape
    img_log = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                img_log[i, j, k] = round((math.log(1 + img[i, j, k])) / (math.log(256)) * 255)
    print("img_log:", img_log[1, 1, 1])
    return img_log


def img_trans_gamma(img, gamma):
    '''
    :param img: 输入图像
    :param gamma: 幂
    :return: 伽马变换后图像
    '''
    height, width, channels = img.shape
    print("img:", img[1, 1, 1])
    img_gamma = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                img_gamma[i, j, k] = round(math.pow(img[i][j][k], gamma)/ math.pow(256, gamma) * 255)
    return img_gamma


def img_trans_piecewiseliner(img, r1, s1, r2, s2):
    height, width, channels = img.shape
    img_piecewiseliner = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if img[i][j][k] < r1:
                    img_piecewiseliner[i][j] = round((s1 / r1) * img[i][j][k])
                elif img[i][j][k] < r2:
                    img_piecewiseliner[i][j][k] = round(((s2 - s1) / (r2 - r1)) * (img[i][j][k] - r1) + s1)
                else:
                    img_piecewiseliner[i][j][k] = round(((255 - s2) / (255 - r2)) * (img[i][j][k] - r2) + s1)
    return img_piecewiseliner


def img_trans_bitplaneslicing(img):
    height, width = img.shape
    img_plane = np.zeros((height, width, 8), np.float)
    for i in range(height):
        for j in range(width):
            n = str(np.binary_repr(int(img[i, j]), 8))  # 将灰度值转换为二进制的八位字符串
            # print(n)
            for k in range(8):
                img_plane[i, j, k] = n[k]
    return img_plane


def rgbtransgray(img):
    height, width, channels = img.shape
    img_gray = np.zeros((height, width))
    for r in range(height):
        for l in range(width):
            temp = (1 / 3 * img[r, l, 0] + 1 / 3 * img[r, l, 1] + 1 / 3 * img[r, l, 2])
            img_gray[r, l] = np.clip(temp, 0, 255)
    return img_gray


img = cv2.imread('images/imghis.jpg')
cv2.imshow("img", img)
# 对数变换
# img_log = img_trans_log(img)
# cv2.imshow("img_log", img_log)
# 伽马变换
# img_gamma = img_trans_gamma(img, 0.6)
# cv2.imshow("0.6", img_gamma)
# img_gamma2 = img_trans_gamma(img, 0.4)
# cv2.imshow("0.4", img_gamma2)
# img_gamma3 = img_trans_gamma(img, 0.3)
# cv2.imshow("0.3", img_gamma3)
# img_gamma = img_trans_gamma(img, 3.0)
# cv2.imshow("3.0", img_gamma)
# img_gamma2 = img_trans_gamma(img, 4.0)
# cv2.imshow("4.0", img_gamma2)
# img_gamma3 = img_trans_gamma(img, 5.0)
# cv2.imshow("5.0", img_gamma3)
# 分段线性变换
# img_iecewiseliner = img_trans_piecewiseliner(img, 65, 0, 133, 255)
# cv2.imshow("img_iecewiseliner", img_iecewiseliner)
# plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.hist(img_iecewiseliner.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.show()

# # 转化为灰度图像
# img_gray=rgbtransgray(img)
# cv2.imshow("img",img_gray.astype("uint8"))
# #比特平面分层
# img_plane=img_trans_bitplaneslicing(img_gray)
# for i in range(8):
#     cv2.imshow("img_bit_plane{}".format(i+1),img_plane[:,:,i])

cv2.waitKey(0)
