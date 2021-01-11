#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_transform(img):

    Imin, Imax = cv2.minMaxLoc(img)[:2]
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    img_match = a * img + b
    img_match = img_match.astype(np.uint8)
    return img_match

img = cv2.imread("imghis.jpg",0)
img_match = normalize_transform(img)
cv2.imshow('img',img)
cv2.imshow('img_match', img_match)
plt.hist(img_match.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
cv2.waitKey()
