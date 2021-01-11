#!/user/bin/env python
# -*- coding:utf-8 -*-
# !/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

def imgaddname(img, namelist, x, y):
    img_add_name = np.copy(img)
    for i in range(len(namelist)):
        for j in range(len(namelist[i])):
            temp = str(np.binary_repr(int(img[i + x, j + y, 0]), 8))  # 将灰度值转换为二进制的八位字符串
            temp = list(temp)
            temp[7] = str(namelist[i][j])
            temp1 = ''.join(temp)
            img_add_name[i + x, j + y] = int(temp1, 2)
    return img_add_name

def img_show_name(img):
    height, width,_ = img.shape
    img_name = np.zeros((height, width, 1), np.float)
    for i in range(height):
        for j in range(width):
            n = str(np.binary_repr(int(img[i, j, 0]), 8))  # 将灰度值转换为二进制的八位字符串
            # print(n)
            img_name[i, j, 0] = n[7]
    return img_name

namelist = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

print(namelist[2][2], type(namelist))
img = cv2.imread("images/csu.jpg")
cv2.imshow("img", img)
img_addname = imgaddname(img, namelist, 10, 20)
cv2.imshow("img_addname", img_addname)
img_showname = img_show_name(img_addname.astype("uint8"))
cv2.imshow("img_showname", img_showname)
cv2.waitKey(0)
