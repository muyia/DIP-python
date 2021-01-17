#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import  numpy as np

def imgaddname(img,namelist,x,y):
    img_add_name = np.copy(img)
    for i in range(len(namelist)):
        for j in range(len(namelist[i])):
            temp = str(np.binary_repr(int(img[i+x, j+y]), 8))  # 将灰度值转换为二进制的八位字符串
            temp=list(temp)
            temp[7]=str(namelist[i][j])
            temp1=''.join(temp)
            img_add_name[i+x, j+y]=int(temp1,2)
    return img_add_name


def img_show_name(img):
    height, width = img.shape
    img_name = np.zeros((height, width), np.float)
    for i in range(height):
        for j in range(width):
            n = str(np.binary_repr(int(img[i, j]), 8))  # 将灰度值转换为二进制的八位字符串
            # print(n)
            img_name[i, j] = n[7]
    return img_name

def rgbtransgray(img):
    height, width, channels = img.shape
    img_gray = np.zeros((height, width),np.uint8)
    for r in range(height):
        for l in range(width):
            temp = (1 / 3 * img[r, l, 0] + 1 / 3 * img[r, l, 1] + 1 / 3 * img[r, l, 2])
            img_gray[r, l] = np.clip(temp, 0, 255)
    return img_gray

def main():
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
    img_gray = rgbtransgray(img)
    cv2.imshow("img_gray", img_gray.astype("uint8"))
    img_addname = imgaddname(img_gray, namelist, 10, 20)
    cv2.imshow("img_addname", img_addname)
    img_showname = img_show_name(img_addname.astype("uint8"))
    cv2.imshow("img_showname", img_showname)
    cv2.waitKey(0)
    return 0

img = cv2.imread("../images/yiqin.png")
cv2.imshow("img", img)
img_gray = rgbtransgray(img)
cv2.imshow("imgy", img_gray)
cv2.imwrite("../images/yiqin_gray.jpg",img_gray)
cv2.waitKey(0)