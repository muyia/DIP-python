#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np


def respectiveRGB(img):
    imgR = np.zeros((img.shape), np.uint8)
    imgG = np.zeros((img.shape), np.uint8)
    imgB = np.zeros((img.shape), np.uint8)
    imgB[:, :, 0] = img[:, :, 0]
    imgG[:, :, 1] = img[:, :, 1]
    imgR[:, :, 2] = img[:, :, 2]
    return imgR,imgG,imgB

def respectiveHSI(img):
    imgHSI = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B, G, R = img[i,j]
            B, G, R = int(B),int(G),int(R)
            I =  (R+G+B)/3
            if I == 0:
                S = 0
                H = 0
                imgHSI[i,j] = H,S,I
                continue
            S = (1-3*np.min([B, G, R])/(R+G+B))*255
            if S==0:
                H = 0
                imgHSI[i, j] = H, S, I
                continue
            # thea = np.arccos(((R - G)/2 + R - B) / (np.sqrt((R - G) ** 2 + (R - B) * (G - B))))
            # 注释的这句代码报错ValueError: cannot convert float NaN to integer
            # 下面是正确的
            thea = np.arccos((2 * R - G - B) / (2 * np.sqrt((R - G) ** 2 + (R - B) * (G - B))))
            if B<=G:
                H= np.rad2deg(thea)/360*255
            else:
                H = np.rad2deg(np.pi*2 - thea)/360*255

            imgHSI[i, j] = H, S, I
    return imgHSI

def RGB():
    img = cv2.imread("../images/csu.jpg")
    cv2.imshow("img", img)
    imgR, imgG, imgB = respectiveRGB(img)
    cv2.imshow("imgR", imgR)
    cv2.imshow("imgG", imgG)
    cv2.imshow("imgB", imgB)
    cv2.waitKey(0)
    return 0
def HSI():
    img = cv2.imread("../images/color.jpg")
    cv2.imshow("img", img)

    imgHSI = respectiveHSI(img)
    cv2.imshow("imgH", imgHSI[:,:,0])
    cv2.imshow("imgS", imgHSI[:,:,1])
    cv2.imshow("imgI", imgHSI[:, :, 2])
    cv2.waitKey(0)
    return 0

if __name__ == "__main__":
    # RGB()
    HSI()
