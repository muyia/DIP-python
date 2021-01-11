#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import random
import skimage
from skimage import util



def ImgAverage(img1,img2):
    height, width, channels = img1.shape
    NewImage = np.zeros((height, width, channels), np.float)
    for i in range(height):
        for j in range(width):
            NewImage[i, j]=(img1[i,j]+img2[i,j])/2
    return NewImage

img=cv2.imread('img.jpg')
noiseimg1=util.random_noise(img,"gaussian")
noiseimg2=util.random_noise(img,"gaussian")
print(noiseimg1[1,1])
averageimg=ImgAverage(noiseimg1,noiseimg2)

cv2.imshow("img",img)
cv2.imshow("noiseimg1",noiseimg1)
cv2.imshow("noiseimg2",noiseimg2)
cv2.imshow("averageimg",averageimg)
cv2.waitKey(0)