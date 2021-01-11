#!/user/bin/env python
# -*- coding:utf-8 -*
import cv2
import numpy as np
import random
import skimage
from skimage import util

def imgAverage(img,imgnum):
    # 随机生成imgnum张高斯噪声图像
    imgnames=locals()
    for i in range (imgnum):
        imgnames['noiseimg%s'%i]=util.random_noise(img,"gaussian")
    cv2.imshow("noiseimg",eval('noiseimg' + str(1)))
    height, width, channels = img.shape
    NewImage = np.zeros((height, width, channels), np.float)#random_noise()函数生成的图像为浮点数据图像
    for i in range(height):
        for j in range(width):
            temp=0
            for k in range (imgnum):
                temp+= eval('noiseimg' + str(k))[i, j]#平均，eval('noiseimg' + str(k))为添加噪声的图像
            NewImage[i,j]=temp/imgnum
    return NewImage

img = cv2.imread('csu.jpg')
cv2.imshow("img",img)
imageaverage5=imgAverage(img,5)
imageaverage10=imgAverage(img,10)
imageaverage20=imgAverage(img,20)
imageaverage50=imgAverage(img,50)
cv2.imshow("imageaverage5",imageaverage5)
cv2.imshow("imageaverage10",imageaverage10)
cv2.imshow("imageaverage20",imageaverage20)
cv2.imshow("imageaverage50",imageaverage50)
cv2.waitKey(0)