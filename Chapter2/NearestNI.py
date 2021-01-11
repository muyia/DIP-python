#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
def function(img):
    height,weight,channels=img.shape
    print(height,weight)
    NewHeight,NewWeight=750,750
    NewImage=np.zeros((NewHeight,NewWeight,channels),np.uint8)
    for i in range(NewHeight-1):
        x=round(i*height/NewHeight)
        for j in range(NewWeight-1):
            y=round(j*weight/NewWeight)
            NewImage[i,j]=img[x,y]
    return NewImage
img=cv2.imread('img.jpg')
newimg=function(img)
'''
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(newimg)
plt.show()
'''
cv2.imshow("img",img)
cv2.imshow("newimg",newimg)
cv2.waitKey(0)

