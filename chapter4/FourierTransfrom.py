#!/user/bin/env python
# -*- coding:utf-8 -*-
from cv2 import cv2
import numpy as np
import math


def fourierTransfrom(image):
    height, width, channels = image.shape
    FTimg = np.zeros(image.shape, np.float)
    print(FTimg.shape)
    index=0
    for k in range(channels):
        for u in range(height):
            for v in range(width):
                index+=1
                # temp = 0
                # FTimg[u, v, k] = 10
                for x in range(height):
                    for y in range(width):
                        if index%1000==0:
                            print(x,',',y)
                # temp+=(image[x,y,k]*np.exp(-1.j*2*math.pi*(u*x/height+v*y/width))).real/(height*width)
                FTimg[u,v,k]=10
    print(FTimg.shape)
    return FTimg


img = cv2.imread('../images/csu.jpg')
cv2.imshow("image", img)
dftimg = fourierTransfrom(img)
cv2.imshow("dftimage", dftimg)
cv2.waitKey(0)
