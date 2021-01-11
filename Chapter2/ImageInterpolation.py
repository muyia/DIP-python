#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

def NearestFunction(img):
    height,width,channels=img.shape
    NewHeight,NewWidth=500,500      #300*300 to 500*500
    NewImage=np.zeros((NewHeight,NewWidth,channels),np.uint8)
    for i in range(NewHeight):
        x=int(i*height/NewHeight)
        for j in range(NewWidth):
            y=int(j*width/NewWidth)
            NewImage[i,j]=img[x,y]
    return NewImage

def BilinearFunction(img):
    height, width, channels = img.shape
    NewHeight, NewWidth = 500, 500
    NewImage = np.zeros((NewHeight, NewWidth, channels), np.uint8)
    for i in range(NewHeight):
        x_temp=height*i/NewHeight
        x=int(x_temp)
        u=x_temp-x
        if x>=height-1:
            x=height-2
        for j in range(NewWidth):
            y_temp=width*j/NewWidth
            y=int(y_temp)
            v=y_temp-y
            if y>=width-1:
                y=width-2
            NewImage[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return NewImage

def BiCubicWeight(x):
    x=abs(x)
    if x<=1:
        return  (x**3)-2*(x**2)+1
    elif x<2:
        return  -(x**3)+5*(x**2)-8*x+4
    else :
        return 0
def BiCubicFunction(img):
    height, width, channels = img.shape
    NewHeight, NewWidth = 500, 500
    NewImage = np.zeros((NewHeight, NewWidth, channels), np.uint8)
    for i in range(NewHeight):
        x_temp = height * i / NewHeight
        x = int(x_temp)
        u = x_temp - x
        for j in range(NewWidth):
            y_temp = width * j / NewWidth
            y = int(y_temp)
            v = y_temp - y
            temp=0
            # f(i,j)=f(x+u,y+v)=ABC
            for ii in range(-1,3):
                for jj in range(-1,3):
                    if ii+x<0 or jj+y<0 or ii+x>=height or jj+y>=width:
                        continue
                    temp+=img[x+ii,y+jj]*BiCubicWeight(ii-u)*BiCubicWeight(jj-v)
            NewImage[i, j]=np.clip(temp,0,255) #将灰度值限制在0-255，否则会超
    return NewImage




img=cv2.imread('csu.jpg')
NearestImg=NearestFunction(img)
BilinearImg=BilinearFunction(img)
BiCubicImg=BiCubicFunction(img)

cv2.imshow("img",img)
cv2.imshow("NearestImg",NearestImg)
cv2.imshow("BilinearImg",BilinearImg)
cv2.imshow("BiCubicImg",BiCubicImg)
cv2.waitKey(0)