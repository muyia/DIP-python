#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np


# （大气湍流模型）
def atmosphericDegradationFunction(u, v, M, N, k=0.0025):
    return np.exp(-k * (((u - M / 2) ** 2 + (v - N / 2) ** 2) ** (5 / 6)))
# （运动模糊模型）
def motionBlurDegradationFunction(u, v,a=0.1, b=0.1,T=1):
    r = (u*a + v*b) * np.pi
    if r == 0 :
        return T
    else:
        return T / r * np.sin(r) * np.exp(-1.j * r)

def dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def idft(ishift):
    ishift=np.fft.ifftshift(ishift)
    img = np.fft.ifft2(ishift)
    return img


# 图像退化
def degradationImage(img , op ="atmospheric"):
    img = dft(img)
    height, width = img.shape
    result = np.copy(img)
    for u in range(height):
        for v in range(width):
            # 选择退化函数
            if op == "atmospheric" :
                result[u, v] = img[u, v] * atmosphericDegradationFunction(u, v, height, width)
            elif op == "motionBlur":
                result[u, v] = img[u, v] * motionBlurDegradationFunction(u, v)
    result = idft(result)
    # 高斯噪声
    noise = np.random.normal(0., 1,img.shape)
    return result+noise


# //全逆滤波图像复原
def InverseFilterImageRestoration(img):
    img = dft(img)
    height, width = img.shape
    result = np.copy(img)
    for u in range(height):
        for v in range(width):
            result[u, v] = img[u, v] / atmosphericDegradationFunction(u, v, height, width)
    result = idft(result)
    return result

# //半径（D0）受限逆滤波图像复原
def xianzhiInverseFilterImageRestoration(img,D0=70,n=10):
    # result = chapter4.threeTypicalFilter.ButterworthFilterSmooth(img, D0, n=10)
    img = dft(img)
    height, width = img.shape
    result = np.copy(img)
    for u in range(height-1):
        for v in range(width-1):
            #逆滤波后乘上一个布特沃斯低通函数
            result[u, v] = img[u,v] / atmosphericDegradationFunction(u, v, height, width) * (1 / (1 + pow(((u - height/2) ** 2 + (v - width/2) ** 2) ** 0.5 / D0, 2*n)))
    result = idft(result)
    return result


# 转灰度图像
def rgbtransgray(img):
    height, width, channels = img.shape
    img_gray = np.zeros((height, width))
    for r in range(height):
        for l in range(width):
            temp = (1 / 3 * img[r, l, 0] + 1 / 3 * img[r, l, 1] + 1 / 3 * img[r, l, 2])
            img_gray[r, l] = np.clip(temp, 0, 255)
    return img_gray




image = cv2.imread('../images/hangpai.jpg')
imgy = rgbtransgray(image)  # 转化为灰度图
imgy = (np.abs(imgy) / np.abs(imgy).max() * 255).astype(np.uint8)
cv2.imshow("image", imgy)
degraimage = degradationImage(imgy)
# degraimage1 = np.clip(degraimage, 0, 255)
degraimage1 = (np.abs(degraimage) / np.abs(degraimage).max() * 255).astype(np.uint8)
cv2.imshow("degraimage",degraimage1)
restorationimg = InverseFilterImageRestoration(degraimage)
restorationimg = (np.abs(restorationimg) / np.abs(restorationimg).max() * 255).astype(np.uint8)
cv2.imshow("restorationimg", restorationimg)
# result1 = xianzhiInverseFilterImageRestoration(degraimage,20)
# result1 = (np.abs(result1) / np.abs(result1).max() * 255).astype(np.uint8)
# cv2.imshow("r=20", result1)
result2 = xianzhiInverseFilterImageRestoration(degraimage,40)
result2 = (np.abs(result2) / np.abs(result2).max() * 255).astype(np.uint8)
cv2.imshow("r=40", result2)
result3 = xianzhiInverseFilterImageRestoration(degraimage,70)
result3 = (np.abs(result3) / np.abs(result3).max() * 255).astype(np.uint8)
cv2.imshow("r=70", result3)
result4 = xianzhiInverseFilterImageRestoration(degraimage,85)
result4 = (np.abs(result4) / np.abs(result4).max() * 255).astype(np.uint8)
cv2.imshow("r=85", result2)
# result5 = xianzhiInverseFilterImageRestoration(degraimage,100)
# result5 = (np.abs(result5) / np.abs(result5).max() * 255).astype(np.uint8)
# cv2.imshow("r=100", result5)


cv2.waitKey(0)
