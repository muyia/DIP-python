#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from skimage import util
from chapter4 import threeTypicalFilter
import chapter4
from  chapter4 import twoDFourierTransform


def degradationFunction(u, v, M, N, k=0.0025):
    # （大气湍流模型）
    return np.exp(-k * (((u - M / 2) ** 2 + (v - N / 2) ** 2) ** (5 / 6)))
    # return np.exp(-k * ((u ** 2 + v ** 2) ** (5 / 6)))


def dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def idft(ishift):
    ishift=np.fft.ifftshift(ishift)
    img = np.fft.ifft2(ishift)
    return img



# 图像退化
def degradationImage(img):
    img = dft(img)
    # img2 = (np.abs(img) / np.abs(img).max() * 255).astype(np.uint8)
    # img2 = chapter4.twoDFourierTransform.img_trans_log(img2)
    # cv2.imshow("1",img2)
    height, width = img.shape
    result = np.copy(img)
    for i in range(height):
        for j in range(width):
            # print(degradationFunction(i, j, height, width))
            result[i, j] = img[i, j] * degradationFunction(i, j, height, width)
    # result2 = (np.abs(result) / np.abs(result).max() * 255).astype(np.uint8)
    # result2 = chapter4.twoDFourierTransform.img_trans_log(result2)
    # cv2.imshow("2", result2)
    result = idft(result)
    noise = np.zeros(img.shape,np.uint8)
    noise = np.random.normal(0., 1,img.shape)
    noise = (np.abs(noise) / np.abs(noise).max() * 255).astype(np.uint8)
    return result


# //图像复原
def InverseFilterImageRestoration(img):
    img = dft(img)
    height, width = img.shape
    result = np.copy(img)
    for i in range(height):
        for j in range(width):
            result[i, j] = img[i, j] / degradationFunction(i, j, height, width)
    result = idft(result)
    # result = (np.abs(result) / np.abs(result).max() * 255).astype(np.uint8)
    result2 = chapter4.threeTypicalFilter.ButterworthFilterSmooth(result,D0=20,n=10)
    print(result,result2)
    return result
# //图像复原半径受限D0
def xianzhiInverseFilterImageRestoration(img,D0=40,n=10):
    # result = chapter4.threeTypicalFilter.ButterworthFilterSmooth(img, D0, n=10)
    img = dft(img)
    height, width = img.shape
    result = np.copy(img)
    for i in range(height-1):
        for j in range(width-1):
            result[i, j] = img[i,j] * (1 / (1 + pow(((i - height) ** 2 + (j - width) ** 2) ** 0.5 / D0, 2*n)))
            # result[i, j] = result[i, j] / degradationFunction(i, j, height, width)
    result = idft(result)
    # result = (np.abs(result) / np.abs(result).max() * 255).astype(np.uint8)
    # print(result,result2)
    return result


def rgbtransgray(img):
    height, width, channels = img.shape
    img_gray = np.zeros((height, width))
    for r in range(height):
        for l in range(width):
            temp = (1 / 3 * img[r, l, 0] + 1 / 3 * img[r, l, 1] + 1 / 3 * img[r, l, 2])
            img_gray[r, l] = np.clip(temp, 0, 255)
    return img_gray


def addnoisefreq(img):
    height, width = img.shape
    noisefreq = np.zeros((img.shape))
    noisefreq = np.fft.ifft2(noisefreq)
    for i in range(height):
        for j in range(width):
            noisefreq += img[i, j]
    return noisefreq


image = cv2.imread('../images/csu.jpg')
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
result2 = xianzhiInverseFilterImageRestoration(degraimage,40)
result2 = (np.abs(result2) / np.abs(result2).max() * 255).astype(np.uint8)
cv2.imshow("result2", result2)

# filimg = np.fft.fft2(imgy)
# degraimage = degradationImage(filimg)
# addnoise = addnoisefreq(degraimage)
# #
# # cv2.imshow("degraimg", degraimage)
# tuihua = np.fft.ifft2(addnoise)
# tuihua = (np.abs(tuihua) / np.abs(tuihua).max() * 255).astype(np.uint8)
# cv2.imshow("tuihua",tuihua)
# # noiseimage = util.random_noise(degraimage,"gaussian")
# # cv2.imshow("noiseimg",noiseimage)
#
# # 恢复图像
# restorationimg = InverseFilterImageRestoration(degraimage)
# restorationimg = np.fft.ifft2(restorationimg)
# restorationimg = (np.abs(restorationimg) / np.abs(restorationimg).max() * 255).astype(np.uint8)
# cv2.imshow("restorationimg", restorationimg)

cv2.waitKey(0)
