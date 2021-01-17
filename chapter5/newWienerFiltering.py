#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np


def dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def idft(ishift):
    ishift = np.fft.ifftshift(ishift)
    img = np.fft.ifft2(ishift)
    return img


# （大气湍流模型）
def atmosphericDegradationFunction(u, v, M, N, k=0.0025):



    return np.exp(-k * (((u - M / 2) ** 2 + (v - N / 2) ** 2) ** (5 / 6)))


# （运动模糊模型）
def motionBlurDegradationFunction(u, v, a=0.1, b=0.1, T=1):
    r = (u * a + v * b) * np.pi
    if r == 0:
        return T
    else:
        return T * np.sin(r) * np.exp(-1.j * r) / r


def wiener(u, v, height, width, K=0.002):
    temp = motionBlurDegradationFunction(u - height / 2, v - width / 2)
    conj = np.conj(temp)
    return conj*temp / (conj * temp + K) /temp


def degradationImage(img, op="atmospheric",esp = 1):
    # img = fallin(img)
    imgdft = dft(img)
    height, width = imgdft.shape
    result = np.copy(imgdft)
    for u in range(height):
        for v in range(width):
            # 选择退化函数
            if op == "atmospheric":
                result[u, v] = imgdft[u, v] * atmosphericDegradationFunction(u, v, height, width)
            elif op == "motionBlur":
                result[u, v] = imgdft[u, v] * motionBlurDegradationFunction(u - height / 2, v - width / 2)
    # resulydft1 = np.log((np.abs(result) / np.abs(result).max() * 255)).astype(np.uint8)
    # cv2.imshow("resulydft1", resulydft1)
    result = idft(result)
    # 高斯噪声 esp：方差
    noise = np.random.normal(0., esp, img.shape)
    return result+noise


def wienerFilterImageRestoration(img):
    imgdft = dft(img)
    # imgdft1 = np.log((np.abs(imgdft) / np.abs(imgdft).max() * 255)).astype(np.uint8)
    # cv2.imshow("imgdft",imgdft1)
    height, width = imgdft.shape
    result = np.copy(imgdft)
    for u in range(height):
        for v in range(width):
            result[u, v] = imgdft[u, v] * wiener(u, v, height, width)
    result = idft(result)
    return result

# //全逆滤波图像复原
def InverseFilterImageRestoration(img):
    imgdft = dft(img)
    height, width = img.shape
    result = np.copy(img)
    for u in range(height):
        for v in range(width):
            result[u, v] = imgdft[u, v] / motionBlurDegradationFunction(u - height / 2, v - width / 2)

    result = idft(result)
    return result

#图像填充
def fallin(img,a=2):
    height , width = img.shape
    fallimg = np.zeros((height*3, width*3), np.uint8)
    for i in range(height,height*2):
        for j in range(width,width*2):
            fallimg[i, j] += img[i-height, j-width]
    return fallimg
def fallout(img, a=2):
    height, width = img.shape
    outimg = np.zeros((int(height/a), int(width/a)), np.uint8)
    for i in range(int(height/a)):
        for j in range(int(width/a)):
            outimg[i, j] += img[i, j]
    return outimg


img = cv2.imread('../images/csu256.jpg', 0)
cv2.imshow("img", img)
# img = fallin(img)

degraimage = degradationImage(img, "motionBlur")
#
# degraimage10 = degradationImage(img, "motionBlur", 10)
degraimage1 = (np.abs(degraimage) / np.abs(degraimage).max() * 255).astype(np.uint8)
# degraimage2 = (np.abs(degraimage10) / np.abs(degraimage10).max() * 255).astype(np.uint8)
# degraimage1 = ((np.real(degraimage) - np.real(degraimage).min()) / (np.real(degraimage).max() - np.real(degraimage).min()) * 255).astype(np.uint8)
# degraimage2 = np.clip(degraimage2, 0, 255).astype(np.uint8)
cv2.imshow("degraimage", degraimage1)
# cv2.imshow("degraimage10", degraimage2)

# degraimage = fallout(degraimage)
# degraimage = fallin(degraimage)
restorationimg = wienerFilterImageRestoration(degraimage)
print(restorationimg)
restorationimg = ((np.real(restorationimg) - np.real(restorationimg).min()) / (np.real(restorationimg).max() - np.real(restorationimg).min()) * 255).astype(np.uint8)
print("\n\n\n",restorationimg)
cv2.imshow("restorationimg", restorationimg)
# restorationimg10 = wienerFilterImageRestoration(degraimage10)
# restorationimg10 = (np.abs(restorationimg10) / np.abs(restorationimg10).max() * 255).astype(np.uint8)
# cv2.imshow("restorationimg10", restorationimg10)
# restorationimg2 = InverseFilterImageRestoration(degraimage)
# # restorationimg2 = (np.abs(restorationimg2) / np.abs(restorationimg2).max() * 255).astype(np.uint8)
# restorationimg2 = (np.abs(restorationimg2)/ np.abs(restorationimg2).max() * 255).astype(np.uint8)
# cv2.imshow("restorationimg2", restorationimg2)

cv2.waitKey(0)
