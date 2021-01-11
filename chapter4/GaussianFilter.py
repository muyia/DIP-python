#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

def gaussianFilter(img):
    height, width = img.shape
    # 1.对图像进行填充，填充后的图像尺寸为(2M*2N)
    p = 2 * height
    q = 2 * width
    fillimg = np.zeros((p, q), np.float)
    midimg = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            if (i < height and j < width):
                # 2.形成p*q的填充图像
                fillimg[i, j] = float(img[i, j])
                # 3.中心化
                midimg[i, j] = fillimg[i, j] * (-1) ** (i + j)
    #4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    D0 = 20
    for i in range(p):
        for j in range(q):
            D = ((i - height)**2 + (j - width)**2)**0.5   #D是距离频率矩形中点的距离
            H[i, j] = np.exp(-1 / 2 * ( D**2/ D0**2))  #滤波函数H(u, v)
             # 滤波过程
            HF[i,j] = H[i,j]*fillimgdft[i,j]
            G[i,j] = HF[i,j]*(-1)**(i+j)
    # 6.得到处理后的图像
    resimg=np.fft.ifft2(HF)
    for i in range(p):
        for j in range(q):
            resimg[i,j]=resimg[i,j]*(-1)**(i+j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i,j] = resimg[i,j]
    print(fillimgdft[10, 11])
    return fillimg,midimg,fillimgdft,H,HF,resimg,resultimg

# def gaussianFilter(img):
#     height, width, channels = img.shape
#     # 对图像进行填充，填充后的图像尺寸为(2M*2N)
#     p = 2 * height
#     q = 2 * width
#     fillimg = np.zeros((p, q, channels), np.float)
#     for k in range(channels):
#         for i in range(p):
#             for j in range(q):
#                 if (i < height and j < width):
#                     fillimg[i, j, k] = float(img[i, j, k])
#                     # print(fillimg[10,10])
#                     fillimg[i, j, k] = fillimg[i, j, k] *(-1) ** (i + j)
#     fillimgdft = fft2(fillimg)
#     print(fillimg[10, 11])
#     return fillimg


img = cv2.imread("./img/csu.jpg",0)
cv2.imshow("img", img)
fillimg,midimg,dft,H,HF,resimg,resultimg = gaussianFilter(img)
fillimg = (np.abs(fillimg) / np.abs(fillimg).max() * 255).astype(np.uint8)
cv2.imshow("fillimg", fillimg)
midimg = (np.abs(midimg) / np.abs(midimg).max() * 255).astype(np.uint8)
cv2.imshow("midimg", midimg)
dft = (np.abs(dft) / np.abs(dft).max() * 255).astype(np.uint8)
cv2.imshow("dft", dft)
H = (np.abs(H) / np.abs(H).max() * 255).astype(np.uint8)
cv2.imshow("H", H)
HF = (np.abs(HF) / np.abs(HF).max() * 255).astype(np.uint8)
cv2.imshow("HF", HF)
resimg = (np.abs(resimg) / np.abs(resimg).max() * 255).astype(np.uint8)
cv2.imshow("resimg", resimg)
resultimg = (np.abs(resultimg) / np.abs(resultimg).max() * 255).astype(np.uint8)
cv2.imshow("resultimg", resultimg)
cv2.waitKey(0)
