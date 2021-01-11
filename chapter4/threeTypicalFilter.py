#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

def gaussianFilterSmooth(img, D0=10):
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
    # 4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            D = ((i - height) ** 2 + (j - width) ** 2) ** 0.5
            H[i, j] = np.exp(-1 / 2 * (D ** 2 / D0 ** 2))  # 滤波函数H(u, v)
            # 滤波过程
            HF[i, j] = H[i, j] * fillimgdft[i, j]
            G[i, j] = HF[i, j] * (-1) ** (i + j)
    # 6.得到处理后的图像
    resimg = np.fft.ifft2(HF)
    # for i in range(p):
    #     for j in range(q):
    #         resimg[i, j] = resimg[i, j] * (-1) ** (i + j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i, j] = resimg[i, j]
    print(fillimgdft[10, 11])
    return resultimg


def ideaFilterSmooth(img, D0=10):
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
    # 4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            D = ((i - height) ** 2 + (j - width) ** 2) ** 0.5
            # 滤波函数H(u, v)
            if D <= D0:
                H[i, j] = 1
            else:
                H[i, j] = 0
            # 滤波过程
            HF[i, j] = H[i, j] * fillimgdft[i, j]
            G[i, j] = HF[i, j] * (-1) ** (i + j)
    # 6.得到处理后的图像
    resimg = np.fft.ifft2(HF)
    for i in range(p):
        for j in range(q):
            resimg[i, j] = resimg[i, j] * (-1) ** (i + j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i, j] = resimg[i, j]
    print(fillimgdft[10, 11])
    return resultimg


def ButterworthFilterSmooth(img, D0=30, n=2):
    '''
    :param img: 输入图像
    :param D0: 截止频率
    :param n: 布特沃斯滤波器的阶乘
    :return: 平滑后的图像
    '''
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
    # 4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # fillimgdft = np.fft.fftshift(fillimgdft)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            D = ((i - height) ** 2 + (j - width) ** 2) ** 0.5
            # 滤波函数H(u, v)
            H[i, j] = 1 / (1 + pow(D / D0, 2 * n))
            # 滤波过程
            HF[i, j] = H[i, j] * fillimgdft[i, j]
            G[i, j] = HF[i, j] * (-1) ** (i + j)
    # 6.得到处理后的图像
    # HF = np.fft.ifftshift(HF)
    resimg = np.fft.ifft2(HF)
    for i in range(p):
        for j in range(q):
            resimg[i, j] = resimg[i, j] * (-1) ** (i + j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i, j] = resimg[i, j]
    # print(fillimgdft[10, 11])
    return resultimg


def gaussianFilterSharpen(img, D0=30):
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
    # 4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            D = ((i - height) ** 2 + (j - width) ** 2) ** 0.5
            H[i, j] = 1 - np.exp(-1 / 2 * (D ** 2 / D0 ** 2))  # 滤波函数H(u, v)
            # 滤波过程
            HF[i, j] = H[i, j] * fillimgdft[i, j]
            G[i, j] = HF[i, j] * (-1) ** (i + j)
    # 6.得到处理后的图像
    resimg = np.fft.ifft2(HF)
    for i in range(p):
        for j in range(q):
            resimg[i, j] = resimg[i, j] * (-1) ** (i + j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i, j] = resimg[i, j]
    print(fillimgdft[10, 11])
    return resultimg


def ideaFilterSharpen(img, D0=10):
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
    # 4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            D = ((i - height) ** 2 + (j - width) ** 2) ** 0.5
            # 滤波函数H(u, v)
            if D <= D0:
                H[i, j] = 0
            else:
                H[i, j] = 1
            # 滤波过程
            HF[i, j] = H[i, j] * fillimgdft[i, j]
            G[i, j] = HF[i, j] * (-1) ** (i + j)
    # 6.得到处理后的图像
    resimg = np.fft.ifft2(HF)
    for i in range(p):
        for j in range(q):
            resimg[i, j] = resimg[i, j] * (-1) ** (i + j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i, j] = resimg[i, j]
    print(fillimgdft[10, 11])
    return resultimg


def ButterworthFilterSharpen(img, D0=30, n=2):
    '''
    :param img: 输入图像
    :param D0: 截止频率
    :param n: 布特沃斯滤波器的阶乘
    :return: 锐化后的图像
    '''
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
    # 4.得到dft图谱
    fillimgdft = np.fft.fft2(midimg)
    # 5.生成滤波函数
    H = np.zeros((p, q), np.float)
    HF = np.zeros((p, q), np.float)
    G = np.zeros((p, q), np.float)
    for i in range(p):
        for j in range(q):
            D = ((i - height) ** 2 + (j - width) ** 2) ** 0.5
            # 滤波函数H(u, v)
            if D != 0:
                H[i, j] = 1 / (1 + pow(D0 / D, 2 * n))
            # 滤波过程
            HF[i, j] = H[i, j] * fillimgdft[i, j]
            G[i, j] = HF[i, j] * (-1) ** (i + j)
    # 6.得到处理后的图像
    resimg = np.fft.ifft2(HF)
    for i in range(p):
        for j in range(q):
            resimg[i, j] = resimg[i, j] * (-1) ** (i + j)
    resultimg = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            resultimg[i, j] = resimg[i, j]
    print(fillimgdft[10, 11])
    return resultimg


# img = cv2.imread("./img/csu.jpg", 0)
# cv2.imshow("img", img)
# gaussianFilterSmoothResultimg = ButterworthFilterSmooth(img,10)
# gaussianFilterSmoothResultimg = (np.abs(gaussianFilterSmoothResultimg) / np.abs(gaussianFilterSmoothResultimg).max() * 255).astype(np.uint8)
# cv2.imshow("D0=10", gaussianFilterSmoothResultimg)
# gaussianFilterSmoothResultimg1 = ButterworthFilterSmooth(img,30)
# gaussianFilterSmoothResultimg1 = (np.abs(gaussianFilterSmoothResultimg1) / np.abs(gaussianFilterSmoothResultimg1).max() * 255).astype(np.uint8)
# cv2.imshow("D0=30", gaussianFilterSmoothResultimg1)
# gaussianFilterSmoothResultimg2 = ButterworthFilterSmooth(img,60)
# gaussianFilterSmoothResultimg2 = (np.abs(gaussianFilterSmoothResultimg2) / np.abs(gaussianFilterSmoothResultimg2).max() * 255).astype(np.uint8)
# cv2.imshow("D0=60", gaussianFilterSmoothResultimg2)
# gaussianFilterSmoothResultimg3 = ButterworthFilterSmooth(img,160)
# gaussianFilterSmoothResultimg3 = (np.abs(gaussianFilterSmoothResultimg3) / np.abs(gaussianFilterSmoothResultimg3).max() * 255).astype(np.uint8)
# cv2.imshow("D0=160", gaussianFilterSmoothResultimg3)
# cv2.waitKey(0)

# ideaFilterSmoothResultimg = ideaFilterSmooth(img)
# ideaFilterSmoothResultimg = (np.abs(ideaFilterSmoothResultimg) / np.abs(ideaFilterSmoothResultimg).max() * 255).astype(
#     np.uint8)
# cv2.imshow("ideaFilterSmoothResultimg", ideaFilterSmoothResultimg)
# butterworthFilterSmoothResultimg = ButterworthFilterSmooth(img)
# butterworthFilterSmoothResultimg = (np.abs(butterworthFilterSmoothResultimg) / np.abs(butterworthFilterSmoothResultimg).max() * 255).astype(np.uint8)
# cv2.imshow("butterworthFilterSmoothResultimg", butterworthFilterSmoothResultimg)

# gaussianFilterSmoothResultimg1 = gaussianFilterSharpen(img,30)
# gaussianFilterSmoothResultimg1 = (np.abs(gaussianFilterSmoothResultimg1) / np.abs(gaussianFilterSmoothResultimg1).max() * 255).astype(np.uint8)
# cv2.imshow("D0=30", gaussianFilterSmoothResultimg1)
# gaussianFilterSmoothResultimg2 = gaussianFilterSharpen(img,60)
# gaussianFilterSmoothResultimg2 = (np.abs(gaussianFilterSmoothResultimg2) / np.abs(gaussianFilterSmoothResultimg2).max() * 255).astype(np.uint8)
# cv2.imshow("D0=60", gaussianFilterSmoothResultimg2)
# gaussianFilterSmoothResultimg3 = gaussianFilterSharpen(img,160)
# gaussianFilterSmoothResultimg3 = (np.abs(gaussianFilterSmoothResultimg3) / np.abs(gaussianFilterSmoothResultimg3).max() * 255).astype(np.uint8)
# cv2.imshow("D0=160", gaussianFilterSmoothResultimg3)




# ideaFilterSmoothResultimg = ideaFilterSharpen(img,30)
# ideaFilterSmoothResultimg = (np.abs(ideaFilterSmoothResultimg) / np.abs(ideaFilterSmoothResultimg).max() * 255).astype(np.uint8)
# cv2.imshow("D0=30", ideaFilterSmoothResultimg)
# ideaFilterSmoothResultimg2 = ideaFilterSharpen(img,60)
# ideaFilterSmoothResultimg2 = (np.abs(ideaFilterSmoothResultimg2) / np.abs(ideaFilterSmoothResultimg2).max() * 255).astype(np.uint8)
# cv2.imshow("D0=60", ideaFilterSmoothResultimg2)
# ideaFilterSmoothResultimg3 = ideaFilterSharpen(img,160)
# ideaFilterSmoothResultimg3 = (np.abs(ideaFilterSmoothResultimg3) / np.abs(ideaFilterSmoothResultimg3).max() * 255).astype(np.uint8)
# cv2.imshow("D0=160", ideaFilterSmoothResultimg3)

# butterworthFilterSmoothResultimg = ButterworthFilterSharpen(img,30,2)
# butterworthFilterSmoothResultimg = (np.abs(butterworthFilterSmoothResultimg) / np.abs(butterworthFilterSmoothResultimg).max() * 255).astype(np.uint8)
# cv2.imshow("D0=30,n=2", butterworthFilterSmoothResultimg)
# butterworthFilterSmoothResultimg2 = ButterworthFilterSharpen(img,60,2)
# butterworthFilterSmoothResultimg2 = (np.abs(butterworthFilterSmoothResultimg2) / np.abs(butterworthFilterSmoothResultimg2).max() * 255).astype(np.uint8)
# cv2.imshow("D0=60,n=2", butterworthFilterSmoothResultimg2)
# butterworthFilterSmoothResultimg3 = ButterworthFilterSharpen(img,160,2)
# butterworthFilterSmoothResultimg3 = (np.abs(butterworthFilterSmoothResultimg3) / np.abs(butterworthFilterSmoothResultimg3).max() * 255).astype(np.uint8)
# cv2.imshow("D0=160,n=2", butterworthFilterSmoothResultimg3)
#
#
# cv2.waitKey(0)
