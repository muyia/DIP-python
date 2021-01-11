#!/user/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np




def selectiveFilter(img,filt="gaussian",op="bandreject",W=1 ,D0=30, n=2):
    '''
    :param img: 输入图像
    :param filt: 选择的滤波器 gaussian,idea,butterworth
    :param op:  带阻带通 bandreject，bandpass
    :param W:  带宽
    :param D0: 截止频率
    :param n: 布特沃斯滤波器的阶乘
    :return: 操作后的图像
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
            if filt == "gaussian":
                if op == "bandreject":
                    if D>0:
                        H[i, j] = 1 - np.exp(- pow((D**2-D0**2)/D*W,2))
                if op == "bandpass":
                    if D>0:
                        H[i, j] = np.exp(- pow((D ** 2 - D0 ** 2) / D * W, 2))
            if filt == "idea":
                if op == "bandreject":
                    if D>=(D0-W/2) and D<=D0 + W/2 :
                        H[i, j] = 0
                    else: H[i, j] = 1
                if op == "bandpass":
                    if D>=(D0-W/2) and D<=D0 + W/2 :
                        H[i, j] = 1
                    else: H[i, j] = 0
            if filt == "butterworth":
                if op == "bandreject":
                    if D > D0 :
                        H[i, j] = 1 / (1 + pow(D*W / D**2 - D0**2, 2 * n))
                if op == "bandpass":
                    if D > D0 :
                        H[i, j] = 1 - 1 / (1 + pow(D*W / D**2 - D0**2, 2 * n))

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
# resultimg = selectiveFilter(img,"gaussian","bandreject",2,30,2)
# resultimg = (np.abs(resultimg) / np.abs(resultimg).max() * 255).astype(np.uint8)
# cv2.imshow("gaussianbandreject", resultimg)
# resultimg2 = selectiveFilter(img,"gaussian","bandpass",2,30,2)
# resultimg2 = (np.abs(resultimg2) / np.abs(resultimg2).max() * 255).astype(np.uint8)
# cv2.imshow("gaussianbandpass", resultimg2)
# resultimg3 = selectiveFilter(img,"idea","bandreject",2,30,2)
# resultimg3 = (np.abs(resultimg3) / np.abs(resultimg3).max() * 255).astype(np.uint8)
# cv2.imshow("ideabandreject", resultimg3)
# resultimg4 = selectiveFilter(img,"idea","bandpass",2,30,2)
# resultimg4 = (np.abs(resultimg4) / np.abs(resultimg4).max() * 255).astype(np.uint8)
# cv2.imshow("ideabandpass", resultimg4)
# resultimg5 = selectiveFilter(img,"butterworth","bandreject",2,30,2)
# resultimg5 = (np.abs(resultimg5) / np.abs(resultimg5).max() * 255).astype(np.uint8)
# cv2.imshow("butterbandreject", resultimg5)
# resultimg6 = selectiveFilter(img,"butterworth","bandpass",2,30,2)
# resultimg6 = (np.abs(resultimg6) / np.abs(resultimg6).max() * 255).astype(np.uint8)
# cv2.imshow("butterbandpass", resultimg6)
#
#
#
# cv2.waitKey(0)