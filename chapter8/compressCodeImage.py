#!/user/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib as plt


# 结点对象
class Node:
    def __init__(self):
        # 权重
        self.probability = 0
        # 符号名
        self.name = None
        # 左结点
        self.l_child = None
        # 右结点
        self.r_child = None
        self.code = None

    # 对频数排序
    def __lt__(self, other):
        return self.probability < other.probability


# 从图像中获取每个灰度值的概率
def Probability(img):
    height, width,channels = img.shape
    # 计算概率
    probability = np.zeros(256, np.float)
    for i in range(256):
        probability[i] = len(img[np.where(img == i)]) / (height * width*channels)
        print(probability[i])
    return probability


# 创建哈夫曼树
def establish_huffman_tree(probability):
    # 输出哈夫曼树的根结点
    node_list = []
    for i in range(256):
        a = Node()
        # 频数
        a.probability = probability[i]
        # 符号名
        a.name = i
        node_list.append(a)

    # while len(node_list) > 1:
    # 生成N-1个新结点
    for j in range(len(probability) - 1):
        # 从大到小排序
        node_list.sort(reverse=True)
        # 提取概率最小的两个结点，并将它们从node_list中删除
        node_1 = node_list.pop()
        node_2 = node_list.pop()

        # 新结点
        new_node = Node()
        new_node.probability = node_1.probability + node_2.probability
        # 左边概率较右边概率大
        new_node.l_child = node_2
        new_node.r_child = node_1
        node_list.append(new_node)

    # 返回哈夫曼树的根结点
    return node_list[0]


# 哈夫曼编码
def encode(base_node, rst_dict, code):
    if base_node.name is not None:
        rst_dict.update({base_node.name: code})
        return
    code += '1'
    encode(base_node.l_child, rst_dict, code)
    code = code[:-1]
    code += '0'
    encode(base_node.r_child, rst_dict, code)
    return rst_dict


# 编码图像
def encode_img(code_dict, img):
    string = ''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                string += code_dict[img[i,j,k]]
    string += '\n'
    return string


# 解码
def decode(codeString, img, base_node):
    root = base_node
    height, width, channels = img.shape
    decodeimg = np.zeros((height, width, channels), np.uint8)
    flag = 0
    for c in codeString:
        if c == '1':
            child = root.l_child
            if child.name is not None:
                i,j,k = int(flag / 3 / width % height), int(flag / 3 % width), int(flag % 3)
                decodeimg[i,j,k] = child.name
                flag +=1
                root = base_node
            else:
                root = child
        elif c == '0':
            child = root.r_child
            if child.name:
                i, j, k = int(flag / 3 / width % height), int(flag / 3 % width), int(flag % 3)
                decodeimg[i, j, k] = child.name
                flag += 1
                root = base_node
            else:
                root = child
        else : break
    return decodeimg

def ImageHuffmanCode(img):
    height, width, channels = img.shape
    # 计算概率
    P = Probability(img)
    # 构建哈夫曼树,根节点base_node
    base_node = establish_huffman_tree(P)
    # 哈夫曼编码
    code_dict = encode(base_node, {}, '')
    # 编码
    encodeimg = encode_img(code_dict, img)
    print("编码结果：\n",encodeimg)
    avelenth = len(encodeimg)/(height * width * channels)
    print("平均编码长度：",avelenth)
    # 解码
    decodeimg = decode(encodeimg,img, base_node)
    cv2.imshow("decdoeimg",decodeimg)
    return 0

if __name__ == "__main__":
    img = cv2.imread("../images/csu.jpg")
    cv2.imshow("img", img)
    ImageHuffmanCode(img)
    cv2.waitKey(0)
