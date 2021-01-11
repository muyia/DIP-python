#!/user/bin/env python
# -*- coding:utf-8 -*-
import math
import numpy as np
import cv2
print(int(round(0.75555555555)))
print(round(2/3333))
for i in range(-1,2):
    print(i)
print(pow(10,0.2))
print("log:",np.log(22))
list=[1,2,3,4,5]
list1=[[1,2],[3,4],[5]]
print(list[2-1:2+1+1])
print("list1_len",len(list1))


createVar = locals()
for i in range(5):
    createVar['noiseimg%s'%i] = i
    print(eval('noiseimg' + str(i)))

