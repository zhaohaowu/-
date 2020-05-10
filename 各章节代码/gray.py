# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:22:33 2020

@author: zhaohaowu
"""

import cv2
import numpy as np
import matplotlib.image as mping
import matplotlib.pyplot as plt
image = mping.imread('D:/test3.png')
grayimg1_1 = np.zeros_like(image)
grayimg1_2 = np.zeros_like(image)
grayimg1_3 = np.zeros_like(image)
grayimg2 = np.zeros_like(image)
grayimg3 = np.zeros_like(image)
grayimg4 = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        grayimg1_1[i,j] = image[i,j][0]
        grayimg1_2[i,j] = image[i,j][1]
        grayimg1_3[i,j] = image[i,j][2]
        grayimg2[i,j] = max(image[i,j][0], image[i,j][1], image[i,j][2])
        grayimg3[i,j] = 1/3*(image[i,j][0] + image[i,j][1] + image[i,j][2])
        grayimg4[i,j] = 0.5 * image[i,j][0] + 0.5 * image[i,j][1] 
# plt.imshow(image)  
# plt.imshow(grayimg1_1)
# plt.imshow(grayimg1_2)
# plt.imshow(grayimg1_3)
# plt.imshow(grayimg2)
# plt.imshow(grayimg3)
# plt.imshow(grayimg4)
plt.imshow(grayimg3)