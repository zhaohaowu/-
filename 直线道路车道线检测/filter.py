# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:01:33 2020

@author: zhaohaowu
"""
import cv2
import numpy as np
import matplotlib.image as mping
import matplotlib.pyplot as plt

img = cv2.imread("D:\\test3.png")

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 均值滤波
img_mean = cv2.blur(img, (5,5))

# 中值滤波
img_median = cv2.medianBlur(img, 5)

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img,(5,5),0)

titles = ['srcImg','mean','median','Gaussian']
imgs = [img, img_mean, img_median, img_Guassian]

plt.imshow(img)
# for i in range(4):
#     plt.subplot(2,2,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
#     plt.imshow(imgs[i])
#     plt.title(titles[i])
# plt.show()


# # 双边滤波
# img_bilater = cv2.bilateralFilter(img,9,75,75)

# # 展示不同的图片
# titles = ['srcImg','mean', 'Gaussian', 'median', 'bilateral']
# imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

# for i in range(5):
#     plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
#     plt.imshow(imgs[i])
#     plt.title(titles[i])
# plt.show()
