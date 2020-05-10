#coding = utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping


img = cv2.imread("D:\\test3.png")

#canny算子
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaus = cv2.GaussianBlur(img,(5,5),0)
canny_edge = cv2.Canny(gaus, 90, 180, apertureSize=3)
canny_edge = cv2.cvtColor(canny_edge,cv2.COLOR_BGR2RGB)
plt.imshow(canny_edge)

# Sobel算子
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# absX = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
# scaled_sobel = np.uint8(255*absX/np.max(absX))
# absX = np.zeros_like(scaled_sobel)
# absX[(scaled_sobel >= 30) & (scaled_sobel <= 90)] = 1
# absX = np.dstack((absX, absX, absX))*255
# absY = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
# scaled_sobel = np.uint8(255*absY/np.max(absY))
# absY = np.zeros_like(scaled_sobel)
# absY[(scaled_sobel >= 30) & (scaled_sobel <= 90)] = 1
# absY = np.dstack((absY, absY, absY))*255
# sobel_edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# plt.imshow(sobel_edge)

 
# Roberts算子
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# grayImage = cv2.cvtColor(grayImage,cv2.COLOR_BGR2RGB)
# kernelx = np.array([[-1,0],[0,1]], dtype=int)
# kernely = np.array([[0,-1],[1,0]], dtype=int)
# x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
# y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# #转uint8 
# absX = cv2.convertScaleAbs(x)      
# absY = cv2.convertScaleAbs(y)    
# Roberts = cv2.addWeighted(absX,0.5,absY,0.5,0)
# plt.imshow(Roberts)

# Sobel算子
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# grayImage = cv2.cvtColor(grayImage,cv2.COLOR_BGR2RGB)
# x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0) #对x求一阶导
# y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1) #对y求一阶导
# absX = cv2.convertScaleAbs(x)      
# absY = cv2.convertScaleAbs(y)    
# Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# plt.imshow(absY)

# Prewitt算子
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# grayImage = cv2.cvtColor(grayImage,cv2.COLOR_BGR2RGB)
# kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
# kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
# x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
# y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# #转uint8
# absX = cv2.convertScaleAbs(x)       
# absY = cv2.convertScaleAbs(y)    
# Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)
# plt.imshow(Prewitt)


# # #用来正常显示中文标签
# # plt.rcParams['font.sans-serif']=['SimHei']
# # #显示图形
# # titles = [u'原始图像', u'Roberts算子']  
# # images = [lenna_img, Roberts]  
# # for i in range(2):  
# #    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
# #    plt.title(titles[i])  
# #    plt.xticks([]),plt.yticks([])  
# # plt.show()






