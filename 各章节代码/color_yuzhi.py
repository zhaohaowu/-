# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:04:48 2020

@author: zhaohaowu
"""


# coding = utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import glob
from moviepy.editor import VideoFileClip

objp = np.zeros((6 * 9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob("camera_cal/calibration*.jpg")
if len(images) > 0:
    print("images num for calibration : ", len(images))
ret_count = 0
for idx, fname in enumerate(images):
    img2 = cv2.imread(fname)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_size = (img2.shape[1], img2.shape[0])
    # Finde the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        ret_count += 1
        objpoints.append(objp)
        imgpoints.append(corners)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
print('Do calibration successfully')

# video_input = 'project_video.mp4'
# cap = cv2.VideoCapture(video_input)
# count = 1
# while(True):
#     ret, image = cap.read()
#     if ret:
#         undistort_image = undistortImage(image, mtx, dist)
#         cv2.imwrite('a/'+str(count) + '.jpg', undistort_image)
#         count += 1
#     else:
#         break
# cap.release()

def  process_img(img):  
    
    test_distort_image = img
    image_undistorted = cv2.undistort(test_distort_image, mtx, dist, None, mtx)
    
    src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])
    image_size = (image_undistorted.shape[1], image_undistorted.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(image_undistorted, M,image_size, flags=cv2.INTER_LINEAR)
    
    # thresh1=(220, 255)
    # hls = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
    # l_channel = hls[:,:,1]
    # hls_l = np.zeros_like(l_channel)
    # hls_l[(l_channel > thresh1[0]) & (l_channel <= thresh1[1])] = 1   
    
    # thresh2=(195, 255)
    # lab=cv2.cvtColor(warped_image, cv2.COLOR_RGB2LAB)
    # b_channel = lab[:,:,2]
    # if np.max(b_channel) > 100:
    #     b_channel = b_channel*(255/np.max(b_channel))
    # lab_b = np.zeros_like(b_channel)
    # lab_b[((b_channel > thresh2[0]) & (b_channel <= thresh2[1]))] = 1  
    # combined_binary = np.zeros_like(hls_l)
    # combined_binary[(hls_l == 1) | (lab_b == 1)] = 1
    
    thresh1=(220, 255)
    hls = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    hls_l = np.zeros_like(l_channel)
    hls_l[(l_channel > thresh1[0]) & (l_channel <= thresh1[1])] = 1
    combined_binary = np.zeros_like(hls_l)
    combined_binary[(hls_l == 1)] = 1
    
    # thresh2=(150, 255)
    # lab=cv2.cvtColor(warped_image, cv2.COLOR_RGB2LAB)
    # b_channel = lab[:,:,2]
    # lab_b = np.zeros_like(b_channel)
    # lab_b[((b_channel > thresh2[0]) & (b_channel <= thresh2[1]))] = 1
    # combined_binary = np.zeros_like(lab_b)
    # combined_binary[(lab_b == 1)] = 1
    
    out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    
    return out_img
img = mping.imread('test_images/test2.jpg')
image = process_img(img)
plt.imshow(image)