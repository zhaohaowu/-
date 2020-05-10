# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:40:10 2020

@author: zhaohaowu
"""


# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import glob

#1th
def getCameraCalibrationCoefficients(chessboardname, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(chessboardname)
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return
    
    ret_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Finde the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)
         
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs
nx = 9 
ny = 6
ret, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients('camera_cal/calibration*.jpg', nx, ny)

#2th
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)

# Read distorted chessboard image
# test_distort_image = cv2.imread('./camera_cal/calibration4.jpg')
# # Do undistortion
# test_undistort_image = undistortImage(test_distort_image, mtx, dist)
# plt.imshow(test_distort_image)

test_distort_image = cv2.imread('test_images/test2.jpg')
# Do undistortion
test_undistort_image = undistortImage(test_distort_image, mtx, dist)
plt.imshow(test_distort_image[:,:,::-1])