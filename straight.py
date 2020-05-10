# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import glob
from moviepy.editor import VideoFileClip
import matplotlib.image as mping

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

def  process_img(img):
    
   
    test_distort_image = img
    test_undistort_image = cv2.undistort(test_distort_image, mtx, dist, None, mtx)
    
    gray = cv2.cvtColor(test_undistort_image, cv2.COLOR_RGB2GRAY)
 
    gaus = cv2.GaussianBlur(gray,(3,3),0.0)

    edges = cv2.Canny(gaus, 90, 180, apertureSize=3)

    h = img.shape[0]
    w = img.shape[1]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask , np.array([[(0,h),(3*w/8,h/2),(w*5/8,h/2),(w,h)]],np.int32), 255)
    #cv2.fillPoly(mask , np.array([[(0,h),(w/2,h/2),(w,h)]],np.int32), 255)
    roi = cv2.bitwise_and(edges, mask)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=10)

    # for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(img, (x1, y1), (x2, y2),(255,0,0), 3)

    left_x,left_y,right_x,right_y=[],[],[],[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y2 - y1 != 0 and x2 - x1 != 0:
                slop = (y2-y1)/(x2-x1)
                if abs(slop) > 0.5:
                    if slop<0:
                        left_x.extend([x1,x2])
                        left_y.extend([y1,y2])
                    else:
                        right_x.extend([x1,x2])
                        right_y.extend([y1,y2])

    #print(left_x,left_y,right_x,right_y)

    left = np.polyfit(left_y,left_x,1)
    poly_left = np.poly1d(left)
    left_y_start = int(img.shape[0])
    left_y_end = int(img.shape[0]*(3/5))
    left_x_start = int(poly_left(left_y_start))
    left_x_end = int(poly_left(left_y_end))

    right = np.polyfit(right_y,right_x,1)
    poly_right = np.poly1d(right)
    right_y_start = int(img.shape[0])
    right_y_end = int(img.shape[0]*(3/5))
    right_x_start = int(poly_right(right_y_start))
    right_x_end = int(poly_right(right_y_end))

    lines = [[[left_x_start, left_y_start, left_x_end,left_y_end],[right_x_start,right_y_start,right_x_end,right_y_end]]]

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
  
    return img
image = mping.imread('c/534.jpg')
image = process_img(image)
plt.imshow(image)


output = 'out_straight.mp4'#ouput video
clip = VideoFileClip("straight.mp4")#input video
out_clip = clip.fl_image(process_img)#对视频的每一帧进行处理
out_clip.write_videofile(output, audio=True)#将处理后的视频写入新的视频文件
