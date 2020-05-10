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

# # 读取视频
# cap = cv2.VideoCapture("harder_challenge_video.mp4")
# # 获取FPS(每秒传输帧数(Frames Per Second))
# fps = cap.get(cv2.CAP_PROP_FPS)
# # 获取总帧数
# totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(fps)
# print(totalFrameNumber)
# # 当前读取到第几帧
# COUNT = 0
# # 若小于总帧数则读一帧图像
# while COUNT < totalFrameNumber:
#     # 一帧一帧图像读取
#     ret, frame = cap.read()
#     # 把每一帧图像保存成jpg格式（这一行可以根据需要选择保留）
#     cv2.imwrite(str(COUNT) + '.jpg', frame)
#     # 显示这一帧地图像
#     cv2.imshow('video', frame)
#     COUNT = COUNT + 1
#     # 延时一段33ms（1s➗30帧）再读取下一帧，如果没有这一句便无法正常显示视频
#     cv2.waitKey(33)
# cap.release();

def  process_img(img):  
    
    test_distort_image = img
    image_undistorted = cv2.undistort(test_distort_image, mtx, dist, None, mtx)
    
    src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])
    image_size = (image_undistorted.shape[1], image_undistorted.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(image_undistorted, M,image_size, flags=cv2.INTER_LINEAR) 
    
    thresh1=(220, 255)
    hls = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    hls_l = np.zeros_like(l_channel)
    hls_l[(l_channel > thresh1[0]) & (l_channel <= thresh1[1])] = 1   
    thresh2=(156, 255)
    lab=cv2.cvtColor(warped_image, cv2.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]
    # if np.max(b_channel) > 100:
    #     b_channel = b_channel*(255/np.max(b_channel))
    lab_b = np.zeros_like(b_channel)
    lab_b[((b_channel > thresh2[0]) & (b_channel <= thresh2[1]))] = 1  
    combined_binary = np.zeros_like(hls_l)
    combined_binary[(hls_l == 1) | (lab_b == 1)] = 1
     
    left_fit = []
    right_fit = []
    ploty = []
    histogram2 = np.sum(combined_binary[combined_binary.shape[0]//2:,:], axis=0)
    out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    midpoint = np.int(histogram2.shape[0]/2)
    leftx_base = np.argmax(histogram2[:midpoint])
    rightx_base = np.argmax(histogram2[midpoint:])+midpoint
    nwindows = 9
    window_height = np.int(combined_binary.shape[0]/nwindows)
    nonzero = combined_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = [] 
    for window in range(nwindows):
        win_y_low = combined_binary.shape[0]-(window+1)*window_height
        win_y_high = combined_binary.shape[0]-window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))   
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # margin2 = 60
    # left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    #                 left_fit[2] - margin2)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    #                 left_fit[1]*nonzeroy + left_fit[2] + margin2)))
    # right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    #                 right_fit[2] - margin2)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    #                 right_fit[1]*nonzeroy + right_fit[2] + margin2)))
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds] 
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)
    # ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    # window_img = np.zeros_like(out_img)
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]     
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin2, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin2, ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin2, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin2, ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # result1 = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)     
                                               
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (combined_binary.shape[1], image_undistorted.shape[0])) 
    result2 = cv2.addWeighted(image_undistorted, 1, newwarp, 0.3, 0)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])    
    curvature = ((left_curverad + right_curverad) / 2)
    lane_width = np.absolute(leftx[671] - rightx[671])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((leftx[671] + rightx[671]) * lane_xm_per_pix) / 2.)
    cen_pos = ((combined_binary.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = veh_pos - cen_pos
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm" % (round(curvature))
    if distance_from_center > 0:
        pos_flag = 'right'
    else:
        pos_flag = 'left'
    cv2.putText(result2, radius_text, (100, 100), font, 2, (255, 255, 255), 2)
    center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
    cv2.putText(result2, center_text, (100, 170), font, 2, (255, 255, 255), 2)   
               
    return result2
img = mping.imread('c/534.jpg')   #a554  a319  a531  a1036  a756  c534
image = process_img(img)
plt.imshow(image)


# output = 'out_curve.mp4'#ouput video
# clip = VideoFileClip("curve.mp4")#input video
# out_clip = clip.fl_image(process_img)#对视频的每一帧进行处理
# out_clip.write_videofile(output, audio=True)#将处理后的视频写入新的视频文件
    
    
    
    
    
    
    
    
    
    