import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os

import time
import cv2
import scipy
import numpy as np
from skimage.metrics import structural_similarity as ssim


def cal_ssim(img1, img2, win_size=11, sigma=1.5):
    res = ssim(img1, img2, win_size=win_size, sigma=sigma, multichannel=True)
    return res

# TODO
# 1. 确定视频的起点和终点
# 2. 确定图中的区域大小 (256, 256) ; 或者划一片区域然后能够转成256的
# 3. 添加一个功能，隔一段时间保存图片

### Function to Crop Images ###
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

def cut_video(clear_video_name, shadow_video_name):
    global refPt
    H_matrix = np.array([[1.00476859e+00,-3.45690534e-02,-2.62218249e+01],
                        [3.57667123e-02, 9.92150127e-01,-4.21958744e+00],
                        [1.06401679e-04, -8.28318658e-05, 1.0]])
    ### 裁剪原始的video data ###
    frame_clear, frame_shadow = read_video(clear_video_name, shadow_video_name)

    transform_video_name = os.path.dirname(clear_video_name) + "/left_trans.avi"
    frame_transform = read_single_video(transform_video_name)
    assert len(frame_transform) == len(frame_clear)

    cap = cv2.VideoCapture(clear_video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 先确定区域
    image = frame_shadow[0]
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
    cv2.imshow("image", image)

    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        print("ROI is:({}, {}) to ({}, {})".format(refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1]))
        cv2.imshow("ROI", roi)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

    # 计算一下transform之后的shadow_image四个角点
    # TODO
    # point_left_up = H_matrix.dot(np.array([refPt[0][0], refPt[0][1], 1.0]).transpose())
    # point_left_down = H_matrix.dot(np.array([refPt[0][0], refPt[1][1], 1.0]).transpose())
    # point_right_up = H_matrix.dot(np.array([refPt[1][0], refPt[0][1], 1.0]).transpose())
    # point_right_down = H_matrix.dot(np.array([refPt[1][0], refPt[1][1], 1.0]).transpose())


    # 展示区域内的image，并且同时确定起点和终点     
    cut_video_flag = False
    print("original image shape:({}, {})".format(frame_clear[0].shape[0], frame_clear[0].shape))
    image_width = frame_clear[0].shape[1]
    image_height = frame_clear[0].shape[0]
    # 防止越界
    start_x = refPt[0][0]
    start_y = refPt[0][1]
    if refPt[1][0] >= image_width:
        end_x = image_width
    else:
        end_x = refPt[1][0] + 1

    if refPt[1][1] >= image_height:
        end_y = image_height
    else:
        end_y = refPt[1][1] + 1

    for i in range(len(frame_clear)):
        clear_image = frame_clear[i][start_x:end_x, start_y:end_y]
        shadow_image = frame_shadow[i][start_x:end_x, start_y:end_y]
        img = cv2.hconcat([clear_image, shadow_image])
        cv2.imshow('clear_frame', img)
        cv2.imshow('shadow_frame', shadow_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S') and cut_video_flag == False:
            print("Start to cut video idx:{}".format(i))
            cut_video_flag = True
        elif key == ord('q') or key == ord('Q') and cut_video_flag == True:
            break

        if cut_video_flag:
    #         # save video
            cv2.imwrite(os.path.dirname(clear_video_name) + "/clear/clear_" + str(i) + ".jpg", clear_image)
            cv2.imwrite(os.path.dirname(shadow_video_name) + "/shadow/shadow_" + str(i) + ".jpg", shadow_image)

    refPt.clear()

def read_single_video(file_name):
    cap = cv2.VideoCapture(file_name)
    # read clear video
    frame_clear = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_clear.append(frame)
            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_clear
    
def read_video(clear_video_name, shadow_video_name):
    clear_video = cv2.VideoCapture(clear_video_name)
    shadow_video = cv2.VideoCapture(shadow_video_name)

    # read clear video
    frame_clear = []
    while (clear_video.isOpened()):
        ret, frame = clear_video.read()
        if ret == True:
            frame_clear.append(frame)
            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    clear_video.release()

    # read shadow video
    frame_shadow = []
    while (shadow_video.isOpened()):
        ret, frame = shadow_video.read()
        if ret == True:
            frame_shadow.append(frame)
            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    shadow_video.release()
    cv2.destroyAllWindows()
    # return all frames
    assert len(frame_shadow) == len(frame_clear) and len(frame_shadow) > 0
    
    return frame_clear, frame_shadow

def estimate_video_ssim(file_name):
    #### calculate ssim  ####
    total_ssim = 0
    for i in range(len(file_name)):
        frame_clear, frame_shadow = read_video(file_name[i][0], file_name[i][1])
        cur_ssim = 0
        for i in range(len(frame_shadow)):
            cur_ssim += cal_ssim(frame_shadow[i], frame_clear[i])
        cur_ssim /= len(frame_shadow)
        print("i:{}\tAverage ssim is:{}".format(i, cur_ssim))
        total_ssim += cur_ssim
    total_ssim /= len(file_name)
    print("Total average ssim is:{}".format(total_ssim))
    return cur_ssim

if __name__ == "__main__":
    #### Read files ####
    # motion_name = ['clap_hands', 'jump', 'lunge', 'stretch', 'wave']
    # file_name = []
    # for i in range(len(motion_name)):
    #     clear_file = file_base + "/" + motion_name[i] + "/clear.avi"
    #     shadow_file = file_base + "/" + motion_name[i] + "/shadow_original.avi"
    #     file_name.append((clear_file, shadow_file))

    file_base = "/home/yifan/Documents/Research/ShadowData/medium_ssim"

    file_name = []
    # add file names
    
    file_name.append((file_base + "/left.avi", file_base + "/ref.avi"))
    cut_video(file_name[0][0], file_name[0][1])

