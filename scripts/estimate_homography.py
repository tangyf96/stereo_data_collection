# coding: UTF-8
import sys
import cv2
import numpy as np
import glob
import natsort
from pupil_apriltags import Detector

import pickle

file_name = natsort.natsorted(glob.glob("/home/yifan/Documents/calibrate_myntAI/*.jpg"), reverse = False)

img_pair = []
for i in range(len(file_name)//2):
    # find left
    left_name = file_name[i]
    # find right
    right_name = file_name[i+len(file_name)//2]

    img_left = cv2.imread(left_name, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_name, cv2.IMREAD_GRAYSCALE)

    # append to pair list
    img_pair.append((img_left, img_right))

src = []  # left
dst = []  # right
detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
for pair in img_pair:
    left = pair[0]
    right = pair[1]
    tag_left = detector.detect(left)
    tag_right = detector.detect(right)
    if len(tag_left) != len(tag_right):
        continue
    print("%d apriltags have been detected."%len(tag_left))
    for tag in tag_left:
        # cv2.circle(left, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
        # cv2.circle(left, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
        # cv2.circle(left, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
        # cv2.circle(left, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom
        src.append(tag.corners[0].astype(int))
        src.append(tag.corners[1].astype(int))
        src.append(tag.corners[2].astype(int))
        src.append(tag.corners[3].astype(int))
    
    # cv2.imshow("left", left)
    
    print("%d apriltags have been detected."%len(tag_right))
    for tag in tag_right:
        # cv2.circle(right, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-top
        # cv2.circle(right, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
        # cv2.circle(right, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
        # cv2.circle(right, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom
        dst.append(tag.corners[0].astype(int))
        dst.append(tag.corners[1].astype(int))
        dst.append(tag.corners[2].astype(int))
        dst.append(tag.corners[3].astype(int))

    # cv2.imshow("right",right)
    # cv2.waitKey(10)

array_src = np.asarray(src)
array_dst = np.asarray(dst)
h, status = cv2.findHomography(array_src, array_dst)
print(h)
# check if homography is correct
error = 0
idx = 0
for pair in img_pair:
    left = pair[0]
    right = pair[1]

    transform_img = cv2.warpPerspective(left, h, (left.shape[1], left.shape[0]))
    tags = detector.detect(transform_img)
    corner = []
    for tag in tags:
        corner.append(tag.corners[0].astype(int))
        corner.append(tag.corners[1].astype(int))
        corner.append(tag.corners[2].astype(int))
        corner.append(tag.corners[3].astype(int))
    corner = np.asarray(corner)

    tags = detector.detect(right)
    ref_corner = []
    for tag in tags:
        ref_corner.append(tag.corners[0].astype(int))
        ref_corner.append(tag.corners[1].astype(int))
        ref_corner.append(tag.corners[2].astype(int))
        ref_corner.append(tag.corners[3].astype(int))

    ref_corner = np.asarray(ref_corner)
    cur_error = np.linalg.norm(ref_corner - corner) / ref_corner.shape[0]
    error += cur_error
    print("l2 average is:{}".format(cur_error))

    cv2.imwrite("right_"+str(idx) + ".jpg", right)
    cv2.imwrite("trans_left" + str(idx) + ".jpg", transform_img)
    cv2.imwrite("left" + str(idx) + ".jpg", left)
    idx += 1

print("total error:{}".format(error/len(img_pair)))
print("Done")
