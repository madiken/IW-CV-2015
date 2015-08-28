# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from skimage  import transform as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import numpy as np
import cv2

try:
    kitty = cv2.imread(".\IW2015\kitty.png")
    kitty = cv2.resize(kitty,(640, 480))
    pattern = cv2.imread(".\IW2015\pattern.png")
    
    ################################################################
    # Initiate STAR detector
    orb = cv2.ORB()
    
    # find the keypoints with ORB
    kp = orb.detect(pattern,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(pattern, kp)
    
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(pattern,kp,color=(0,255,0), flags=0)
    plt.imshow(img2),plt.show()
    
    ################################################################
    
    cap = cv2.VideoCapture(0)
    
    orb = cv2.ORB()
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    while(True):
        status, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # find the keypoints with ORB
        kpf = orb.detect(gray_frame,None)
        
        # compute the descriptors with ORB
        kpf, desf = orb.compute(gray_frame, kpf)
        
        img3 = cv2.drawKeypoints(frame,kpf,color=(0,255,0), flags=0)
        #############################################################
        
        #match section
        # create BFMatcher object
        
        # Match descriptors.
        matches = bf.match(des,desf)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        good = matches[:min(100, len(matches))]
        
        
        src_points =  np.float32([kp[m.queryIdx].pt for m in good])
        dst_points =  np.float32([kpf[m.trainIdx].pt for m in good])
        
        homography = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5)
        homography_inv = np.linalg.inv(homography[0])
        
        kitty_transf = (tf.warp(kitty, tf.ProjectiveTransform(homography_inv))*255).astype(np.uint8)
        kitty_mask = np.ones_like(kitty) - np.array(kitty_transf > 0, dtype = np.uint8)
        frame = frame * kitty_mask + kitty_transf
        cv2.imshow('edges',img3)
        cv2.imshow('frame',frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
except:
    print "Unexpected error:", sys.exc_info()
    cap.release()
    cv2.destroyAllWindows()

    