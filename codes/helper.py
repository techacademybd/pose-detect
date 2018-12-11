import cv2
import numpy as np
import imutils

# pip install imutils
# pip install opencv-python

def roi(frame):
  # ROI 
  upper_left = (0, 0)
  bottom_right = (360, 640)
  frame = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
  return frame


def frame2thresh(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  blurred = cv2.GaussianBlur(gray, (11, 11), 0)
  thresh = cv2.threshold(blurred, 50, 100, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(mask, None, iterations=1)
  thresh = cv2.dilate(mask, None, iterations=3)
  
  return thresh


def red_segment(frame): 
    # HSV values for red
    lower_red = np.array([80, 100, 170])
    upper_red = np.array([150, 150, 255])

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([30,150,50])
    #upper_red = np.array([255,200,180])
    
    # color detection
    mask = cv2.inRange(frame, lower_red, upper_red)
    proc = cv2.bitwise_and(frame, frame, mask=mask)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(mask, None, iterations=1)
    red_seg = cv2.dilate(mask, None, iterations=3)
    
    return red_seg


def blue_segment(frame): 
    
    # convert to hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV values for blue
    lower_blue= np.array([78,158,124])
    upper_blue = np.array([138,255,255])
    # color detection
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    proc = cv2.bitwise_and(frame, frame, mask=mask)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(mask, None, iterations=1)
    blue_segment = cv2.dilate(mask, None, iterations=3)
    
    return blue_segment


def green_segment(frame):
    
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV values for green
    lower_green = np.array([65,60,60])
    upper_green = np.array([80,255,255])
    # color detection
    mask = cv2.inRange(hsv, lower_green, upper_green)
    proc = cv2.bitwise_and(frame, frame, mask=mask)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(mask, None, iterations=1)
    green_segment = cv2.dilate(mask, None, iterations=3)
    
    return green_segment
