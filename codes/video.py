import cv2
import numpy as np
import imutils

# pip install imutils
# pip install opencv-python

# load the video
# script and video should be in same folder
cap = cv2.VideoCapture("demo.mp4")


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
  
  #thresh = cv2.erode(mask, None, iterations=1)
  #thresh = cv2.dilate(mask, None, iterations=3)
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
    
    # bitwise mask and original image
    # res = cv2.bitwise_and(frame,frame, mask= mask)
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
  

# check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  # capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # rotate frame
    frame = imutils.rotate(frame, -90)
    # get h,w and then resize the frame
    height, width = frame.shape[:2]
    frame = cv2.resize(frame,(width//2, height//2), interpolation = cv2.INTER_CUBIC)

    # make dummy black frame to overlay outputs
    blacky = np.zeros((height//2, width//2, 3), np.uint8)
    
    # get red segnmented frame and draw contours on original frame
    red = red_segment(frame)   
    _, contours_red, _ = cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours_red[5]
    cv2.drawContours(blacky, contours_red, -1, (0, 0, 255), 2)
    
    # get blue segment and draw contours
    blue = blue_segment(frame)
    _, contours_blue, _ = cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blacky, contours_blue, -1, (255, 0, 0), 2)
    
    # get green segment and draw contours
    green = green_segment(frame)
    _, contours_green, _ = cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blacky, contours_green, -1, (0, 255, 0), 2)
    
    cv2.imshow("Track", blacky)
    cv2.imshow("Original", frame)
    #cv2.imshow("Green", green)
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # break the loop
  else: 
    break
 
# when everything done, release the video capture object
cap.release()
# closes all the frames
cv2.destroyAllWindows()