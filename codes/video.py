import cv2
import numpy as np
import imutils
from helper import red_segment, green_segment, blue_segment

# pip install imutils
# pip install opencv-python

# load the video
# script and video should be in same folder
cap = cv2.VideoCapture("demo.mp4")

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