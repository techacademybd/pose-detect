import cv2
import numpy as np
import imutils

# pip install imutils
# pip install opencv-python

# load the video
# script and video should be in same folder
cap = cv2.VideoCapture("demo.mp4")

# HSV values for blue
lower_red = np.array([80, 50, 0])
upper_red = np.array([255, 150, 255])

# check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  # capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    # rotate frame
    frame = imutils.rotate(frame, -90)
    
    # resize the frame
    height, width = frame.shape[:2]
    frame = cv2.resize(frame,(width//2, height//2), interpolation = cv2.INTER_CUBIC)

    mask = cv2.inRange(frame, lower_red, upper_red)
    proc = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Processed", proc)

    
    # Display the resulting frame
    cv2.imshow('Frame',frame)

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
