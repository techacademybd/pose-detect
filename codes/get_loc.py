import cv2
import numpy as np
import imutils
from helper import red_segment, green_segment, blue_segment

# pip install imutils
# pip install opencv-python

def read_contours(img):

    height, width = img.shape[:2]
    blacky = np.zeros((height, width, 3), np.uint8)
    red = red_segment(img)   
    cnts = cv2.findContours(red, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    location = []
    count = 0
    # loop over the contours
    for c in cnts[:4]:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        location.append((cX, cY))
        # draw the contour and center of the shape on the image
        #cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)
        id_c = str(count)
        cv2.putText(img, id_c, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        count+=1
        print(location)

    return img, red


#img = cv2.imread("frame.jpg")
#img, red = read_contours(img)

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
    height, width = frame.shape[:2]
    
    frame = cv2.resize(frame,(width//2, height//2), interpolation = cv2.INTER_CUBIC)
    # get h,w and then resize the frame
    img, red = read_contours(frame)
    cv2.imshow("Track", img)
    cv2.imshow("Original", red)
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

'''
cv2.imshow("original", img)
cv2.imshow("red", red)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''