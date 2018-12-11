import cv2
import numpy as np
import imutils
from helper import red_segment, green_segment, blue_segment

# pip install imutils
# pip install opencv-python

img = cv2.imread("frame.jpg")
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
#cv2.imshow("original", img)
cv2.imshow("red", red)
cv2.waitKey(0)
cv2.destroyAllWindows()