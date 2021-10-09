import cv2 as cv
import numpy as np
from numpy.core.defchararray import upper
cap = cv.VideoCapture("rtsp://192.168.12.173/live/2/2")
while 1:
    img, frame = cap.read()
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    # green
    lower_green = np.array([35,43,46])
    upper_green = np.array([77,255,255])
    mask_G = cv.inRange(hsv, lower_green, upper_green)
    # res = cv.bitwise_and(frame,frame,mask=mask_G)
    # blue
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask_B = cv.inRange(hsv, lower_blue, upper_blue)
    # red
    lower_red = np.array([0,43,46])
    upper_red = np.array([20,255,255])
    mask_R = cv.inRange(hsv, lower_blue, upper_blue)

    mask = mask_R + mask_G + mask_B
    res = cv.bitwise_and(frame,frame,mask=mask)
    cv.imshow("frame",frame)
    cv.imshow("mask",mask)
    cv.imshow("res",res)
    k = cv.waitKey(5)&0xff
    if k == 27:
        break
cv.destroyAllWindows()