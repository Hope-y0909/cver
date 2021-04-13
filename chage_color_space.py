import cv2 as cv
import numpy as np
cap=cv.VideoCapture("img/test.mp4")

while(1):

    _, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    lower_green = np.array([50, 50, 120])
    upper_green = np.array([70, 255, 255]) 
    green_mask = cv.inRange(hsv, lower_green, upper_green)


    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
    mask = blue_mask + green_mask


    res = cv.bitwise_and(frame,frame, mask= mask)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()