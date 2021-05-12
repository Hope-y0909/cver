import cv2 as cv
import numpy as np
img = cv.imread("img/logo.jpg")
cv.imshow("hs",img)
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow("hsv",hsv)
low_blue = np.array([110, 50, 50])
high_blue = np.array([130, 255, 255])
mask_blue = cv.inRange(hsv,low_blue,high_blue)

low_red = np.array([0, 50, 50])
high_red = np.array([10, 255, 255])
mask_red = cv.inRange(hsv,low_red,high_red)

low_green = np.array([35, 50, 50])
high_green = np.array([77, 255, 255])
mask_green = cv.inRange(hsv,low_green,high_green)

mask = mask_red + mask_green + mask_blue
res = cv.bitwise_and(img, img, mask=mask)
cv.imshow("mask", mask)
cv.imshow("res", res)
cv.waitKey(0)
