import cv2 as cv
import numpy as np
img=cv.imread("./img/logo.jpg")
imggray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
ret,th=cv.threshold(imggray,127,255,0)
contours,hierarchy = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
contours1,hierarchy1 = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img,contours,-1,(175,255,175),3)
cv.drawContours(img,contours1,-1,(125,125,125),3)
cv.imshow("img",img)
cv.waitKey(0)