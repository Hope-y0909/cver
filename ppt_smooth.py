import cv2 as cv
import numpy as np
import time

img1 = cv.imread("img/logo.jpg")
img2 = cv.imread("img/test.jpg")
w,h = img1.shape[0:2]
img2 = cv.resize(img2,(h,w))

alpha=0
cv.namedWindow("ppt_smooth",True)
dst=cv.addWeighted(img1,alpha,img2,1-alpha,-1)
# cv.imshow("ppt_smooth",dst)
# cv.waitKey(0)

while alpha<1.0:
    dst=cv.addWeighted(img1,alpha,img2,1-alpha,-1)
    cv.imshow("ppt_smooth",dst)
    cv.waitKey(100)
    alpha+=0.1
cv.waitKey(0)
cv.destroyAllWindows()