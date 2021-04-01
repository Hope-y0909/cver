import cv2 as cv
import numpy as np

# img1=cv.imread("img/1.jpg")
# img2=cv.imread("img/logo.jpg")
# img3=cv.resize(img2,(img1.shape[1],img1.shape[0]),cv.INTER_CUBIC)
# dst=cv.addWeighted(img1,0.7,img3,0.3,0)
# cv.imshow("dst",dst)
# cv.waitKey(0)

img1=cv.imread("img/1.jpg")
img2=cv.imread("img/logo.jpg")

rows,cols,channel=img2.shape
roi=img1[0:rows,0:cols]

img2gray=cv.cvtColor(img2,cv.COLOR_RGB2GRAY)
ret,mask=cv.threshold(img2gray,10,255,cv.THRESH_BINARY)
cv.imshow("mask",mask)
mask_inv=cv.bitwise_not(mask)
cv.imshow("mask1",mask_inv)
img1_bg=cv.bitwise_and(roi,roi,mask=mask_inv)
img2_fg=cv.bitwise_and(img2,img2,mask=mask)

dst=cv.add(img1_bg,img2_fg)
img1[0:rows,0:cols]=dst
cv.imshow("img",img1)
cv.waitKey(0)