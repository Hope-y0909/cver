import cv2 as cv
import numpy as np
from numpy.core.defchararray import rsplit
img = cv.imread("img/logo.jpg")
cv.imshow("img",img)
h,w = img.shape[:2]
res = cv.resize(img,(int(0.5*w), int(0.5*h)),interpolation=cv.INTER_CUBIC)
cv.imshow("resize",res)

M=np.float32([[1,0,100],[0,1,50]])
res = cv.warpAffine(img, M, (w,h))
cv.imshow("warpaffine",res)

M = cv.getRotationMatrix2D((w/2, h/2), 45, 1)
res = cv.warpAffine(img,M,(w,h))
cv.imshow("rotate", res)

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
res=cv.warpAffine(img,M,(w,h))
cv.imshow("affinetrans",res)
cv.waitKey(0)