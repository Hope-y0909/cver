import numpy as np
import cv2 as cv
from numpy.lib.histograms import _histogram_bin_edges_dispatcher
img = cv.imread("./img/1.jpg")
height,width,ch = img.shape#img.shape[:2]
print(width,height)
cv.imshow("img", img)
res = cv.resize(img,None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
# width1,heigth1 = res.shape[:2]
# print(width1,heigth1)

M = np.float32([[1,0,100],[0,1,50]])
res1 = cv.warpAffine(img,M,(width, height))
cv.imshow("res1", res1)

M = cv.getRotationMatrix2D(((width-1)/2.0, (height-1)/2.0), 90, 1)
res2 = cv.warpAffine(img, M, (width, height))
cv.imshow("res2",res2)
cv.waitKey(0)
cv.destroyAllWindows()