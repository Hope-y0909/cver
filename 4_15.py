import cv2 as cv
import numpy as np
from numpy.core.defchararray import _multiply_dispatcher

img = cv.imread("./img/coins.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
ret ,th = cv.threshold(gray,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
#形态学操作
kernal = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(th,cv.MORPH_OPEN,kernal,iterations=2)
bg = cv.dilate(opening,kernal,iterations=3)
dist_trans = cv.distanceTransform(opening,cv.DIST_L2,5)
ret,fg = cv.threshold(dist_trans,0.7*dist_trans.max(),255,0)
fg = np.uint8(fg)
unknow = cv.subtract(bg,fg)
ret,markers = cv.connectedComponents(fg)
markers = markers+1
markers[unknow==255] = 0
# b = markers.astype(np.uint8)
# b = cv.applyColorMap(b, cv.COLORMAP_JET)
# cv.imshow("jet color",b)
markers = cv.watershed(img,markers)
img[markers == -1] = [127,175,100]
cv.imshow("res",img)
res = np.hstack((th,opening,bg,dist_trans,fg,unknow,markers))
cv.imshow("process",res)
cv.waitKey(0)
