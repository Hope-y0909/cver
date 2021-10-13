import cv2 as cv
import numpy as np
img = cv.imread("./img/ABC.jpg",0)
ret, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(th, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)#[Next, Previous, First_Child, Parent]
cv.drawContours(img,contours,-1,(175,0,255),3)
cv.imshow("img",img)
cv.waitKey(0)
