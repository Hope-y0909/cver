import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#opencv中的霍夫曼变换

img = cv.imread("./img/sudoku.png")
img1 = img.copy()
gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
edges = cv.Canny(gray,50,150,apertureSize=3)
lines = cv.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(255,0,0),3)
cv.imshow("img",img)


#概率霍夫曼变换

lines1 = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines1:
    x1,y1,x2,y2 = line[0]
    cv.line(img1,(x1,y1),(x2,y2),(255,0,0),3)
cv.imshow("img1",img1)

#霍夫圈变换
img3 = cv.imread("./img/opencv-logo-white.png",0)
img3 = cv.medianBlur(img3,5)
cimg = cv.cvtColor(img3,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img3,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for cen in circles[0,:]:
    cv.circle(cimg,(cen[0],cen[1]),cen[2],(0,255,0),3)
    cv.circle(cimg,(cen[0],cen[1]),2,(0,0,255),3)
cv.imshow("img3",cimg)
cv.waitKey(0)