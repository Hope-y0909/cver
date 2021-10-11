import cv2 as cv
import numpy as np
from numpy.core.defchararray import center
img = cv.imread("./img/countour.png")
gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
ret,th = cv.threshold(gray,175,255,0)
contours, hierarchy = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
M = cv.moments(cnt)
print(M)
#质心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("质心：x:{:d} y:{:d}".format(cx,cy))
#轮廓面积
area = cv.contourArea(cnt)
print("面积：{:.3f}".format(area))
#轮廓周长
perimeter = cv.arcLength(cnt,True)
print("周长：{:.3f}".format(perimeter))
#轮廓近似
epsilon = 0.000001*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
cv.drawContours(img,approx,-1,(0,255,0),3)
cv.imshow("img",img)
#轮廓凸包
hull =cv.convexHull(cnt)
cv.drawContours(img,approx,-1,(0,0,255),3)
cv.imshow("img1",img)
#检查凸度 0 false 1 ture
k = cv.isContourConvex(cnt)
print("是否凸出：{:d}".format(k))
#边界矩形-直角矩形
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
cv.imshow("img2",img)
#边界矩形-旋转矩形
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(125,125,125),3)
cv.imshow("img3",img)
#最小闭合圈
(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv.circle(img,center,radius,(175,175,175),3)
cv.imshow("img4",img)
#拟合一个椭圆
ellipse =cv.fitEllipse(cnt)
cv.ellipse(img,ellipse,(0,175,0),3)
cv.imshow("img5",img)
#拟合一个直线
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt,cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx)+y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(175,0,0),3)
cv.imshow("img6",img)

cv.waitKey(0)