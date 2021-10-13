from operator import eq
from PIL.Image import NONE
import cv2 as cv
from matplotlib.cbook import flatten
from matplotlib.image import imread
import numpy as np
from matplotlib import pyplot as plt
from numpy.__config__ import show
from numpy.core.fromnumeric import nonzero, resize

img = cv.imread("./img/test.jpg",0)

#matplotlib绘制hist
plt.hist(img.ravel(),256,[0,256])
plt.show()

#matplotlib法线图
img1 = cv.imread("./img/test.jpg")
color=('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

#opencv绘制hist
mask = np.zeros(img.shape[:2], np.uint8)
mask[:,:] = 255
mask_img = cv.bitwise_and(img,img,mask=mask)
hist = cv.calcHist([img],[0],None,[256],[0,256])
mask_hist = cv.calcHist([mask_img],[0],None,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(mask_img, 'gray')
plt.subplot(224), plt.plot(hist), plt.plot(mask_hist)
plt.xlim([0,256])
plt.show()


#直方图均衡化
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf*float(hist.max())/cdf.max()
plt.plot(cdf_normalized,color='r')
plt.hist(img.flatten(),256,[0,256],color='g')
plt.xlim([0,256])
plt.legend(("cdf","histogram"),loc = "upper left")
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype("uint8")
img2 = cdf[img]
hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])
cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2*float(hist2.max())/cdf2.max()
plt.plot(cdf_normalized2,color='r')
plt.hist(img2.flatten(),256,[0,256],color='g')
plt.xlim([0,256])
plt.legend(("cdf","histogram"),loc = "upper left")
plt.show()

#opencv中的直方图均衡化
equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imshow("opencv_hist",res)


#对比度受限的自适应直方图均衡
clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1=clahe.apply(img)
res = np.hstack((img,cl1))
cv.imshow("hist",res)

#绘制二维直方图
img3 = imread("./img/home.jpg")
hsv = cv.cvtColor(img3,cv.COLOR_BGR2HSV)
hist3 = cv.calcHist([hsv],[0,1],None,[100,256],[0,100,0,256])
# cv.imshow("hist_np",hist3)
# cv.waitKey(0)
plt.subplot(121),plt.imshow(img3)
plt.subplot(122),plt.imshow(hist3,interpolation = 'nearest')
plt.show()

#直方图反投影
img4 = cv.imread("./img/messi.jpg")
roi = cv.imread("./img/roi.jpg")
img4_hsv = cv.cvtColor(img4,cv.COLOR_BGR2HSV)
roi_hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
I = cv.calcHist([img4_hsv],[0,1],None,[180,256],[0,180,0,256])
M = cv.calcHist([roi_hsv],[0,1],None,[180,256],[0,180,0,256])

#numpy反投影算法
# R = M/I
# h,s,v = cv.split(img4_hsv)
# B = R[h.ravel(),s.ravel()]
# B = np.minimum(B,1)
# B = B.reshape(img4_hsv.shape[:2])

#opencv中的反投影
cv.normalize(M,M,0,255,cv.NORM_MINMAX)
B = cv.calcBackProject([img4_hsv],[0,1],M,[0,180,0,256],1)

disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv.normalize(B,B,0,255,cv.NORM_MINMAX)
ret,th = cv.threshold(B,50,255,0)
th = cv.merge((th,th,th))
res = cv.bitwise_and(img4,th)
res = np.hstack((img4,th,res))
cv.imshow("anti",res)
cv.waitKey(0)
