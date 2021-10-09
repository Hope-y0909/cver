import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt
from numpy.lib.shape_base import tile
img = cv.imread("./img/1.jpg",0)
ret, thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
title = ['original','binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
images = [img, thresh1, thresh2, thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()

ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

imgs = [img,0,th1,
        img,0,th2,
        blur,0,th3]
titles = ['original noisy image','hostogram', 'global thresholding(v=127)',
          'original noisy image','hostogram', 'otsu thresholding',
          'gaussian filter image', 'hostogram', 'otsu threshingh']
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(imgs[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(imgs[i*3].ravel(),256)
    plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(imgs[i*3+2],'gray')
    plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
plt.show()