import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("./img/test_0.jpg")
laplcian = cv.Laplacian(img,cv.CV_64F)
sobel_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobel_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
sobel = cv.Sobel(sobel_x,cv.CV_64F,0,1,ksize=5)
plt.subplot(231),plt.imshow(img),plt.title("original")
plt.xticks([]),plt.yticks([])
plt.subplot(232),plt.imshow(laplcian),plt.title("laplcian")
plt.xticks([]),plt.yticks([])
plt.subplot(234),plt.imshow(sobel_x),plt.title("sobel_x")
plt.xticks([]),plt.yticks([])
plt.subplot(235),plt.imshow(sobel_y),plt.title("sobel_y")
plt.xticks([]),plt.yticks([])
plt.subplot(236),plt.imshow(sobel),plt.title("sobel")
plt.xticks([]),plt.yticks([])
plt.show()

sobel_8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
sobel_64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.abs(sobel_64f)
sobel_8uu = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8uu,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()

