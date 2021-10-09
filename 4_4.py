import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread("./img/logo.jpg")
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img, -1,kernel)
blur = cv.blur(img,(5,5))
gassian_blur = cv.GaussianBlur(img,(5,5),0)
middle_blur = cv.medianBlur(img,5)
bilateral_filter = cv.bilateralFilter(img,9,75,75)
plt.subplot(231),plt.imshow(img),plt.title("original")
plt.xticks([]),plt.yticks([])
plt.subplot(232),plt.imshow(dst),plt.title("averaging")
plt.xticks([]),plt.yticks([])
plt.subplot(233),plt.imshow(blur),plt.title("blur")
plt.xticks([]),plt.yticks([])
plt.subplot(234),plt.imshow(gassian_blur),plt.title("gassin_blur")
plt.xticks([]),plt.yticks([])
plt.subplot(235),plt.imshow(middle_blur),plt.title("middle_blur")
plt.xticks([]),plt.yticks([])
plt.subplot(236),plt.imshow(bilateral_filter),plt.title("bilateral_filter")
plt.xticks([]),plt.yticks([])
plt.show()