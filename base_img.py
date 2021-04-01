import cv2 as cv
import numpy as np

# img = cv.imread("img/1.jpg")#imread b,g,r
# cv.imshow("im",img)
# px = img[100,100]
# print(px)
# blue = img[100,100,0]
# print(blue)
# img[100,100] = [0,0,0]
# cv.imshow("img",img)
# cv.waitKey(0)

#better at piexl
# print(img.item(100,100,0))
# img.itemset((100,100,0),0)
# print(img.item(100,100,0))

# print(img.shape)
# print(img.size)
# print(img.dtype)

# #crop roi and copy roi to img
# roi = img[100:200,100:150]
# img[300:400,500:550] = roi
# cv.imshow("roi",roi)
# cv.imshow("img",img)
# cv.waitKey(0)

# #split and merge
# [b,g,r]=cv.split(img)
# cv.imshow("R",r)
# cv.imshow("G",g)
# cv.imshow("B",b)
# cv.waitKey(0)
# img1 = cv.merge((b,g,r))

#set img border
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img = plt.imread("img/1.jpg")
replicate = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

