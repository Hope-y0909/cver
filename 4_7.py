import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# img = cv.imread("./img/1.jpg",0)
# edges = cv.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap="gray"),plt.title("original")
# plt.subplot(122),plt.imshow(edges),plt.title("edges")
# plt.show()
T=[100,200]
def edge1(x):
    T[0] = x 
def edge2(y):
    T[1] = y
img = cv.imread("./img/1.jpg",0)
cv.namedWindow("imshow")
cv.createTrackbar("minval","imshow",0,255,edge1)
cv.createTrackbar("maxval","imshow",0,255,edge2)

while True:
    edges = cv.Canny(img,T[0],T[1])
    cv.imshow("imshow",edges)
    k = cv.waitKey(1)&0xFF
    if k == 27:
        break
    
cv.destroyAllWindows()

