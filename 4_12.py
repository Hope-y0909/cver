import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.defchararray import rsplit

#模板匹配

img = cv.imread("./img/messi.jpg", 0)
img2 = img.copy()
tem = cv.imread("./img/messi_face.jpg", 0)
w, h = tem.shape[::-1]
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)

    res = cv.matchTemplate(img, tem, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0]+w, top_left[1]+h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray'), plt.title("match res"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray'), plt.title("detected point"), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

#多对象模板匹配
img_rgb = cv.imread("./img/multi_star.png")
img_copy = img_rgb.copy()
img_gray = cv.cvtColor(img_rgb,cv.COLOR_BGRA2GRAY)
img_tem = cv.imread("./img/star.png",0)
w,h = img_tem.shape[:]
res = cv.matchTemplate(img_gray,img_tem,cv.TM_CCORR_NORMED)
print(res)
th = 0.96
loc = np.where(res >= th)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
res =np.hstack((img_copy,img_rgb))
cv.imshow("res",res)
cv.waitKey(0)
# plt.subplot(121), plt.imshow(img_gray), plt.title("input"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_rgb), plt.title("res"), plt.xticks([]), plt.yticks([])
# plt.show()