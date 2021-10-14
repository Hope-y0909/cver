import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.defchararray import not_equal
img = cv.imread("./img/chessboard.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)

#哈里斯角检测
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0,0.04)
dst = cv.dilate(dst,None)

img[dst>0.01*dst.max()]=[0,0,255]

ret,dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
#寻找质心
ret,labels,stats,centroids = cv.connectedComponentsWithStats(dst)
#定义停止和完善拐角条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,100,0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
res = np.hstack((centroids,corners))
res = np.int64(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]]=[0,255,0]

cv.imshow("img",img)


#shi-tomasi拐点检测
img2 = cv.imread("./img/blox.jpg")
gray = cv.cvtColor(img2,cv.COLOR_BGRA2GRAY)
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int64(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img2,(x,y),2,255,3)
cv.imshow("img2",img2)
# plt.show()

# sift特征检测
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
sift1 = cv.xfeatures2d.SIFT_create()
kp1,des = sift1.detectAndCompute(gray,None)
img_sift = img2.copy()
img_sift0 = cv.drawKeypoints(gray,kp,img_sift)
img_sift1 = cv.drawKeypoints(gray,kp1,img_sift)
res = np.hstack((img_sift0,img_sift1))
cv.imshow("img3",res)

#surf特征检测
surf = cv.xfeatures2d.SURF_create(400)
kp_surf,res_surf = surf.detectAndCompute(gray,None)
img_surf = img2.copy()
img_surf = cv.drawKeypoints(gray,kp_surf,img_surf)
print(len(kp_surf))
cv.imshow("surf",img_surf)

#用于角点检测的fast算法
fast = cv.FastFeatureDetector_create()
kp_fast = fast.detect(gray,None)
img_fast = cv.drawKeypoints(gray,kp,None,color=(255,0,0))
cv.imshow("fast",img_fast)

fast.setNonmaxSuppression(0)
kp_fast1 = fast.detect(gray,None)

img_fast1 = cv.drawKeypoints(gray, kp, None, color=(255,0,0))
cv.imshow("fast1",img_fast1)
cv.waitKey(0)
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp_fast)))
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp_fast1)) )