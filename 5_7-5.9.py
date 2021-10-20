from PIL.Image import NONE
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import index_tricks

img = cv.imread("./img/test.jpg",0)
# #初始化Fast检测器
# star = cv.xfeatures2d.StarDetector_create()
# #初始化brief提取器
# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# #找star关键点
# kp = star.detect(img)
# kp1,des = brief.compute(img,kp)
# img1 = img.copy()
# img_kp = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# img_kp1 = cv.drawKeypoints(img1, kp, None, color=(255,0,0))
# plt.subplot(121),plt.imshow(img_kp)
# plt.subplot(122),plt.imshow(img_kp1)
# plt.show()
# print( brief.descriptorSize() )
# print( des.shape )

#初始化ORB
# orb = cv.ORB_create()
# kp_orb = orb.detect(img,None)
# kp_orb,des = orb.compute(img,kp_orb)
# img2 = cv.drawKeypoints(img,kp_orb,None,color=(0,255,0),flags=0)
# plt.imshow(img2),plt.show()

img1 = cv.imread("./img/star.png",cv.IMREAD_GRAYSCALE)
img2 = cv.imread("./img/multi_star.png",cv.IMREAD_GRAYSCALE)
#使用ORB描述符进行Brute-Force匹配
# orb = cv.ORB_create()
# kp1,des1 = orb.detectAndCompute(img1,None)
# kp2,des2 = orb.detectAndCompute(img2,None)
# bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
# matches = bf.match(des1,des2)
# matches = sorted(matches,key=lambda x:x.distance)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
#带有SIFT描述符和比例测试的Brute-Force匹配
# sift = cv.xfeatures2d.SIFT_create()
# kp1,des1 = sift.detectAndCompute(img1,None)
# kp2,des2 = sift.detectAndCompute(img2,None)
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# goods=[]
# for m,n in matches:
#     if m.distance < 0.75 * n.distance:
#         goods.append([m])
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,goods,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
#基于匹配器的FLANN
sift = cv.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask = matchesMask,flags=cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3),plt.show()