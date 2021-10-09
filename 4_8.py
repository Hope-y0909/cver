import cv2 as cv
import numpy as np
import sys
A = cv.imread("./img/apple.jpg")
B = cv.imread("./img/orange.jpg")
A = cv.resize(A,(256,256),interpolation=cv.INTER_CUBIC) 
B = cv.resize(B,(256,256),interpolation=cv.INTER_CUBIC)

# 生成A,B的高斯金字塔
G = A.copy()
gpA = [G]
for i in range(5):
    G = cv.pyrDown(G)
    gpA.append(G)
G = B.copy()
gpB = [G]
for i in range(5):
    G = cv.pyrDown(G)
    gpB.append(G)
# 生成A,B的拉普拉斯金字塔
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1], GE)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i-1], GE)
    lpB.append(L)
#在每个级别中添加左右两半图像
LS=[]
for la,lb in zip(lpA, lpB):
    rows,cols,dpt=la.shape
    print(la.shape)
    print(lb.shape)
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    # ls=np.hstack((la[:,0:cols/2],lb[:,cols/2:]))
    LS.append(ls)
#重建
ls_=LS[0]
for i in range(1,6):
    ls_=cv.pyrUp(ls_)
    ls_=cv.add(ls_,LS[i])


real= np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv.imshow("img",ls_)
cv.imshow("img1",real)
cv.waitKey(0)


