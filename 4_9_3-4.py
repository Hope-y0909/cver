import cv2 as cv
import numpy as np
from numpy.core.defchararray import equal
img = cv.imread("./img/countour.png")
gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
ret,th = cv.threshold(gray,175,255,0)
contours, hierarchy = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

#长宽比
x,y,w,h = cv.boundingRect(cnt)
ratio = float(w)/h
print("长宽比：{:.3f}".format(ratio))

#范围(轮廓区域与边界矩形区域的比值)
area = cv.contourArea(cnt)
rect_area = w*h
extend = float(area)/rect_area
print("范围:{:.3f}".format(extend))

#坚实度(等高线面积与其凸包面积之比)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
print("坚实度:{:.3f}".format(solidity))

#等效直径(面积与轮廓面积相同的圆的直径)
equi_diameter = np.sqrt(4*area/np.pi)
print("等效直径：{:.3f}".format(equi_diameter))

#取向(物体指向的角度) （x, y）代表椭圆中心点的位置（a, b）代表长短轴长度，应注意a、b为长短轴的直径，而非半径 angle 代表了中心旋转的角度
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)
print("中心点位置坐标：{:.2f} {:.2f} 长轴：{:.2f} 短轴：{:.2f} 物体指向角度：{:.2f}".format(x,y,MA,ma,angle))

#掩码和像素点
mask = np.zeros(gray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
#pixelpoint = np.transpose(np.nonzero(mask))
pixelpoint = cv.findNonZero(mask)
# cv.imshow("img",mask)
# cv.waitKey(0)

#最大值，最小值和它们的位置
min_val,max_val,min_loc,max_loc = cv.minMaxLoc(gray,mask=mask)
print("最大值：{:.2f} 最小值：{:.2f}".format(max_val,min_val))
print(max_loc,min_loc)

#平均颜色或平均强度
mean_val = cv.mean(img,mask=mask)
print(mean_val)

#极端点(对象的最顶部，最底部，最右侧和最左侧的点)
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
points_list = [leftmost,rightmost,topmost,bottommost]

for point in points_list:
	cv.circle(img, point, 3, (0,255,0), 3)
# cv.imshow("img2",img)


#凸性缺陷
hull = cv.convexHull(cnt,returnPoints=False)
defects = cv.convexityDefects(cnt,hull)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0] #起点、终点、最远点、到最远点的近似距离
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img,start,end,[0,255,0],2)
    cv.circle(img,far,5,[0,0,255],-1)
cv.imshow("img3",img)

#点多边形测试

dst = cv.pointPolygonTest(cnt,(50,50), False)
print(dst)

#形状匹配(比较两个形状或两个轮廓，并返回一个显示相似性的度量。结果越低，匹配越好)
img1 = cv.imread("./img/apple.jpg",0)
img2 = cv.imread("./img/orange.jpg",0)
# gray1 = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
# gray2 = cv.cvtColor(img2,cv.COLOR_RGB2GRAY)
ret,th1 = cv.threshold(img1,127,255,0)
ret,th2 = cv.threshold(img2,127,255,0)
contours1, hierarchy1 = cv.findContours(th1,2,1)
cnt1 = contours1[0]
contours2,hierarchy2 = cv.findContours(th2,2,1)
cnt2 = contours2[0]
ret = cv.matchShapes(cnt1,cnt2,1,0.0)
print(ret)

#练习1
img_star = cv.imread('img/star.jpg') 
img_gray = cv.cvtColor(img_star, cv.COLOR_BGR2GRAY)
kernel = cv.getStructuringElement(shape=1, ksize=(3, 3), anchor=(-1, -1))
im_open = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)
im_close = cv.morphologyEx(im_open, cv.MORPH_CLOSE, kernel)
ret, thresh = cv.threshold(im_close, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 2, 1)
cnt = contours[1]
rows, cols, channels = img_star.shape
for i in range(rows):
    for j in range(cols):
            dist = cv.pointPolygonTest(cnt, (i, j), True)
            absdist=int(abs(dist))
            if absdist>255:
                absdist=255
            if dist<0:
                cv.circle(img_star,(i,j), 1, (255-absdist,0,0), -1)
            elif dist>0:
                cv.circle(img_star,(i,j), 1, (0,0,255-absdist), -1)
            else :
                cv.circle(img_star,(i,j), 1, (0,255,0), -1)
cv.imshow('img_star', img_star)

#练习2
ABC = cv.imread("./img/ABC.jpg",0)
ABC1 = cv.cvtColor(ABC,cv.COLOR_GRAY2BGR)
_, thresh = cv.threshold(ABC, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# 搜索轮廓
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
hierarchy = np.squeeze(hierarchy)
# 载入标准模板图
img_a = cv.imread('./img/A.jpg', 0)
_, th = cv.threshold(img_a, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
contours1, hierarchy1 = cv.findContours(th, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# 字母A的轮廓
template_a = contours1[0]
# 记录最匹配的值的大小和位置
min_pos = -1
min_value = 2
for i in range(len(contours)):
	# 参数3：匹配方法；参数4：opencv预留参数
	value = cv.matchShapes(template_a,contours[i],1,0.0)
	if value < min_value:
		min_value = value
		min_pos = i
# 参数3为0表示绘制本条轮廓contours[min_pos]
cv.drawContours(ABC,[contours[min_pos]],0,[255,0,0],3)
cv.imshow("match",ABC)
cv.waitKey(0)

