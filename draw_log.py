import cv2
import numpy as np
 
h,w = 400,420
img = np.zeros((h,w,3), np.uint8)
#最上面那个红环
cv2.ellipse(img, (w//2,h//4),(h//4,h//4),0,-240,60,(0,0,255),-1)  #按顺时针画椭圆
cv2.circle(img, (w//2,h//4),h//10,(0,0,0),-1)
#左边那个绿环
cv2.ellipse(img, (w//4,3*h//4),(h//4,h//4),0,0,300,(0,255,0),-1)
cv2.circle(img, (w//4,3*h//4),h//10,(0,0,0),-1)
#右边那个蓝环
cv2.ellipse(img, (3*w//4,3*h//4),(h//4,h//4),0,-60,240,(255,0,0),-1)
cv2.circle(img, (3*w//4,3*h//4),h//10,(0,0,0),-1)
 
cv2.startWindowThread()
cv2.imshow('logo',img)
 
k=cv2.waitKey(0)
if k == ord('s'):
     cv2.imwrite('logo.jpg',img)  #输入s，保存图片
cv2.destroyAllWindows()