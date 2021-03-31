import cv2 as cv
import numpy as np

#change background color
def nothing(x):
    pass

# img=np.zeros([500,500,3],np.uint8)
# cv.namedWindow('img')
# cv.createTrackbar('R','img',0,255,nothing)
# cv.createTrackbar('G','img',0,255,nothing)
# cv.createTrackbar('B','img',0,255,nothing)
# swtich ='0:ON\n1:/OFF'
# cv.createTrackbar(swtich,'img',0,1,nothing)
# while True:
#     cv.imshow('img',img)
#     k=cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     r=cv.getTrackbarPos('R','img')
#     g=cv.getTrackbarPos('G','img')
#     b=cv.getTrackbarPos('B','img')
#     s=cv.getTrackbarPos(swtich,'img')
#     if s==0:
#         img[:]=0
#     else:
#         img[:]=[b,g,r]
# cv.destroyAllWindows()

#change radius and color draw
events = [i for i in dir(cv) if 'EVENT' in i]
print(events)

def draw_circle(event,x,y,flags,param):
    r=cv.getTrackbarPos('R','img')
    g=cv.getTrackbarPos('G','img')
    b=cv.getTrackbarPos('B','img')
    color=(b,g,r)
    radi=cv.getTrackbarPos('Radius','img')

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),radi,color,-1)

img=np.zeros([600,600,3],np.uint8)

cv.namedWindow('img')
cv.createTrackbar('R','img',0,255,nothing)
cv.createTrackbar('G','img',0,255,nothing)
cv.createTrackbar('B','img',0,255,nothing)
cv.createTrackbar('Radius','img',0,100,nothing)
cv.setMouseCallback('img',draw_circle)
while True:
    cv.imshow('img',img)
    if cv.waitKey(20) & 0xFF==27:
        break
cv.destroyAllWindows()
