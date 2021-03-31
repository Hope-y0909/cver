import cv2 as cv
import numpy as np
events = [i for i in dir(cv) if 'EVENT' in i]
print(events)
# mouse double_clclk
# def draw_circle(event,x,y,flags,param):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img,(x,y),100,(200,200,200),-1)

# img=np.zeros([600,600,3],np.uint8)
# cv.namedWindow('IMG')
# cv.setMouseCallback('IMG',draw_circle)
# while True:
#     cv.imshow('IMG',img)
#     if cv.waitKey(20) & 0xFF==27:
#         break
# cv.destroyAllWindows()

# mouse move
drawing = False  # if enter mouse ,true
mode = True  # if true ,draw rect,enter m change to curve
ix, iy = -1, -1


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (100, 200, 245), 1)
            else:
                cv.circle(img, (x, y), 10, (200, 200, 200), 1)
    elif event == cv.EVENT_RBUTTONUP:
        drawing = False
        # if mode == True:
        #     cv.rectangle(img, (ix, iy), (x, y), (100, 200, 245), 1)
        # else:
        #     cv.circle(img, (x, y), 10, (200, 200, 200), 1)


img = np.zeros([600, 600, 3], np.uint8)
cv.namedWindow('IMG')
cv.setMouseCallback('IMG', draw_circle)
while True:
    cv.imshow('IMG', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
