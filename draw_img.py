import cv2 as cv
import numpy as np

img=np.zeros((512,512,3),np.uint8)

#draw line ,left_top to right_bottom
cv.line(img,(0,0),(511,511),(255,0,0))
#draw rect 
cv.rectangle(img,(380,0),(510,128),(0,255,0),3)
#draw circle 
cv.circle(img,(400,200),25,(0,0,255),-1)
#draw ellipse
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
#draw multi_bound 
pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32).reshape((-1,1,2))
#pts=np.reshape((1,2))
cv.polylines(img,pts,True,(0,255,255))

front=cv.FONT_HERSHEY_COMPLEX
cv.putText(img,'opencv',(10,500),front,4,(255,255,255),2,cv.LINE_AA)
cv.imshow("img",img)
if (cv.waitKey(0)==ord('q')):
    cv.destroyAllWindows()
    exit()
