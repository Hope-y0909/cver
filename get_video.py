import cv2 as cv
import numpy as np

# read file
cap=cv.VideoCapture("img/test.mp4")
while(cap.isOpened()):
    ret,frame=cap.read()
    if not ret:
        printf("can not read frame.")
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow("frame",gray)
    if(cv.waitKey(1)==ord('q')): #cv.waitKey(num) num bigger,video slower
        break
cap.release()
cv.destroyAllWindows()

# #read camera
# cap=cap.VideoCapture(0)
# if not cap.isOpened():
#     printf("cant open camera")
#     exit()
# while True:
#     ret,frame=cap.read()
#     if not ret:
#         print("can not read frame")
#         break
#     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     cv.imshow("camera",frame)
#     if(cv.waitKey(1)==ord('q')):
#         break
# cap.release()
# cap.destroyAllWindows()

#save video
# cap=cv.VideoCapture("test.mp4")

# fourcc=cv.VideoWriter_fourcc(*'XVID')
# out=cv.VideoWriter("out.avi",fourcc,20.0,(640,480))

# while(cap.isOpened()):
#     ret,frame=cap.read()
#     if not ret:
#         printf("can not read frame.")
#         break
#     frame=cv.flip(frame,0)
#     out.write(frame)
#     cv.imshow("frame",frame)
#     if(cv.waitKey(1)==ord('q')):
#         break
# cap.release()
# out.release()
# cv.destroyAllWindows()