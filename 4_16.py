

import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt


class App():
    Blue = [255,0,0]
    Green = [0,255,0]
    Red = [0,0,255]
    Black = [0,0,0]
    White = [255,255,255]
    Draw_Bg = {'color':Black,'val':0}
    Draw_Fg = {'color':White,'val':1}
    Draw_PR_Bg = {'color':Red,'val':2}
    Draw_PR_Fg = {'color':Green,'val':3}

    rect=(0,0,1,1)
    drawing = False
    rectangle = False
    rect_over = False
    rect_or_mask = 100
    value = Draw_Fg
    thickness = 3

    def onmouse(self,event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.rectangle = True
            self.ix,self.iy = x,y
        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img,(self.ix,self.iy),(x,y),self.Blue,2)
                self.rect = (min(self.ix,x),min(self.iy,y))
                self.rect_or_mask = 0
        elif event == cv.EVENT_LBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img,(self.ix,self.iy),(x,y),self.Blue,2)
            self.rect= (min(self.ix,x),min(self.iy,y),abs(self.ix -x),abs(self.iy-y))
            self.rect_or_mask  =0
    def Run(self):
        self.img = cv.imread("./img/messi.jpg")
        self.img2 = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2],np.uint8)
        self.output = np.zeros(self.img.shape,np.uint8)
        cv.namedWindow("input")
        cv.namedWindow("output")
        cv.setMouseCallback("input",self.onmouse)
        cv.moveWindow("input",self.img.shape[1]+10,90)

        while True:
            cv.imshow("input",self.img)
            cv.imshow("output",self.output)
            k = cv.waitKey(1)
            if k == 27:
                break
            elif k == ord('n'):
                try:
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgbmodel = np.zeros((1,65),np.float64)
                    if (self.rect_or_mask == 0):
                        cv.grabCut(self.img2,self.mask,self.rect,bgdmodel,fgbmodel,1,cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif (self.rect_or_mask == 1):
                        cv.grabCut(self.img2,self.mask,self.rect,bgdmodel,fgbmodel,1,cv.GC_INIT_WITH_RECT)
                except:
                    import traceback
                    traceback.print_exc()
            elif k == ord('r'):
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_over = False
                self.rect_or_mask = 100
                self.value = self.Draw_Bg
                self.img = self.img2.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8)
                self.output = np.zeros(self.img.shape, np.uint8)
            elif k == ord('0'):
                self.value = self.Draw_Bg
            elif k == ord('1'):
                self.value = self.Draw_Fg
            elif k == ord('2'):
                self.value = self.Draw_PR_Bg
            elif k == ord('3'):
                self.value = self.Draw_PR_Fg
            mask2 = np.where((self.mask==1)+(self.mask==3),255,0).astype("uint8")
            self.output = cv.bitwise_and(self.img2,self.img2,mask=mask2)



if __name__ == '__main__':
    App().Run()
    cv.destroyAllWindows()
