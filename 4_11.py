import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.scimath import log

img = cv.imread("./img/messi.jpg",0)
#numpy_傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title("input"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray'),plt.title("magnitude spectrum"),plt.xticks([]),plt.yticks([])
plt.show()

rows,cols = img.shape
crows,ccols = rows//2,cols//2
fshift[crows-30:crows+31,ccols-30:ccols+31] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)
plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title("input"),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(img_back,cmap='gray'),plt.title("image after HPF"),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(img_back),plt.title("res in JET"),plt.xticks([]),plt.yticks([])
plt.show()

#opencv_傅里叶变换
dft = cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title("input"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray'),plt.title("magnitude spectrum"),plt.xticks([]),plt.yticks([])
plt.show()

rows,cols = img.shape
crows,ccols = rows//2,cols//2
mask = np.zeros((rows,cols,2),np.uint8)
mask[crows-30:crows+31,ccols-30:ccols+31]=1
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title("input"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back,cmap='gray'),plt.title("magnitude spectrum"),plt.xticks([]),plt.yticks([])
plt.show()

#滤波器比较
mean_filter = np.ones((3,3))
# 创建高斯滤波器
x = cv.getGaussianKernel(5,10)
gaussian = x*x.T
# 不同的边缘检测滤波器
# x方向上的scharr
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# x方向上的sobel
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# y方向上的sobel
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# 拉普拉斯变换
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])
filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()