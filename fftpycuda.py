from __future__ import print_function
import numpy as np
import cv2
from reikna.fft import FFT
from reikna import cluda
import  reikna as rk
import pycuda.autoinit
from timeit import default_timer as timer
import skcuda.fft as cu_fft
import pycuda.gpuarray as gpuarray
import cudafft2d as cuda


N = 1080
M =1920
batch_size = 12
ran=np.exp(2*np.pi*1j*np.reshape(np.random.rand(1920*1080), (1080, 1920)))
#ran=1

x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
u,v=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
f=np.arange(400,520,10)

lmd=632e-6
p=8e-3

k=2*np.pi/lmd
colors=['images/c.bmp','images/a.bmp','images/b.bmp', 'images/a.bmp', 'images/a.bmp','images/b.bmp','images/a.bmp','images/b.bmp','images/a.bmp', 'images/a.bmp', 'images/a.bmp','images/c.bmp']
color=np.zeros((batch_size,1080,1920), dtype=np.complex128)
uplatform=np.zeros((batch_size,1080,1920), dtype=np.complex128)
#ran = np.exp(2*np.pi*1j*np.reshape(np.random.rand(1920*1080), (1080, 1920)))
E=0

start = timer()
for i in range(batch_size):
    ran = np.exp(2 * np.pi * 1j * np.reshape(np.random.rand(1920 * 1080), (1080, 1920)))
    color[i] = cv2.imread(colors[i], 0)
    px = lmd * f[i] / (p * 1920)
    py = lmd * f[i] / (p * 1080)
    color[i] = (255 - color[i]) / 255*ran
    color[i] = color[i] * np.exp(np.pi * 1j / (lmd * f[i]) * ((x * px) ** 2 + (y * py) ** 2))
    uplatform[i] = np.exp(k / f[i] * 1j / 2 * ((u * p) ** 2 + (v * p) ** 2))

    print (i)
start = timer()
color= cuda.fft_2d(color, 1080,1920,batch_size)
print (color)
# x_gpu = gpuarray.to_gpu(color)
# cu_fft.fft(x_gpu, x_gpu, plan)
#color2= cuda.fft_2d(color, 1080,1920,3)
for i in range(batch_size):
    color[i]=np.fft.fftshift(color[i])
    E += color[i]*uplatform[i]
print (E.shape)
#E=np.sum(E,axis=0)

r=np.sqrt((u * p) ** 2 + (v * p) ** 2+450**2)
#E =E/(np.exp(1j*k*r))
z = np.angle(E)/np.pi
z1=(z+1)/2*255
z1=z1.astype("uint8")
timeit=timer()-start
print("fft took %f seconds " % timeit)
print (z1)
cv2.imwrite('fftlll.bmp', z1)
cv2.imshow("tupian",z1)
cv2.waitKey()

cv2.destroyAllWindows()
# px = lmd * f[i] / (p * 1920)
# py = lmd * f[i] / (p * 1080)
# E0 = color*np.exp(np.pi * 1j /(lmd*f[i])* ((x * px) ** 2 + (y * py) ** 2))
# E1 = np.exp(k/f[i] * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))
# print f[i]
# Ef1=np.fft.fft2(E0)
# Ef=np.fft.fftshift(Ef1)
#
# E = Ef*E1
# r=np.sqrt((u * p) ** 2 + (v * p) ** 2+450**2)
# E =E/(np.exp(1j*k*r))
# z = np.angle(E)/np.pi
# z1=(z+1)/2*255
# z1=z1.astype("uint8")
# timeit=timer()-start
# print("fft took %f seconds " % timeit)
# print z1
# cv2.imwrite('ffthuijuabc.bmp', z1)
# cv2.imshow("tupian",z1)
# cv2.waitKey()
#
# cv2.destroyAllWindows()