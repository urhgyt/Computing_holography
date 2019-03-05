import numpy as np
import cv2
from reikna.fft import FFT
from reikna import cluda
import  reikna as rk
from timeit import default_timer as timer
import cudafft2d as cuda


def reikna_fft(arr, P, Q):
    api = cluda.cuda_api()
    thr = api.Thread.create()
    x = thr.to_device(arr)
    X = thr.array((P,Q), dtype=np.complex128)
    fft = FFT(x)
    fftc = fft.compile(thr)
    fftc(X, x, 0)
    fft = X.get()
    thr.release()
    return fft


#color= cv2.imread('images/a.bmp',0)
#color=(255-color)/255
#color=color.astype("complex128")
ran=np.exp(2*np.pi*1j*np.reshape(np.random.rand(1920*1080), (1080, 1920)))
#ran=1

x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
u,v=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
f=[400,450,500]
lmd=632e-6
p=8e-3

k=2*np.pi/lmd
colors=['images/a.bmp','images/b.bmp','images/c.bmp']
E=0

start = timer()
for i in range(3):
    color = cv2.imread(colors[i], 0)
    color = (255 - color) / 255
    color = color * ran
    px = lmd * f[i] / (p * 1920)
    py = lmd * f[i] / (p * 1080)
    E0 = color*np.exp(np.pi * 1j /(lmd*f[i])* ((x * px) ** 2 + (y * py) ** 2))
    E1 = np.exp(k/f[i] * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))
    print f[i]
#    Ef1=np.fft.fft2(E0)
    print E0.dtype
    Ef1= cuda.fft_2d(E0, 1080,1920, 1)
#    Ef2=reikna_fft(E0,1080,1920)
    Ef=np.fft.fftshift(Ef1)
    E += Ef*E1
    print E

r=np.sqrt((u * p) ** 2 + (v * p) ** 2+450**2)
U2 = np.exp(1j*k*r)
E =E/(np.exp(1j*k*r))
z = np.angle(E)/np.pi
z1=(z+1)/2*255
z1=z1.astype("uint8")
timeit=timer()-start
print("fft took %f seconds " % timeit)
print z1
cv2.imwrite('fftpingxingabc.bmp', z1)
cv2.imshow("tupian",z1)
cv2.waitKey()

cv2.destroyAllWindows()