from numba import jit, njit, prange
import numpy as np
import cv2

color= cv2.imread('color.jpg',0)
color=color.astype(np.float32)
depth= cv2.imread('depth.jpg',0)
depth=depth.astype(np.float32)
#print color.dtype


x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
#x=x.astype(np.float32)
#y=y.astype(np.float32)
print x.shape
print x.dtype
# f=280
#
# p=8e-3
# ps=0.15
# k=2*np.pi/lmd
# Acomlx = 0
# f1=450


@jit(parallel=True)
def dianyun(x,y,color,depth):
    Acomlx = 0
    lmd=632e-6
    k = 2 * np.pi / lmd
    for u in prange(0,99, 5):
        for v in prange(0,99,5):
            A=color[v,u]
            d=630-depth[v,u]
            v1=50-v
            u1=u-50
            Acomlx += A * np.exp(-k * 1j * (np.sqrt((x * 8e-3-u1*0.15) ** 2 + (-y * 8e-3-v1*0.15) ** 2 + d ** 2)-np.sqrt(450**2+(x*8e-3)**2+(y*8e-3)**2)))
#            Acomlx = Acomlx + A
            print v
    return Acomlx
    pass

#Acomlx=Acomlx/np.exp(-1j*np.sqrt(f1**2+(x*p)**2+(y*p)**2))
Acomlx = dianyun(x, y, color, depth)
z=-np.angle(Acomlx)/np.pi
z1=(z+1)/2*255
z2=z1.astype("uint8")
print z
cv2.imwrite('tuzihuiju.bmp', z2)
cv2.imshow("tupian",z2)
cv2.waitKey()

cv2.destroyAllWindows()