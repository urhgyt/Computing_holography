import numpy as np
import cv2


x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
f=500
p=8e-3
lmd = 632e-6
k = 2*np.pi/lmd

E=np.exp(1j*k*np.sqrt(((x*p)**2+(y*p)**2+f**2)))

z=np.angle(E)
z=(z+np.pi)/(2*np.pi)*255
z2=z.astype("uint8")
print z
cv2.imwrite('toujing.bmp', z2)
cv2.imshow("tupian",z2)
cv2.waitKey()

cv2.destroyAllWindows()