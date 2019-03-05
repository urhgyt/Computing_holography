from __future__ import print_function
from timeit import default_timer as timer
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import skcuda.fft as cu_fft


N = 1080
M =1920
batch_size = 16

print('Testing in-place fft..')
x=np.random.rand(N*M*batch_size).astype(np.complex64)
y=np.random.rand(N*M*batch_size).astype(np.complex64)
x = np.resize(x,(batch_size, N, M))
# for i in range(batch_size):
#     x[i, :, :] = np.asarray(np.random.rand(N, M), np.complex64)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
start = timer()

for i in range(batch_size):
    x[i]= np.fft.fft2(x[i, :, :])
#    x[i]=np.fft.ifft2(x[i])

timeit1=timer()-start

for i in range(batch_size):
    x[i]=np.fft.ifft2(x[i])
plan = cu_fft.Plan((N, M), np.complex64, np.complex64, batch_size)

start = timer()

cu_fft.fft(x_gpu, x_gpu, plan)
cu_fft.fft(y_gpu, y_gpu, plan)

timeit2=timer()-start
x_gpu1=x_gpu.get()

for i in range(batch_size):
    x_gpu1[i]=np.fft.ifft2(x_gpu1[i])
#cu_fft.ifft(x_gpu, x_gpu, plan, True)

# print('/n Success status1: ', x[1])
# print('Success status2: ', x_gpu1[1])
#print('Success status: ', np.allclose(x[0], x_gpu[0].get(), atol=1e-6))
print('Success status: ', np.allclose(x[0], x_gpu1[0], atol=1e-6))
print('np take time:',timeit1, 'pycuda take time:', timeit2)
print(x.shape, x_gpu1.shape)
print('boostrate:',timeit1/timeit2)