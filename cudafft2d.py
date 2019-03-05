from __future__ import print_function
from timeit import default_timer as timer
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import skcuda.fft as cu_fft


def fft_2d(x, N, M, batch_size):
    # print('Testing in-place fft..')

    # for i in range(batch_size):
    #     x[i, :, :] = np.asarray(np.random.rand(N, M), np.complex64)
    x_gpu = gpuarray.to_gpu(x)
    # start = timer()



    plan = cu_fft.Plan((N, M), np.complex128, np.complex128, batch_size)


    cu_fft.fft(x_gpu, x_gpu, plan)

    # timeit2=timer()-start
    x_gpu1=x_gpu.get()
    # print ('take time:',timeit2)
    return x_gpu1
