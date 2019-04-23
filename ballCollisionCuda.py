# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:41:18 2019

@author: Vosburgh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:03:19 2019

@author: Vosburgh
"""

import numpy
import os
import pycuda.autoinit
import pycuda.driver as drv
import random
import time

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from Shapes import Circle
from tkinter import Frame, Tk

#if on windows, try to find cl.exe
if os.name=='nt':
    if (os.system("cl.exe")):
        os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

def generateRandomCircles(numCircles, x_range, y_range, radius_range):
    random.seed()
    x = [random.choice(x_range) for i in range(numCircles)]
    y = [random.choice(y_range) for i in range(numCircles)]
    rad = [random.choice(radius_range) for i in range(numCircles)]
    circles = [Circle(x[i],y[i],rad[i]) for i in range(numCircles)]
    return circles

def detectCollisionGPU(robot, obstacles):
    print("compiling kernel")
    mod = SourceModule("""
    __global__ void check_collisions(
        float x_robot, float y_robot, float r_robot,
        float *x_obs, float *y_obs, float *r_obs, 
        bool *collisions, int *indexes)
    {
        int obstacleId = threadIdx.x;
        float distance = hypotf(x_robot - x_obs[obstacleId], y_robot - y_obs[obstacleId]);
        collisions[obstacleId] = (distance <= r_robot + r_obs[obstacleId] );
    }
    """)
    print("compiled kernel")

    
    check_collisions = mod.get_function("check_collisions")
    
    
    x_robot = numpy.float32(robot.x)
    y_robot = numpy.float32(robot.y)#.astype(numpy.float32)
    r_robot = numpy.float32(robot.rad)#.astype(numpy.float32)

    x_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.x for circle in obstacles]).astype(numpy.float32))#nVidia only supports single precision)
    y_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.y for circle in obstacles]).astype(numpy.float32))
    r_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.rad for circle in obstacles]).astype(numpy.float32))

    collisions = numpy.zeros(len(obstacles), dtype=bool)

    print(collisions)
    gpuStart = time.time()
    check_collisions(
            x_robot, y_robot, r_robot,
            x_obs_gpu, y_obs_gpu, r_obs_gpu,
            drv.InOut(collisions),
            block=(len(obstacles),1,1), grid=(1,1))
    print(collisions)
    
    print("gpu time taken = "+str(time.time()-gpuStart))

    return collisions

def detectCollisionCPU(robot, obstacles):
    pass

