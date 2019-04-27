# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:41:18 2019

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
from Shapes import Rectangle

#if on windows, try to find cl.exe
if os.name=='nt':
    if (os.system("cl.exe")):
        os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

def generateRandomRectangles(numRectangles, x_range = range(1,400), y_range = range(1,400), size_range = range(5,50)):
    random.seed()
    x1 = [random.choice(x_range) for i in range(numRectangles)]
    y1 = [random.choice(y_range) for i in range(numRectangles)]
    x2 = numpy.zeros(numRectangles)
    y2 = numpy.zeros(numRectangles)
    for i in range(numRectangles):
        x2[i] = x1[i] + random.choice(size_range)
        #print(x1[i] , ", " , x2[i])
    for i in range(numRectangles):
        y2[i] = y1[i] + random.choice(size_range) 
        #print(y1[i] , ", " , y2[i])
    rectangles = [Rectangle(x1[i],y1[i],x2[i],y2[i]) for i in range(numRectangles)]
    return rectangles   

def detectCollisionGPU(robot, obstacles):
    print("compiling kernel")
    mod = SourceModule("""
    __global__ void check_collisions(
        float x1_robot, float y1_robot, float x2_robot, float y2_robot,
        float *x1_obs, float *y1_obs, float *x2_obs, float *y2_obs,
        bool *collisions)
    {
        int obstacleId = threadIdx.x;
        
        bool xcol = ((x1_obs[obstacleId] <= x1_robot && x1_robot <= x2_obs[obstacleId]) 
                || (x1_obs[obstacleId] <= x2_robot && x2_robot <= x2_obs[obstacleId])) 
                || ( x1_robot <= x1_obs[obstacleId] && x2_robot >= x2_obs[obstacleId]);

        bool ycol = ((y1_obs[obstacleId] <= y1_robot && y1_robot <= y2_obs[obstacleId]) 
                || (y1_obs[obstacleId] <= y2_robot && y2_robot <= y2_obs[obstacleId])) 
                || ( y1_robot <= y1_obs[obstacleId] && y2_robot >= y2_obs[obstacleId]);

        collisions[obstacleId] = (xcol && ycol);
    }
    """)
    print("compiled kernel")

    
    check_collisions = mod.get_function("check_collisions")
    
    
    #constants that will be passed directly to kernel
    x1_robot = robot.x1
    y1_robot = robot.y1
    x2_robot = robot.x2
    y2_robot = robot.y2
    #allocating arrays on the gpu for obstacle coordinates and radii
    x1_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.x1 for rectangle in obstacles]))#nVidia only supports single precision)
    y1_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.y1 for rectangle in obstacles]))
    x2_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.x2 for rectangle in obstacles]))
    y2_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.y2 for rectangle in obstacles]))


    collisions = numpy.zeros(len(obstacles), dtype=bool)
    gpuStart = time.time()
    check_collisions(
            x1_robot, y1_robot, x2_robot, y2_robot,
            x1_obs_gpu, y1_obs_gpu, x2_obs_gpu, y2_obs_gpu,
            drv.InOut(collisions),
            block=(len(obstacles),1,1), grid=(1,1))
    

    print("gpu time taken = "+str(time.time()-gpuStart))
    #print(collisions)
    return collisions

def detectCollisionCPU(robot, obstacles):
    cpuStart = time.time()
    collisions = [False]*len(obstacles)
    i = 0
    x1_robot = numpy.float32(robot.x1)
    y1_robot = numpy.float32(robot.y1)
    x2_robot = numpy.float32(robot.x2)
    y2_robot = numpy.float32(robot.y2)
    while i < len(obstacles):
        obs = obstacles[i]
        x1_obs = obs.x1
        y1_obs = obs.y1
        x2_obs = obs.x2
        y2_obs = obs.y2
        xcol = ((x1_obs <= x1_robot and x1_robot <= x2_obs) or (x1_obs <= x2_robot and x2_robot <= x2_obs)) or ( x1_robot <= x1_obs and x2_robot >= x2_obs)
        ycol = ((y1_obs <= y1_robot and y1_robot <= y2_obs) or (y1_obs <= y2_robot and y2_robot <= y2_obs)) or ( y1_robot <= y1_obs and y2_robot >= y2_obs)
        collisions[i]= xcol and ycol
        i=i+1
    print("cpu time taken = "+str(time.time()-cpuStart))
    #print(collisions)
    return collisions