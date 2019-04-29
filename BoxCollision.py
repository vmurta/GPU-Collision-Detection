import numpy
import os
import pycuda.autoinit
import pycuda.driver as drv
import random
import time

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from Shapes import Box

if os.name=='nt':
    if (os.system("cl.exe")):
        os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

def generateRandomBoxes(numBoxes, x_range = range(1,5), \
        y_range = range(1,5), z_range=(1,40), \
        size_range = range(5,50)):

    random.seed()
    x1 = [random.choice(x_range) for i in range(numBoxes)]
    y1 = [random.choice(y_range) for i in range(numBoxes)]
    z1 = [random.choice(z_range) for i in range(numBoxes)]
    x2 = numpy.zeros(numBoxes)
    y2 = numpy.zeros(numBoxes)
    z2 = numpy.zeros(numBoxes)

    for i in range(numBoxes):
        x2[i] = x1[i] + random.choice(size_range)
        y2[i] = y1[i] + random.choice(size_range)
        z2[i] = z1[i] + random.choice(size_range) 

    boxes = [Box(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]) for i in range(numBoxes)]

    return boxes

def detectCollisionGPU(robot, obstacles):
    #print("compiling kernel")
    mod = SourceModule("""
    __global__ void check_collisions(
        float x1_robot, float y1_robot, float z1_robot,
        float x2_robot, float y2_robot, float z2_robot,
        float *x1_obs, float *y1_obs, float *z1_obs,
        float *x2_obs, float *y2_obs, float *z2_obs,
        bool *collisions)
    {
        int obstacleId = threadIdx.x;
        
        bool xcol = ((x1_obs[obstacleId] <= x1_robot && x1_robot <= x2_obs[obstacleId]) 
                || (x1_obs[obstacleId] <= x2_robot && x2_robot <= x2_obs[obstacleId])) 
                || (x1_robot <= x1_obs[obstacleId] && x2_robot >= x2_obs[obstacleId]);

        bool ycol = ((y1_obs[obstacleId] <= y1_robot && y1_robot <= y2_obs[obstacleId]) 
                || (y1_obs[obstacleId] <= y2_robot && y2_robot <= y2_obs[obstacleId])) 
                || (y1_robot <= y1_obs[obstacleId] && y2_robot >= y2_obs[obstacleId]);

        bool zcol = ((z1_obs[obstacleId] <= z1_robot && z1_robot <= z2_obs[obstacleId]) 
                || (z1_obs[obstacleId] <= z2_robot && z2_robot <= z2_obs[obstacleId])) 
                || (z1_robot <= z1_obs[obstacleId] && z2_robot >= z2_obs[obstacleId]);

        collisions[obstacleId] = (xcol && ycol && zcol);
    }
    """)
    #print("compiled kernel")

    
    check_collisions = mod.get_function("check_collisions")
    
    
    #constants that will be passed directly to kernel
    x1_robot = robot.x1
    y1_robot = robot.y1
    z1_robot = robot.z1
    x2_robot = robot.x2
    y2_robot = robot.y2
    z2_robot = robot.z2
    #allocating arrays on the gpu for obstacle coordinates and radii
    x1_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.x1 for rectangle in obstacles]))#nVidia only supports single precision)
    y1_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.y1 for rectangle in obstacles]))
    z1_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.z1 for rectangle in obstacles]))
    x2_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.x2 for rectangle in obstacles]))
    y2_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.y2 for rectangle in obstacles]))
    z2_obs_gpu = gpuarray.to_gpu(numpy.asarray([rectangle.z2 for rectangle in obstacles]))

    collisions = numpy.zeros(len(obstacles), dtype=bool)
    
    gpuStart = time.time()
    check_collisions(
            x1_robot, y1_robot, z1_robot, x2_robot, y2_robot, z2_robot,
            x1_obs_gpu, y1_obs_gpu, z1_obs_gpu, x2_obs_gpu, y2_obs_gpu, z2_obs_gpu,
            drv.InOut(collisions),
            block=(len(obstacles),1,1), grid=(1,1))
    #print(collisions)
    duration = time.time()-gpuStart
    #print("gpu time taken = "+str(time.time()-gpuStart))

    return collisions, duration

def detectCollisionCPU(robot, obstacles):
    cpuStart = time.time()
    collisions = [False]*len(obstacles)
    i = 0
    x1_robot = robot.x1
    y1_robot = robot.y1
    z1_robot = robot.z1
    x2_robot = robot.x2
    y2_robot = robot.y2
    z2_robot = robot.z2
    while i < len(obstacles):
        obs = obstacles[i]
        x1_obs = obs.x1
        y1_obs = obs.y1
        z1_obs = obs.z1
        x2_obs = obs.x2
        y2_obs = obs.y2
        z2_obs = obs.z2
        xcol = ((x1_obs <= x1_robot and x1_robot <= x2_obs) or (x1_obs <= x2_robot and x2_robot <= x2_obs)) or ( x1_robot <= x1_obs and x2_robot >= x2_obs)
        ycol = ((y1_obs <= y1_robot and y1_robot <= y2_obs) or (y1_obs <= y2_robot and y2_robot <= y2_obs)) or ( y1_robot <= y1_obs and y2_robot >= y2_obs)
        zcol = ((z1_obs <= z1_robot and z1_robot <= z2_obs) or (z1_obs <= z2_robot and z2_robot <= z2_obs)) or ( z1_robot <= z1_obs and z2_robot >= z2_obs)
        collisions[i]= xcol and ycol and zcol
        i=i+1
    duration = time.time()-cpuStart
    #print("cpu time taken = "+str(duration))
    #print(collisions)
    return collisions, duration
