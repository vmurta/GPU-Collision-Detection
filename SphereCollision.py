import numpy
import os
import pycuda.autoinit
import pycuda.driver as drv
import random
import time

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from Shapes import Sphere

#TODO
def generateRandomSpheres(numSpheres=100, x_range = range(1,5), \
        y_range = range(1,5), z_range = range(1,5), \
        radius_range = [1]):
    random.seed()
    x = [random.choice(x_range) for i in range(numSpheres)]
    y = [random.choice(y_range) for i in range(numSpheres)]
    z = [random.choice(y_range) for i in range(numSpheres)]
    rad = [random.choice(radius_range) for i in range(numSpheres)]
    spheres = [Sphere(x[i],y[i],z[i], rad[i]) for i in range(numSpheres)]
    return spheres

#TODO test to see if i can get this faster by squaring the x, y, and z distances
# and comparing against radius squared
def detectCollisionGPU(robot, obstacles):

    mod = SourceModule("""
    __global__ void check_collisions(
        float x_robot, float y_robot, float z_robot, float r_robot,
        float *x_obs, float *y_obs, float *z_obs, float *r_obs, 
        bool *collisions)
    {
        int obstacleId = threadIdx.x;
        float distance = norm3df(x_robot - x_obs[obstacleId],
                y_robot - y_obs[obstacleId], z_robot - z_obs[obstacleId]);
        collisions[obstacleId] = (distance <= r_robot + r_obs[obstacleId] );
    }
    """)

    
    check_collisions = mod.get_function("check_collisions")
    
    
    #constants that will be passed directly to kernel
    x_robot = robot.x
    y_robot = robot.y
    z_robot = robot.z
    r_robot = robot.rad

    #allocating arrays on the gpu for obstacle coordinates and radii
    x_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.x for circle in obstacles]))#nVidia only supports single precision)
    y_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.y for circle in obstacles]))
    z_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.z for circle in obstacles]))
    r_obs_gpu = gpuarray.to_gpu(numpy.asarray([circle.rad for circle in obstacles]))

    collisions = numpy.zeros(len(obstacles), dtype=bool)
    
    gpuStart = time.time()
    check_collisions(
            x_robot, y_robot, z_robot, r_robot,
            x_obs_gpu, y_obs_gpu, z_obs_gpu, r_obs_gpu,
            drv.InOut(collisions),
            block=(len(obstacles),1,1), grid=(1,1))
    duration = time.time()-gpuStart
    #print("gpu time taken = "+str(duration))

    return collisions, duration


#not tested, based on success of the circle one should be good to go
def detectCollisionCPU(robot, obstacles):
    cpuStart = time.time()
    collisions = [False]*len(obstacles)
    i = 0
    x_robot = robot.x
    y_robot = robot.y
    z_robot = robot.z
    r_robot = robot.rad
    while i < len(obstacles):
        obs = obstacles[i]
        distance = numpy.sqrt((x_robot-obs.x)**2 + (y_robot-obs.y)**2 + (z_robot-obs.z)**2)
        collisions[i]= (distance <= r_robot + obs.rad)
        i=i+1
    duration = time.time()-cpuStart
    #print(duration)
    #print("cpu time taken = "+str(time.time()-cpuStart))
    #print(collisions)
    return collisions, duration