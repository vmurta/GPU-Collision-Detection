# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:10:11 2019

@author: Vosburgh
"""
import numpy
import os
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import time

from pycuda.compiler import SourceModule
import random
import Shapes
import CircleCollision
import SphereCollision
import BoxCollision
from CircleCollision import generateRandomCircles
import RectangleCollision
from RectangleCollision import generateRandomRectangles

class CollisionTest():
    def __init__(self):
        self.width = 400
        self.height = 400
        self.numObstacles = 1000
        self.maxObstacleSize = 60#60 on rectangles, 40 on circles
        x_range = range(1, self.width)
        y_range = range(1, self.height)
        radius_range = range(5,self.maxObstacleSize)
        #circles
        #self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)
        #self.robot = generateRandomCircles(1, x_range, y_range, radius_range)[0]
        #rectangles
        self.obstacles = generateRandomRectangles(self.numObstacles, x_range, y_range, radius_range)
        self.robot = generateRandomRectangles(1, x_range, y_range, radius_range)[0]

    def call_evaluation(self):
        return obstacleEval(self.obstacles, self.robot, self)

    def getObstacles(self):
        return self.obstacles

    def getRobot(self):
        return self.robot

    def new_obstacles(self):
        if type(self.obstacles[0])==Shapes.Circle:
            x_range = range(1, self.width)
            y_range = range(1, self.height)
            radius_range = range(1, self.maxObstacleSize)
            self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)

        if type(self.obstacles[0])==Shapes.Rectangle:
            x_range = range(1, self.width)
            y_range = range(1, self.height)
            radius_range = range(1, self.maxObstacleSize)
            self.obstacles = generateRandomRectangles(self.numObstacles,x_range, y_range, radius_range)
       
    # @staticmethod
    # def generateNumbers():
    #     n = 100
    #     xmax = 400
    #     ymax = xmax
    #     rmax = 30
    #     xs = random.sample(range(1, xmax), n)#0 at Left, moves right
    #     ys = random.sample(range(1, ymax), n)#0 at top, moves down
    #     rs = [random.randint(1,rmax) for i in range(n)]
    #     print("xs:", xs)
    #     print("ys", ys)
    #     print("rs:", rs)
    #     #point = random.sample(range(1,xmax), 2)
    #     return xs, ys, rs

def obstacleEval(obstacles, robot, app):
    #print("Detecting collisions on chosen obstacles:")
    #gpu_collisions[i] == true implies robot is in collision with obstacle i
    cpu_time = 0
    gpu_time = 0
    if type(obstacles[0])==Shapes.Circle:
        cpu_collisions_circles, cpu_time = CircleCollision.detectCollisionCPU(robot, obstacles)
        gpu_collisions_circles, gpu_time = CircleCollision.detectCollisionGPU(robot, obstacles)
        if((cpu_collisions_circles != gpu_collisions_circles).all()):
            print("difference of opinion, assume collision detection failed")
    if type(obstacles[0])==Shapes.Rectangle:
        gpu_collisions_rectangles, cpu_time = RectangleCollision.detectCollisionGPU(robot, obstacles)
        cpu_collisions_rectangles, gpu_time = RectangleCollision.detectCollisionCPU(robot, obstacles)
        if((cpu_collisions_rectangles != gpu_collisions_rectangles).all()):
            print("difference of opinion, assume collision detection failed")
    return cpu_time, gpu_time
def main():
    
    #w = Label(root, text="Hello world!")
    app = CollisionTest()
    #w.pack() 
    obstacles = app.getObstacles()
    robot = app.getRobot()
    print("Detecting collisions on chosen obstacles:")
    testCount = 100
    cpuTimes = numpy.zeros(testCount)
    gpuTimes = numpy.zeros(testCount)
    i = 0
    while i < 100:
        cpuTimes[i], gpuTimes[i] = app.call_evaluation()
        app.new_obstacles()
        i=i+1
    print(cpuTimes)
    print(gpuTimes)
    totalCpu = sum(cpuTimes)
    totalGpu = sum(gpuTimes)
    print("cpu time taken = "+str(totalCpu))
    print("gpu time taken = "+str(totalGpu))
    #gpu_collisions[i] == true implies robot is in collision with obstacle i

    #gpu_collisions_circles = CircleCollision.detectCollisionGPU(robot, obstacles)

    #sphere_obstacles=SphereCollision.generateRandomSpheres()
    #sphere_robot=SphereCollision.generateRandomSpheres(numSpheres=1)[0]
    #gpu_collisions_spheres = SphereCollision.detectCollisionGPU(
    #        sphere_robot, sphere_obstacles)
    



    # cpuStart = time.time()
    # cpuCalc = x*y
    # print("cpu time taken = "+str(time.time()-cpuStart))
    # print (dest-cpuCalc)
    # print(len(dest))
main()