# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:10:11 2019

@author: Vosburgh
"""
import numpy
import os
import time

import random
import Shapes
import CircleCollision
import SphereCollision
from SphereCollision import generateRandomSpheres
import BoxCollision
from BoxCollision import generateRandomBoxes
from CircleCollision import generateRandomCircles
import RectangleCollision
from RectangleCollision import generateRandomRectangles

class CollisionTest():
    def __init__(self):
        self.width = 400
        self.height = 400
        self.numObstacles = 1024
        self.maxObstacleSize = 60#60 on rectangles, 40 on circles
        x_range = range(1, self.width)
        y_range = range(1, self.height)
        radius_range = range(5,self.maxObstacleSize)
        #circles
        #self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)
        #self.robot = generateRandomCircles(1, x_range, y_range, radius_range)[0]
        #rectangles
        self.new_obstacles()
        #self.obstacles = generateRandomSpheres(self.numObstacles)
        #self.robot = generateRandomSpheres(1) [0]

    def call_evaluation(self):
        return obstacleEval(self.obstacles, self.robot, self)

    def getObstacles(self):
        return self.obstacles

    def getRobot(self):
        return self.robot

    def new_obstacles(self, id=0):
        if id==0:
            x_range = range(1, self.width)
            y_range = range(1, self.height)
            radius_range = range(1, self.maxObstacleSize)
            self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)
            self.robot = generateRandomCircles(1) [0]

        if id==1:
            x_range = range(1, self.width)
            y_range = range(1, self.height)
            radius_range = range(1, self.maxObstacleSize)
            self.obstacles = generateRandomRectangles(self.numObstacles,x_range, y_range, radius_range)
            self.robot = generateRandomRectangles(1) [0]

        if id==2:
            self.obstacles = generateRandomSpheres(self.numObstacles)
            self.robot = generateRandomSpheres(1) [0]

        if id==3:
            self.obstacles = generateRandomBoxes(self.numObstacles)
            self.robot = generateRandomBoxes(1) [0]
       
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
        cpu_collisions_rectangles, cpu_time = RectangleCollision.detectCollisionCPU(robot, obstacles)
        gpu_collisions_rectangles, gpu_time = RectangleCollision.detectCollisionGPU(robot, obstacles)
        if((cpu_collisions_rectangles != gpu_collisions_rectangles).all()):
            print("difference of opinion, assume collision detection failed")

    if type(obstacles[0])==Shapes.Sphere:
        cpu_collisions_sphere, cpu_time = SphereCollision.detectCollisionCPU(robot, obstacles)
        gpu_collisions_sphere, gpu_time = SphereCollision.detectCollisionGPU(robot, obstacles)
        if((cpu_collisions_sphere != gpu_collisions_sphere).all()):
            print("difference of opinion, assume collision detection failed")

    if type(obstacles[0])==Shapes.Box:
        cpu_collisions_box, cpu_time = BoxCollision.detectCollisionCPU(robot, obstacles)
        gpu_collisions_box, gpu_time = BoxCollision.detectCollisionGPU(robot, obstacles)
        if((cpu_collisions_box != gpu_collisions_box).all()):
            print("difference of opinion, assume collision detection failed")
    # print (cpu_time, gpu_time)

    return cpu_time, gpu_time
def runTests(app, shapeId):
    testCount = 100
    cpuTimes = numpy.zeros(testCount)
    gpuTimes = numpy.zeros(testCount)
    i = 0
    while i < 100:
        cpuTimes[i], gpuTimes[i] = app.call_evaluation()
        app.new_obstacles(shapeId)
        i=i+1
    #print(cpuTimes)
    #print(gpuTimes)

    totalCpu = sum(cpuTimes)
    totalGpu = sum(gpuTimes)
    print("cpu time taken = "+str(totalCpu))
    print("gpu time taken = "+str(totalGpu))
def main():
    
    #w = Label(root, text="Hello world!")
    b = 0
    while b < 4:
        shape = 're'
        if b == 0:
            shape = "circles"
        if b == 1:
            shape = "rectangles"
        if b == 2:
            shape = "spheres"
        if b == 3:
            shape = "boxes"
        print("Test #"+str(b)+ ": " +shape)
        app = CollisionTest()
        runTests(app, b)
        b=b+1
    #w.pack() 
    #obstacles = app.getObstacles()
    #robot = app.getRobot()
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