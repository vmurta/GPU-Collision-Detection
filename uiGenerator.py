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
from tkinter import Frame, Canvas, Tk, BOTH, Button, RIGHT, LEFT, Y, X, BOTTOM
import random
import CircleCollision
import SphereCollision
import Shapes
from CircleCollision import generateRandomCircles
import RectangleCollision
from RectangleCollision import generateRandomRectangles

class CollisionUI(Frame):
    def __init__(self, master):
        self.width = 400
        self.height = 400
        self.numObstacles = 100
        self.maxObstacleSize = 40#60 on rectangles, 40 on circles
        x_range = range(1, self.width)
        y_range = range(1, self.height)
        radius_range = range(5,self.maxObstacleSize)
        #circles
        self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)
        self.robot = generateRandomCircles(1, x_range, y_range, radius_range)[0]
        #rectangles
        #self.obstacles = generateRandomRectangles(self.numObstacles, x_range, y_range, radius_range)
        #self.robot = generateRandomRectangles(1, x_range, y_range, radius_range)[0]
        super().__init__()
        self.initUI(master)

    def initUI(self, master):
        #'''
        self.score=0
        self.master.title("magic")
        frame=Frame(master)
        frame.pack()
        self.qbut = Button(
                frame, text="Calculate", fg="red", command=self.call_evaluation
                )
        self.hibut = Button(
                frame, text="Refresh", command=self.new_obstacles
                )
        self.qbut.pack(side=LEFT)
        self.hibut.pack(side=RIGHT)
        #'''
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self, width = self.width, height = self.height, scrollregion = (0,0,self.width,self.height))
        self.canvas.create_rectangle(2, 2, self.width, self.height, outline="#000", width=2)
        
        
        if type(self.obstacles[0])==Shapes.Circle:
            # add obstacles to canvas
            for circle in self.obstacles:
                self.canvas.create_oval(circle.x-circle.rad,circle.y-circle.rad,circle.x+circle.rad,circle.y+circle.rad,outline="#fb0", fill="#fb0")
            #add robot to canvas
            self.canvas.create_oval(self.robot.x-self.robot.rad,self.robot.y-self.robot.rad,self.robot.x+self.robot.rad,self.robot.y+self.robot.rad,outline="#0bf", fill="#0bf")
        if type(self.obstacles[0])==Shapes.Rectangle:
            for rect in self.obstacles:
                self.canvas.create_rectangle(rect.x1,rect.y1,rect.x2,rect.y2, outline="#fb0", fill="#fb0")
            #add robot to canvas
            self.canvas.create_rectangle(self.robot.x1,self.robot.y1,self.robot.x2,self.robot.y2,outline="#0bf", fill="#0bf")
        self.canvas.configure(scrollregion=(0, 0, self.width, self.height))
        self.canvas.pack(side = BOTTOM, fill=BOTH,expand=1)
    def call_evaluation(self):
        obstacleEval(self.obstacles, self.robot, self)
    def getObstacles(self):
        return self.obstacles

    def getRobot(self):
        return self.robot

    def new_obstacles(self):
        self.canvas.delete("all")
        if type(self.obstacles[0])==Shapes.Circle:
            x_range = range(1, self.width)
            y_range = range(1, self.height)
            radius_range = range(1, self.maxObstacleSize)
            self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)
            # add obstacles to canvas
            for circle in self.obstacles:
                self.canvas.create_oval(circle.x-circle.rad,circle.y-circle.rad,circle.x+circle.rad,circle.y+circle.rad,outline="#fb0", fill="#fb0")
            #add robot to canvas
            self.canvas.create_oval(self.robot.x-self.robot.rad,self.robot.y-self.robot.rad,self.robot.x+self.robot.rad,self.robot.y+self.robot.rad,outline="#0bf", fill="#0bf")
        if type(self.obstacles[0])==Shapes.Rectangle:
            x_range = range(1, self.width)
            y_range = range(1, self.height)
            radius_range = range(1, self.maxObstacleSize)
            self.obstacles = generateRandomRectangles(self.numObstacles,x_range, y_range, radius_range)
            # add obstacles to canvas
            for rect in self.obstacles:
                self.canvas.create_rectangle(rect.x1,rect.y1,rect.x2,rect.y2, outline="#fb0", fill="#fb0")
            #add robot to canvas
            self.canvas.create_rectangle(self.robot.x1,self.robot.y1,self.robot.x2,self.robot.y2,outline="#0bf", fill="#0bf")
       
        self.canvas.create_rectangle(1, 1, 399, 399, outline="#000", width=1)
        self.canvas.pack(fill=Y,expand=0, side=BOTTOM)
    def draw_collisions(self, colls):
        print("Drawing collisions...")
        if type(self.obstacles[0])==Shapes.Circle:
            i=0
            print(len(self.obstacles))
            print(len(colls))
            while i < len(colls):
                status = colls[i]
                if status:
                    circle = self.obstacles[i]
                    self.canvas.create_oval(circle.x-circle.rad,circle.y-circle.rad,circle.x+circle.rad,circle.y+circle.rad,outline="red", fill="#fb0")
                i=i+1
            #re-add robot to canvas
            self.canvas.create_oval(self.robot.x-self.robot.rad,self.robot.y-self.robot.rad,self.robot.x+self.robot.rad,self.robot.y+self.robot.rad,outline="#0bf", fill="#0bf")
        if type(self.obstacles[0])==Shapes.Rectangle:
            i=0
            print(len(self.obstacles))
            print(len(colls))
            while i < len(colls):
                status = colls[i]
                if status:
                    rect = self.obstacles[i]
                    self.canvas.create_rectangle(rect.x1,rect.y1,rect.x2,rect.y2, outline="red", fill="#fb0")
                i=i+1
            #re-add robot to canvas
            self.canvas.create_rectangle(self.robot.x1,self.robot.y1,self.robot.x2,self.robot.y2,outline="#0bf", fill="#0bf")
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
    print("Detecting collisions on chosen obstacles:")
    #gpu_collisions[i] == true implies robot is in collision with obstacle i
    if type(obstacles[0])==Shapes.Circle:
        gpu_collisions_circles = CircleCollision.detectCollisionGPU(robot, obstacles)
        app.draw_collisions(gpu_collisions_circles)
    if type(obstacles[0])==Shapes.Rectangle:
        gpu_collisions_rectangles = RectangleCollision.detectCollisionGPU(robot, obstacles)
        app.draw_collisions(gpu_collisions_rectangles)
def main():
    
    root = Tk()
    app = CollisionUI(root)
    frame = Frame(root, width=100, height=400)
    frame.focus_set()
    frame.pack()
    #w = Label(root, text="Hello world!")
    
    #w.pack() 
    root.geometry("400x400")
    root.mainloop()
    obstacles = app.getObstacles()
    robot = app.getRobot()
    print("Detecting collisions on chosen obstacles:")
    
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
    root.destroy()
main()