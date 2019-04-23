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
import ballCollisionCuda
import Shapes
from ballCollisionCuda import generateRandomCircles
class CollisionUI(Frame):
    def __init__(self, master):
        self.width = 400
        self.height = 400
        self.numObstacles = 100
        self.maxObstacleSize = 30
        x_range = range(1, self.width)
        y_range = range(1, self.height)
        radius_range = range(1,self.maxObstacleSize)
        self.obstacles = generateRandomCircles(self.numObstacles,x_range, y_range, radius_range)
        self.robot = generateRandomCircles(1, x_range, y_range, radius_range)[0]
        super().__init__()
        self.initUI(master)

    def initUI(self, master):
        #'''
        self.score=0
        self.master.title("magic")
        frame=Frame(master)
        frame.pack()
        self.qbut = Button(
                frame, text="Calculate", fg="red", command=frame.quit
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

        self.canvas.configure(scrollregion=(0, 0, self.width, self.height))
        self.canvas.pack(side = BOTTOM, fill=BOTH,expand=1)
       
    def getObstacles(self):
        return self.obstacles

    def getRobot(self):
        return self.robot

    def say_hi(self):
        self.score=self.score+1
        print("score: " + str(self.score))

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

       
        self.canvas.create_rectangle(1, 1, 399, 399, outline="#000", width=1)
        self.canvas.pack(fill=Y,expand=0, side=BOTTOM)

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

def key(event):
    print("pressed", repr(event.char))

def callback(event):
    #frame.focus_set()
    print("clicked at", event.x, event.y)

def main():
    
    root = Tk()
    app = CollisionUI(root)
    frame = Frame(root, width=100, height=400)
    frame.bind("<Key>", key)
    frame.bind("<Button-1>", callback)
    frame.focus_set()
    frame.pack()
    #w = Label(root, text="Hello world!")
    
    #w.pack()
    root.geometry("400x400")
    root.mainloop()
    #print("final score: "+str(app.score))
    obstacles = app.getObstacles()
    robot = app.getRobot()
    print("Detecting collisions on chosen obstacles:")

    #DO POINT GENERATION/COLLISION DETECTION HERE
    #collisions[i] == true implies robot is in collision with obstacle i
    ballCollisionCuda.detectCollisionGPU(robot, obstacles)

    # cpuStart = time.time()
    # cpuCalc = x*y
    # print("cpu time taken = "+str(time.time()-cpuStart))
    # print (dest-cpuCalc)
    # print(len(dest))
    root.destroy()
main()