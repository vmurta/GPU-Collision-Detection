# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:03:19 2019

@author: Vosburgh
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time

from tkinter import Frame, Canvas, Tk, BOTH, Button, RIGHT, LEFT, Y, X, BOTTOM
import random

from pycuda.compiler import SourceModule

import os
if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
#if (os.system("cl.exe")):
    #raise RuntimeError("cl.exe still not found, path probably incorrect")

################# FROM uiGenerator

class App(Frame):
    def __init__(self, master):
        super().__init__()
        self.initUI(master)
    def getObstacles(self):
        return self.xs, self.ys, self.rs
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
                frame, text="Refresh", command=self.new_numbers
                )
        self.qbut.pack(side=LEFT)
        self.hibut.pack(side=RIGHT)
        #'''
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self, width = 400, height = 400, scrollregion = (0,0,400,400))
        self.canvas.create_rectangle(2, 2, 400, 400, outline="#000", width=2)

        self.xs, self.ys, self.rs = generateNumbers()
        
        i =0
        while i < len(self.xs):
            self.canvas.create_oval(self.xs[i]-self.rs[i],self.ys[i]-self.rs[i],self.xs[i]+self.rs[i],self.ys[i]+self.rs[i],outline="#fb0", fill="#fb0")
            i=i+1
        
        self.canvas.configure(scrollregion=(0, 0, 400, 400))
        self.canvas.pack(side = BOTTOM, fill=BOTH,expand=1)
       
    def say_hi(self):
        self.score=self.score+1
        print("score: " + str(self.score))
    def new_numbers(self):
        self.xs, self.ys, self.rs = generateNumbers()
        self.canvas.delete("all")
        
        i =0
        while i < len(self.xs):
            self.canvas.create_oval(self.xs[i]-self.rs[i],self.ys[i]-self.rs[i],self.xs[i]+self.rs[i],self.ys[i]+self.rs[i],outline="#fb0", fill="#fb0")
            i=i+1
        
        self.canvas.create_rectangle(1, 1, 399, 399, outline="#000", width=1)
        self.canvas.pack(fill=Y,expand=0, side=BOTTOM)

def generateNumbers():
    n = 10000000 #100 recommended
    xmax = 40000000 #400 recommended
    ymax = xmax
    rmax = 30
    xs = random.sample(range(1, xmax), n)#0 at Left, moves right
    ys = random.sample(range(1, ymax), n)#0 at top, moves down
    rs = [random.randint(1,rmax) for i in range(n)]
    #print("xs:", xs)
    #print("ys", ys)
    #print("rs:", rs)
    #point = random.sample(range(1,xmax), 2)
    return xs, ys, rs
        
def key(event):
    print("pressed", repr(event.char))
def callback(event):
    #frame.focus_set()
    print("clicked at", event.x, event.y)
def getGpuInfo():
     print("%d device(s) found." % drv.Device.count())
     for i in range(drv.Device.count()):
          dev = drv.Device(i)
          print("Device #%d: %s" % (i, dev.name()))
          print(" Compute Capability: %d.%d" % dev.compute_capability())
          print(" Total Memory: %s GB" % (dev.total_memory()//(1024*1024*1024)))

def main():
    
    root = Tk()
    app = App(root)
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
    x, y, r = app.getObstacles()
    print("Detecting collisions on chosen obstacles:")
    #DO POINT GENERATION/COLLISION DETECTION HERE
    mod = SourceModule("""
    __global__ void check_collisions(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)
    
    check_collisions = mod.get_function("check_collisions")
    
    x = numpy.asarray(x).astype(numpy.float32)#nVidia only supports single precision
    print(x)
    y = numpy.asarray(y).astype(numpy.float32)
    r = numpy.asarray(r).astype(numpy.float32)
    #getGpuInfo()

    dest = numpy.empty_like(x)
    gpuStart = time.time()
    check_collisions(
            drv.Out(dest), drv.In(x), drv.In(y),
            block=(1,512,1), grid=(64,64))
    print("gpu time taken = "+str(time.time()-gpuStart))
    cpuStart = time.time()
    cpuCalc = x*y
    print("cpu time taken = "+str(time.time()-cpuStart))
    print (dest-cpuCalc)
    print(len(dest))
    root.destroy()
main()