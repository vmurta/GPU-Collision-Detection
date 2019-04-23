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
import time

from pycuda.compiler import SourceModule
from uiGenerator import CollisionUI
from tkinter import Frame, Tk

#if on windows, try to find cl.exe
if os.name=='nt':
    if (os.system("cl.exe")):
        os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

def main():   
    root = Tk()
    app = CollisionUI(root)
    frame = Frame(root, width=100, height=400)
    frame.bind("<Key>", CollisionUI.key)
    frame.bind("<Button-1>", CollisionUI.callback)
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
    
    dest = numpy.empty_like(x)
    gpuStart = time.time()
    check_collisions(
            drv.Out(dest), drv.In(x), drv.In(y),
            block=(400,2,1), grid=(1,1))
    print("gpu time taken = "+str(time.time()-gpuStart))
    cpuStart = time.time()
    cpuCalc = x*y
    print("cpu time taken = "+str(time.time()-cpuStart))
    print (dest-cpuCalc)
    print(len(dest))
    root.destroy()
main()