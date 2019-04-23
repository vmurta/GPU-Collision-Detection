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
    pass

def detectCollisionCPU(robot, obstacles):
    pass

