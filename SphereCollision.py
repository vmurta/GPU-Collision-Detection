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
def generateRandomSpheres():
    pass

#TODO
def detectCollisionGPU(robot, obstacles):
    pass

#TODO
def detectCollisionCPU(robot, obstacles):
    pass