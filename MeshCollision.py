import numpy
import os
import pycuda.autoinit
import pycuda.driver as drv
import random
import time

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import open3d

if os.name=='nt':
    if (os.system("cl.exe")):
        os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

#sort and sweep might be useless, no need to sort obstacles if we're only checking collision against
# a single mesh (the robot)
def sortAndSweep(robot, obstacles)
    pass