# GPU-Collision-Detection
Final project for COMP-781, Robotics, Spring 2019 at UNC Chapel Hill
This is a CUDA-based collision detector for motion planning.



##Dependencies 
###CUDA
You'll need a CUDA-enabled NVidia GPU in order to use this package.
To check if your GPU supports CUDA, consult https://developer.nvidia.com/cuda-gpus
You'll also need CUDA version 10.0 or above https://developer.nvidia.com/cuda-downloads
Note: This can be incredibly difficult to install, it took the authors of this framework a good 6 hours each

###Python Depenpencies
You'll need Python-3.7 for any of these
Additional dependencies are 
- PyCUDA
- NumPy
- TkInter

##Running 
From the command line simply run 
```shell
python uiGenerator.py
```
This will pop up the GUI which allows you to play around with the collision detector.
To use the collision detector elsewhere, you'll need to include 
```python
import Shapes
import RectangleCollision
import BoxCollision
import CircleCollision
import SphereCollision
```
in your python code.


