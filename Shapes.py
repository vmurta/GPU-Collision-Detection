#parent class for each shape object that obstacles 
#ideas to make stuff faster:
#1) cast everything as a np.float32 when calling constructor, that way collision 
#   avoiding the costly cast each time we try to run gpu collision
#
#2) don't take square roots when computing kernels, instead just square other thing you're 
#   comparing too 
import numpy

class Shape:
    pass

class Circle(Shape):
    def __init__(self, x, y, radius):
        self.x= numpy.float32(x)
        self.y= numpy.float32(y)
        self.rad= numpy.float32(radius)

class Sphere(Shape):
    def __init__(self, x, y, z, radius):
        self.x= numpy.float32(x)
        self.y= numpy.float32(y)
        self.z= numpy.float32(z)
        self.rad= numpy.float32(radius)

#grid aligned rectangle
class Rectangle(Shape):
    def __init__(self, x1, y1, x2, y2):
        self.x1=numpy.float32(x1)
        self.y1=numpy.float32(y1)
        self.x2=numpy.float32(x2)
        self.y2=numpy.float32(y2)

#grid aligned rectangular prism
class Box(Shape):
    def __init__(self, x1, y1, z1, x2, y2, z2):
        self.x1=numpy.float32(x1)
        self.y1=numpy.float32(y1)
        self.z1=numpy.float32(z1)
        self.x2=numpy.float32(x2)
        self.y2=numpy.float32(y2)
        self.z2=numpy.float32(z2)