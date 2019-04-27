#parent class for each shape object that obstacles 
#ideas to make stuff faster:
#1) cast everything as a np.float32 when calling constructor, that way collision 
#   avoiding the costly cast each time we try to run gpu collision
#
#2) don't take square roots when computing kernels, instead just square other thing you're 
#   comparing too 
import numpy
from open3d  import *

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

class Mesh(Shape):
    def __init__(self, path_to_file):
        mesh = read_triangle_mesh(path_to_file)
        self.vertices = numpy.asarray(mesh.vertices, dtype=numpy.float32)
        self.triangles = numpy.asarray(mesh.triangles, dtype=numpy.int32)
    
    @staticmethod
    def getBoundingBox(mesh):
        vertices = mesh.vertices
        x1 = min(vertices, key = lambda t : t[0])[0]
        x2 = max(vertices, key = lambda t : t[0])[0]
        y1 = min(vertices, key = lambda t : t[1])[1]
        y2 = max(vertices, key = lambda t : t[1])[1]
        z1 = min(vertices, key = lambda t : t[2])[2]
        z2 = max(vertices, key = lambda t : t[2])[2]
        return Box(x1, y1, z1, x2, y2, z2)


test = Mesh("Meshes/knot.ply")
print(test.triangles)
print(test.vertices)
Mesh.getBoundingBox(test)