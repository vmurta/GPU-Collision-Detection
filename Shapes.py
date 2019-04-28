#parent class for each shape object that obstacles 
#ideas to make stuff faster:
#1) don't take square roots when computing kernels, instead just square other thing you're 
#   comparing too 
# use more memory: store some number of bounding boxes with the mesh itself
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
    
    #takes in a list of Mesh objects and combutes an axis aligned bounding box for each
    @staticmethod
    def getBoundingBoxesCPU(meshes):
        x1 = numpy.zeros(len(meshes))
        y1 = numpy.zeros(len(meshes))
        z1 = numpy.zeros(len(meshes))
        x2 = numpy.zeros(len(meshes))
        y2 = numpy.zeros(len(meshes))
        z2 = numpy.zeros(len(meshes))

        x_lambda = lambda t : t[0]
        y_lambda = lambda t : t[1]
        z_lambda = lambda t : t[2]

        for i in range( len(meshes)):    
            vertices = meshes[i].vertices
            x1[i] = min(vertices, key = x_lambda)[0]
            x2[i] = max(vertices, key = x_lambda)[0]
            y1[i] = min(vertices, key = y_lambda)[1]
            y2[i] = max(vertices, key = y_lambda)[1]
            z1[i] = min(vertices, key = z_lambda)[2]
            z2[i] = max(vertices, key = z_lambda)[2]
        return Box(x1, y1, z1, x2, y2, z2)


test = Mesh("Meshes/knot.ply")
print(test.triangles)
print(test.vertices)
Mesh.getBoundingBoxesCPU([test])