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
    def __init__(self, path_to_file = None, vertices=None, triangles=None):
        if (path_to_file == None):
            self.vertices = vertices
            self.triangles = triangles
        else:
            mesh = read_triangle_mesh(path_to_file)
            self.vertices = numpy.asarray(mesh.vertices, dtype=numpy.float32)
            self.triangles = numpy.asarray(mesh.triangles, dtype=numpy.int32)
    
    def getUniqueVertices(self):
        # print(self.triangles)
        # print(self.vertices)
        current_vertices_redundant = numpy.array([self.vertices[triangle] for triangle in self.triangles])
        shape = current_vertices_redundant.shape
        # print(shape)
        myset = set(tuple(i) for i in current_vertices_redundant.reshape(shape[0]*shape[1], shape[2]))
        return list(myset)
    
    #takes in a list of Mesh objects and combutes an axis aligned bounding box for each
    #mesh.triangles is a list of lists of vertices, each corresponding to an index in vertices
    #mesh.vertices is a list of list of floats, each corresponding to a coordinate
    #there may be some vectices in mesh.vertices that are not pointed to by a triangle, we
    # ignore such vertices
    @staticmethod
    def getBoundingBoxesCPU(meshes):
        x1 = numpy.zeros(len(meshes))
        y1 = numpy.zeros(len(meshes))
        z1 = numpy.zeros(len(meshes))
        x2 = numpy.zeros(len(meshes))
        y2 = numpy.zeros(len(meshes))
        z2 = numpy.zeros(len(meshes))

        x_lambda = lambda t: t[0]
        y_lambda = lambda t: t[1]
        z_lambda = lambda t: t[2]

        #only check against vertices referenced by triangles
        for i in range( len(meshes)):
            vertices = meshes[i].getUniqueVertices()
            x1[i] = min(vertices, key=x_lambda)[0]
            x2[i] = max(vertices, key=x_lambda)[0]
            y1[i] = min(vertices, key=y_lambda)[1]
            y2[i] = max(vertices, key=y_lambda)[1]
            z1[i] = min(vertices, key=z_lambda)[2]
            z2[i] = max(vertices, key=z_lambda)[2]
        
        boxes = [Box(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]) for i in range (len(meshes))]
        return boxes

# test = Mesh("Meshes/knot.ply")
# print(test.vertices[test.triangles])
# test_box = Mesh.getBoundingBoxesCPU([test])[0]

# vertices = test.getUniqueVertices()[0]
# print(vertices[0])