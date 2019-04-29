import numpy
import os
import pycuda.autoinit
import pycuda.driver as drv
import random
import time

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import open3d
from Shapes import Mesh
import BoxCollision

if os.name=='nt':
    if (os.system("cl.exe")):
        os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

def detectCollisionCPU(robot, obstacles):

    rob_box = Mesh.getBoundingBoxesCPU(robot)
    obs_box = Mesh.getBoundingBoxesCPU(obstacles)

    collision_found = False
    num_robot_triangles = len(robot[0].triangles)
    collisions = BoxCollision.detectCollisionCPU(rob_box[0], obs_box)
    collision_found = numpy.any(collisions)

    flatten_lambda = lambda l: [item for sublist in l for item in sublist]

    #iterate until either there are no collisions found at a resolution or the minimum resolution is reached
    iterations = 0
    while ((not collision_found) or num_robot_triangles > 1):
        print(iterations)
        iterations += 1
        robot = numpy.array(flatten_lambda([splitMesh(rob) for rob in robot]))
        # print("robot is ", robot)
        # print("obstacles are", obstacles)
        # print("collisions are", collisions) 
        for (i in range(len(robot)))
            obstacles[i] = numpy.array([splitMesh(obs) for obs in obstacles[collisions]]))
        

        rob_boxes = Mesh.getBoundingBoxesCPU(robot)
        obs_boxes = Mesh.getBoundingBoxesCPU(obstacles)
        
        collisions = [False] * len(obstacles)
        for i in range(len(rob_boxes)):
            collisions[i] = numpy.logical_or(collisions, BoxCollision.detectCollisionCPU(rob_boxes[i], obs_boxes))
        collision_found = numpy.any(collisions)
        num_robot_triangles = len(robot[0].triangles)
    
    return collision_found

#breaks up a mesh into two smaller meshes, split along axis with greatest varience at the mean point
#TODO, figure out a way to do this without a bajillion copies
def splitMesh(mesh):

    uVertices = mesh.getUniqueVertices()
    vertices = mesh.vertices
    #calculate the variance along the x, y, and z axes
    variances = [numpy.var([vertex[i] for vertex in uVertices]) for i in  range(3)]
    #split along the axis with the greatest variance
    split_axis = numpy.argmax(variances)

    split_point = numpy.mean(uVertices[split_axis])


    #split the mesh up into two different sets of triangles
    #gotta be a better way to do this, but this should work  
    triangles1 = []
    triangles2 = []
    for triangle in mesh.triangles:
        if vertices[triangle[0]][split_axis] < split_point \
                or mesh.vertices[triangle[1]][split_axis] < split_point \
                or mesh.vertices[triangle[2]][split_axis] < split_point:
            triangles1.append(triangle)
        else:
            triangles2.append(triangle)
    
    mesh1 = Mesh(vertices = mesh.vertices, triangles = triangles1)
    mesh2 = Mesh(vertices = mesh.vertices, triangles = triangles2)

    return (mesh1, mesh2)


test = Mesh("Meshes/knot.ply")
test_box = Mesh.getBoundingBoxesCPU([test])[0]
print("big box bounds are ", test_box.x1, test_box.y1, test_box.z1, test_box.x2, test_box.y2, test_box.z2)
submeshes = splitMesh(test)
lil_box1 = Mesh.getBoundingBoxesCPU([submeshes[0]])[0]
print("lil mesh 1 box bounds are", lil_box1.x1, lil_box1.y1, lil_box1.z1, lil_box1.x2, lil_box1.y2, lil_box1.z2)
lil_box2 = Mesh.getBoundingBoxesCPU([submeshes[1]])[0]
print("lil mesh 2 box bounds are", lil_box2.x1, lil_box2.y1, lil_box2.z1, lil_box2.x2, lil_box2.y2, lil_box2.z2)
print(detectCollisionCPU(numpy.array([submeshes[0]]), numpy.array([submeshes[1]])))