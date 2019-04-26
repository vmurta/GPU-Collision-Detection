#parent class for each shape object that obstacles 
#ideas to make stuff faster:
#1) cast everything as a np.float32 when calling constructor, that way collision 
#   avoiding the costly cast each time we try to run gpu collision
#
#2) don't take square roots when computing kernels, instead just square other thing you're 
#   comparing too 
class Shape:
    pass

class Circle(Shape):
    def __init__(self, x, y, radius):
        self.x=x
        self.y=y
        self.rad=radius

class Sphere(Shape):
    def __init__(self, x, y, z, radius):
        self.x=x
        self.y=y
        self.z=z
        self.rad=radius

#grid aligned rectangle
class Rectangle(Shape):
    def __init__(self, x1, y1, x2, y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2

#grid aligned rectangular prism
class Box(Shape):
    def __init__(self, x1, y1, z1, x2, y2, z2):
        self.x1=x1
        self.y1=y1
        self.z1=z1
        self.x2=x2
        self.y2=y2
        self.z2=z2