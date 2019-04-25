#parent class for each shape object that obstacles 
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
    pass
    
class TriangleMesh(Shape):
    pass