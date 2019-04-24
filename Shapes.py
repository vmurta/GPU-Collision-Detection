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
    pass

#grid aligned rectangular prism
class Box(Shape):
    pass
    
class TriangleMesh(Shape):
    pass