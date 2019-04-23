#parent class for each shape object that obstacles 
class Shape:
    pass

class Circle(Shape):
    def __init__(self, x, y, radius):
        self.x=x
        self.y=y
        self.rad=radius

class Box(Shape):
    pass
    
class Mesh(Shape):
    pass