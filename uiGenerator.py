# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:10:11 2019

@author: Vosburgh
"""

from tkinter import Frame, Canvas, Tk, BOTH, Button, RIGHT, LEFT, Y, X, BOTTOM
import random
class CollisionUI(Frame):
    def __init__(self, master):
        super().__init__()
        self.initUI(master)
    def getObstacles(self):
        return self.xs, self.ys, self.rs
    def initUI(self, master):
        #'''
        self.score=0
        self.master.title("magic")
        frame=Frame(master)
        frame.pack()
        self.qbut = Button(
                frame, text="Calculate", fg="red", command=frame.quit
                )
        self.hibut = Button(
                frame, text="Refresh", command=self.new_numbers
                )
        self.qbut.pack(side=LEFT)
        self.hibut.pack(side=RIGHT)
        #'''
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self, width = 400, height = 400, scrollregion = (0,0,400,400))
        self.canvas.create_rectangle(2, 2, 400, 400, outline="#000", width=2)

        self.xs, self.ys, self.rs = CollisionUI.generateNumbers()
        i =0
        while i < len(self.xs):
            self.canvas.create_oval(self.xs[i]-self.rs[i],self.ys[i]-self.rs[i],self.xs[i]+self.rs[i],self.ys[i]+self.rs[i],outline="#fb0", fill="#fb0")
            i=i+1
        self.canvas.configure(scrollregion=(0, 0, 400, 400))
        self.canvas.pack(side = BOTTOM, fill=BOTH,expand=1)
       
    def say_hi(self):
        self.score=self.score+1
        print("score: " + str(self.score))
    def new_numbers(self):
        self.xs, self.ys, self.rs = CollisionUI.generateNumbers()
        self.canvas.delete("all")
        i =0
        while i < len(self.xs):
            self.canvas.create_oval(self.xs[i]-self.rs[i],self.ys[i]-self.rs[i],self.xs[i]+self.rs[i],self.ys[i]+self.rs[i],outline="#fb0", fill="#fb0")
            i=i+1
        self.canvas.create_rectangle(1, 1, 399, 399, outline="#000", width=1)
        self.canvas.pack(fill=Y,expand=0, side=BOTTOM)

    def generateNumbers():
        n = 100
        xmax = 400
        ymax = xmax
        rmax = 30
        xs = random.sample(range(1, xmax), n)#0 at Left, moves right
        ys = random.sample(range(1, ymax), n)#0 at top, moves down
        rs = [random.randint(1,rmax) for i in range(n)]
        print("xs:", xs)
        print("ys", ys)
        print("rs:", rs)
        #point = random.sample(range(1,xmax), 2)
        return xs, ys, rs
            
    def key(event):
        print("pressed", repr(event.char))
    def callback(event):
        #frame.focus_set()
        print("clicked at", event.x, event.y)

def main():
    
    root = Tk()
    app = CollisionUI(root)
    frame = Frame(root, width=100, height=400)
    frame.bind("<Key>", CollisionUI.key)
    frame.bind("<Button-1>", CollisionUI.callback)
    frame.focus_set()
    frame.pack()
    #w = Label(root, text="Hello world!")
    
    #w.pack()f
    root.geometry("400x400")
    root.mainloop()
    #print("final score: "+str(app.score))
    x, y, r = app.getObstacles()
    #DO POINT GENERATION/COLLISION DETECTION HERE
    print("Detecting collisions on chosen obstacles:")
    print(x)
    root.destroy()
main()