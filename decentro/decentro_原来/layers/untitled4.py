#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:47:02 2020

@author: ledi
"""



#长方形定义
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
#正方形定义
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)
        
square = Square(4)

print(square.area())

# class Cube(Square):
#     def surface_area(self):
#         face_area = super(Square, self).area()
#         return face_area * 6

#     def volume(self):
#         face_area = super(Square, self).area()
#         return face_area * self.length


class Cube(Square):
    
    def __init__(self,length):
        super().__init__(length)
    def surface_area(self):
        face_area = super(Square, self).area()
        return face_area * 6

    def volume(self):
        face_area = super(Square, self).area()
        return face_area * self.length
    
    
cube = Cube(3)
print(cube.surface_area())

print(cube.volume())