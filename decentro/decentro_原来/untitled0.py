#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 19:43:22 2020

@author: ledi
"""

a = [1, 2, 3, 4, 5, 6]
b = filter(lambda x: x % 2 == 1, a)
print(list(b))                     