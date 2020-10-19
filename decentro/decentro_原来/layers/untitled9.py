#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:22:21 2020

@author: ledi
"""


from keras.layers import Dense,Layer,Conv2D

import keras
import tensorflow as tf
class Linear(keras.layers.Layer):
    def __init__(self, input_dim=32, output_dim=32):
        super().__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            
           
            trainable=True,
        )
        # print(self.w)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        #矩阵相乘 Amn*Bnp 的维度是m*p
        return tf.matmul(inputs, self.w) + self.b
    
x = tf.ones((3, 2))
linear_layer = Linear(2, 5)

#函数式变成直接用 linear_layer(x)
y= linear_layer(x)
print(y.shape)

#更本质通用的用法
y = linear_layer.call(x)
print(y.shape)


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



import keras 
import tensorflow as tf
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        
        print(input_shape)
        self.w = self.add_weight(
            
            #本质就是矩阵相乘 Amn *Bnp 
            #这里会提取输入矩阵最后一层的dim 比如说是Amn的n
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
        # super().build(input_shape)

    def call(self, inputs):
        
        # print(self.input_shape(inputs))
        return tf.matmul(inputs, self.w) + self.b
x = tf.ones((2, 2))
linear_layer = Linear(6)
y = linear_layer(x)
print(y)

a='20190402'
a1=list(a)
a2=[int(k) for k in a1]
a3=sum(a2)

a4=[int(k) for k in list(str(a3))]

print(sum(a4))




