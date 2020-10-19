#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:46:57 2020

@author: ledi
"""
import tensorflow as tf
# x has a shape of (2, 3) (two rows and three columns):
x = tf.constant([[1, 1, 1], [1, 1, 1]])
x.numpy()


# sum all the elements
# 1 + 1 + 1 + 1 + 1+ 1 = 6
tf.reduce_sum(x).numpy()

# reduce along the first dimension
# the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
tf.reduce_sum(x, 0).numpy()

# reduce along the second dimension
# the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
tf.reduce_sum(x, 1).numpy()

# keep the original dimensions
tf.reduce_sum(x, 1, keepdims=True).numpy()


# reduce along both dimensions
# the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
# or, equivalently, reduce along rows, then reduce the resultant array
# [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
# 2 + 2 + 2 = 6
tf.reduce_sum(x, [0, 1]).numpy()

import tensorflow as tf
a = tf.ones(shape=[2,3,9])
b = tf.ones(shape=[9,2,6])
c = tf.tensordot(a,b, axes=1)
print(c.shape)


import tensorflow as tf
a = tf.ones(shape=[2,3,4])
b = tf.ones(shape=[6,2,1])
c = tf.tensordot(a,b, axes=2)
print(c.shape)

import tensorflow as tf
a = tf.ones(shape=[2,2,3])
b = tf.ones(shape=[3,2,6])
c = tf.tensordot(a,b, axes=(1,1))

import tensorflow as tf
a = tf.ones(shape=[2,2,3])
b = tf.ones(shape=[3,2,6])
c = tf.tensordot(a,b, axes=((1,2),(0,1)))


























