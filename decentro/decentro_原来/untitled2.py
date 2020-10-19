#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:47:47 2020

@author: ledi
"""
import tensorflow as tf 
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where `None` is the batch
# dimension.
input_array = np.random.randint(1000, size=(32, 10))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape)


import collections
d1 = collections.OrderedDict()
d1['b'] = 'B'
d1['a'] = 'A'
d1['c'] = 'C'
d1['2'] = '2'
d1['1'] = '1'
# OrderedDict([('b', 'B'), ('a', 'A'), ('c', 'C'), ('2', '2'), ('1', '1')])

print(d1)

d1={}

d1['b'] = 'B'
d1['a'] = 'A'
d1['2'] = '2'
d1['c'] = 'C'

d1['1'] = '1'
print(d1)




import tensorflow as tf 
from tensorflow.keras.layers import Input,Embedding
import numpy as np

x1=Input(shape=(1,2))

x2=Embedding(1000, 64)(x1)
# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where `None` is the batch
# dimension.
input_array = np.random.randint(1000, size=(32, 10))
model.compile('rmsprop', 'mse')
# print(model.summery())
output_array = model.predict(input_array)
print(output_array.shape)
