#TensorFlow > Develop > Programmer's Guide > Introduction

#attributes are accessed from respective modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#the central unit of data in TensorFlow is the 'TENSOR'. A tensor
#consists of a set of primitive values shaped into an array of 
#any number of dimensions.  

#A tensor's 'RANK' is its number of dimensions, while its 'SHAPE' is 
#a tuple of integers specifying the array's length along each dmnsion

#3. # a rank 0 tensor; a scalar with shape [],
#[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
#[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
#[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

#TensorFlow uses numpy arrays to represent tensor 'values'.


#TF Core programs consist of two sections:
#Building the computational graph (tf.Graph)
#Running the computational graph (tf.Session)

#GRAPH 
#A computational graph is a series of TF operations arragned into a graph
#Graph is compsed of two object types:
#OPERATIONS(ops): Nodes of graph desribing calculations that consume/produce 
#tensors
#TENSORS:  Edges of graph representin the values that will flow through 
#graph 
#IMPORTANT: tf.Tensors do not have values, they are just represent elements 
#in the computational graph

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

#Tensor("Const:0", shape=(), dtype=float32)
#Tensor("Const_1:0", shape=(), dtype=float32)
#Tensor("add:0", shape=(), dtype=float32)

#each operation in graph has unique name, independent to the names 
#assigned to them in Python.  TENSORS are named after the operation 
#that produces them (ex. add:0)

#TENSORBOARD
#save computation graph to a TesorBoard summary file:
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
print('this allows the summary file to appear but why?')
#to access TensorBoard: 'tensorboard --logdir .'



