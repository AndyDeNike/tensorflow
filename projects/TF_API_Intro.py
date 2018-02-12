#TensorFlow > Develop > Programmer's Guide > Introduction

#attributes are accessed from respective modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#the central unit of data in TensorFlow is the 'tensor'. A tensor
#consists of a set of primitive values shaped into an array of 
#any number of dimensions.  

#A tensor's 'rank' is its number of dimensions, while its 'shape' is 
#a tuple of integers specifying the array's length along each dmnsion

#3. # a rank 0 tensor; a scalar with shape [],
#[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[#[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
#[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

#TensorFlow uses numpy arrays to represent tensor 'values'.


#TF Core programs consist of two sections:
#Building the computational graph (tf.Graph)
#Running the computational graph (tf.Session)

