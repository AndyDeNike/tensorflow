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
#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())
#print('this allows the summary file to appear but why?')
#to access TensorBoard: 'tensorboard --logdir .'

#SESSION 
#to evaluate tensors, instantiate 'tf.Session' object (session)
#following code creates tf.Session object and invokes 'run' to evaluate
#'total' tensor created 
sess = tf.Session()
print(sess.run(total))
#7.0
#TensorFlow backtracks through the graph and runs all nodes that provide
#input to the request output node. 

#Pass multiple tesnors, RUN handles any combination of tuples/dictionaries
print(sess.run({'ab':(a, b), 'total':total}))

#During a call to 'ft.Session.run', any 'tf.Tensor' only has a single
#value.  Ex: 'tf.random_uniform' to produce a 'tf.Tensor' that 
#generates a random 3-element vector(with values in [0,1)):
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
#[ 0.52917576  0.64076328  0.68353939]
#[ 0.66192627  0.89126778  0.06254101]
#(
  #array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
  #array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
#)

#FEEDING 
#A graph can be parameterized to accept external inputs, known as 
#placeholders which 'promise' to provide a value later, like a funciton
#argument
x = tf.placeholder(tf.float32) #x, y are tensors!
y = tf.placeholder(tf.float32)
z = x + y
#using feed_dict we can evaluate multiple inputs
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))


#DATASETS 
#Prefered method of streaming data into a model 
#to get a runnable 'tf.Tensor' from a Dataset you must convert it to 
#a 'tf.data.Iterator' and then call the Iterator's 'get_next' method.
#Simplest way to create an Iterator is with the 'make_one_shot_iterator'
#method.  
my_data = [
	[0, 1,],
	[2, 3,],
	[4, 5,],
	[6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
	try:
		print(sess.run(next_item))
	except tf.errors.OutOfRangeError:
		break


#LAYERS 
#Prefered way to add trainable parameters to graph 
#Ex: A 'densely-connected layer' performs a weighted sum across all
#inputs for each output and applies an optional 'activation function'.
#The connection weights and biases are managed by the layer object.

#CREATING LAYERS 
#The following code creates a 'Dense' layer that takes a batch of input
#vectors and produces a single output value for each.
#To apply a layer to input, call it as though it was a function.
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1) #Dense performs weighted sum 
y = linear_model(x)

#INITIALIZING LAYERS
#Later contains variables that must be initialized before use.
#While we can initialize variables individually, to initialize all:
init = tf.global_variables_initializer()
sess.run(init)
#global_variables_initializer only initializes variables before it session
#it is recommended to be one of the last things added during graph creation.

#EXECUTING LAYERS
#We can evaluate the 'linear_model' output tensor (as we would any tensor).
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
#Ex output: 
#[[-3.41378999]
# [-9.14999008]]

#LAYER FUNCTION SHORTCUTS
#TF supplies a shortcut function that creates and runs layer in a single 
#call (unlike 'tf.layers.Dense') as follows: 
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
#Downside to the above shortcut is it makes use of 'tf.layers.Layer'
#object impossible, as well as debugging/layer reuse unavailable.


#FEATURE COLUMNS 
#Easiest way to experiment with feature columns is using 
#'tf.feature_column.input_layer'
#This function only accepts dense columns as inputs SO to view result
#of a categorical column you must wrap it in a 
#'tf.feature_column.indicator_column'
features = {
	'sales' : [[5], [10], [8], [9]],
	'department' : ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
		'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
	tf.feature_column.numeric_column('sales'),
	department_column
]

inputs = tf.feature_column.input_layer(features, columns)
#running 'inputs' tensor will parse the 'features' into a batch of 
#vectors! 
#Feature columns have internal state, like layers, and need to be
#initialized.
#Categorical columns use 'lookup tables' internally and reqire a 
#seperate initialization operation, 'tf.tables_initializer'.
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
#Once internal state has been initialized, you can run inputs like any
#other 'tf.Tensor'
print(sess.run(inputs))
#This shows the feature columns have packed input vectors, with the 
#one-hot "department" as the first two indices and "sales" as third. 
#[[  1.   0.   5.]
# [  1.   0.  10.]
# [  0.   1.   8.]
# [  0.   1.   9.]]







