#lesson2

#Import the tensorflow module and call it tf
import tensorflow as tf

#Create a constant value called x, and give it the numerical value 35
#Create a Variable called y, and define it as being the equation x + 5
x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')

#Initialize the variables with tf.global_variables_initializer()
model = tf.global_variables_initializer()

#Create a session for computing the values
with tf.Session() as session:
	session.run(model)  #Run the model 
	print(session.run(y))  #Run the variable y and print out current value


#EXCERCISES 
#1) Constants can also be arrays.
import tensorflow as tf

x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))

#ouput: [40 45 50]

#2) Generate a NumPy array of 10,000 random numbers (called x) and create 
#a Variable storing the equation y = 5x^2 - 3x + 15

#You can generate the NumPy array using the following code:
import numpy as np 
data = np.random.randint(1000, size=10000)
x = tf.constant(data, name='x')
y = tf.Variable(5 * (x**2) - (3 * x) + 15)

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))

#3) You can also update variables in loops, which we will use later for
#machine learning.  
import tensorflow as tf

x = tf.Variable(0, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	for i in range(5):
		x = x + 1 
		print(session.run(x))

#output: 1 2 3 4 5

#4) Using the code from (2) and (3) above, create a program that computes
#the rolling average of the following line of code: 
#np.random.randint(1000).
#In other words, keep looping, and in each loop, call 'np.random.randint(1000)'
#once in that loop, and store the current average in a Variable that keeps
#updating each loop.  
#data = np.random.randint(1000, size=5)
mean = tf.Variable(0., name='x')
n = tf.Variable(0., name='n')

model = tf.global_variables_initializer()

m = 10000

with tf.Session() as session:
	for i in range(5):
		# Generate some random numbers.  Have a look at TensorFlow's random fuctions
		new_random_numbers = np.random.randint(1000, size=m)
		# Add them together.
		sum_of_random_numbers = np.sum(new_random_numbers)

		n += m 
		# The equation for running average.
		mean = (mean * (n-m)/n) + (sum_of_random_numbers / n)

	session.run(model)
	print(session.run(mean))

#5) Use TensorBoard to visualise the graph for some of these examples.
#To run TensorBoard, use the command: 
#tensorboard --logdir=path/to/log-directory
import tensorflow as tf

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("/tmp/basic", session.graph)
	session.run(model)
	print(session.run(y))
		


