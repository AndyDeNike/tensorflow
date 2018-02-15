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
	