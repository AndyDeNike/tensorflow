#Import 'tensorflow'
import os
import numpy as np
import skimage
from skimage import data
import tensorflow as tf 
import matplotlib.pyplot as plt

#Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

#Multiply
result = tf.multiply(x1, x2)

#Initialize the Session
#sess = tf.Session()

#Print the result 
#print(result)

#Close the session 
#sess.close()

#Initialize Session and run 'result' (instead of previous commented code)
with tf.Session() as sess:
	output = sess.run(result)
	print(output)


#Loading and Exploring the Data:  Belgian Traffic Signs 

#NOTE: The os.path module is always the path module suitable for the 
#operating system Python is running on, and therefore usable for local 
#paths.

def load_data(data_directory):
	#Using list comprehension, we iterate through files in 
	#train_data_directory (Training folder) using 'd' and if it is a 
	#directory, add it to the directories list
	#Remember that each subdirectory represents a label.
	directories = [d for d in os.listdir(data_directory)
				   if os.path.isdir(os.path.join(data_directory, d))]
	#Initialize two lists:
	labels = []
	images = []
	#Iterate through directories and create 'file_names' list comp 
	#which compiles all images with .ppm ending
	#a for loop then appends all images/labels into respective lists
	for d in directories:
		label_directory = os.path.join(data_directory, d)
		file_names = [os.path.join(label_directory, f)
					  for f in os.listdir(label_directory)
					  if f.endswith(".ppm")]
		for f in file_names:
			images.append(skimage.data.imread(f))
			labels.append(int(d))
	return images, labels 

#ROOT_PATH is directory with all your training/test data 
ROOT_PATH = "/Users/andrew/Desktop/Archive/Coding/tensorflow/projects/TF_Datacamp"
#Join specific paths to ROOT_PATH with join()
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

#Load the data into the train_data_directory variable
#The load_data() function gathers all subdirectories present in t_d_d
images, labels = load_data(train_data_directory)


#TRAFFIC SIGN STATISTICS

#Print the 'images' dimensions 
print(np.array(images).ndim)

#Print the number of 'images's elements
print(np.array(images).size)

#Print the first instance of 'images'
print(np.array(images)[0])

#Print the 'labels' dimensions 
print(np.array(labels).ndim)

#Print the number of 'labels''s elements 
print(np.array(labels).size)

#Count the number of labels
print(len(set(np.array(labels))))


#Make a histogram with 62 bins of the 'labels' data
plt.hist(labels, 62)

#Show the plot 
plt.show()