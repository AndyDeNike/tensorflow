#Import 'tensorflow'
import os
import numpy as np
import skimage
from skimage import data, 
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


#LOADING AND EXPLORING THE DATA:  BELGIAN TRAFFIC SIGNS

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

#VISUALIZING THE TRAFFIC SIGNS 

#Determine the (random) indexes of the images that you want to see
traffic_signs = [300, 2250, 3650, 4000]

#Fill out the subplots with the random images that you defined
for i in range(len(traffic_signs)):
	plt.subplot(1, 4, i+1)
	plt.axis('off')
	plt.imshow(images[traffic_signs[i]])
	plt.subplots_adjust(wspace=0.5)

plt.show()

#Fill out the subplots with the random images and add shape/min/max values
for i in range(len(traffic_signs)):
	plt.subplot(1, 4, i+1)
	plt.axis('off')
	plt.imshow(images[traffic_signs[i]])
	plt.subplots_adjust(wspace=0.5)
	plt.show()
	print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
												  images[traffic_signs[i]].min(),
												  images[traffic_signs[i]].max()))


#Get the unique labels 
unique_labels = set(labels)

#Initialize the figure
plt.figure(figsize=(15,15))

#Set a counter
i = 1 

#For each unique label,
for label in unique_labels: 
	#You pick the first image for each label 
	image = images[labels.index(label)]
	#Define 64 subplots 
	plt.subplot(8, 8, i)
	#plt.subplots_adjust(wspace=0.5)
	#Don't include axes
	plt.axis('off')
	#Add a title to each subplot
	plt.title("Label {0} ({1})".format(label, labels.count(label)))
	#plt.title(label, y=3.08)
	#Add 1 to the counter
	i += 1 
	#And you plot this first image 
	plt.imshow(image)
#Show the plot
plt.show()