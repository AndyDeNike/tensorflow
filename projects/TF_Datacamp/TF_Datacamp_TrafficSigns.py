#Import 'tensorflow'
import os
import numpy as np
import skimage
from skimage import data, transform
from skimage.color import rgb2gray
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

#[ 5 12 21 32]


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
#1 

#Print the number of 'images's elements
print(np.array(images).size)
#4575

#Print the first instance of 'images'
print(np.array(images)[0])

#Print the 'labels' dimensions 
print(np.array(labels).ndim)
#1

#Print the number of 'labels''s elements 
print(np.array(labels).size)
#4575

#Count the number of labels
print(len(set(np.array(labels))))
#62


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
#shape: (236, 256, 3), min: 0, max: 255
#shape: (133, 164, 3), min: 0, max: 255
#shape: (122, 121, 3), min: 0, max: 255
#shape: (123, 123, 3), min: 0, max: 215


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


#FEATURE EXTRACTIONN
#RESCALING IMAGES 
images28 = [transform.resize(image, (28, 28)) for image in images]
print(np.array(images28).shape)
#(4575, 28, 28, 3) images are 784-dimensional 

#Check result of rescaling:
for i in range(len(traffic_signs)):
	plt.subplot(1, 4, i+1)
	plt.axis('off')
	plt.imshow(images28[traffic_signs[i]])
	plt.subplots_adjust(wspace=0.5)
	plt.show()
	print("shape: {0}, min: {1}, max: {2}".format(images28[traffic_signs[i]].shape,
												  images28[traffic_signs[i]].min(),
												  images28[traffic_signs[i]].max()))

#shape: (28, 28, 3), min: 0.05070028011204415, max: 1.0
#shape: (28, 28, 3), min: 0.0, max: 1.0
#shape: (28, 28, 3), min: 0.027040816326528917, max: 1.0
#shape: (28, 28, 3), min: 0.04141781712685025, max: 0.8151298019207689


#IMAGE CONVERSION TO GREYSCALE
#Convert 'images28' to an array 
images28 = np.array(images28)

#Convert 'images28' to grayscale 
images28 = rgb2gray(images28) 

for i in range(len(traffic_signs)):
	plt.subplot(1, 4, i+1)
	plt.axis('off')
	plt.imshow(images28[traffic_signs[i]], cmap="gray")
	plt.subplots_adjust(wspace=0.5)

#Show the plot 
plt.show()


#DEEP LEARNING WITH TENSORFLOW 
#MODELING THE NEURAL NETWORK 
#Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

#Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

#Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

#Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
	 																 logits = logits))

#Define an optimaizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#Convert logits to label indexes 
correct_pred = tf.argmax(logits, 1)

#Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)