#LOADING DATA SET
# Import 'datasets' from 'sklearn' library 
from sklearn import datasets
from sklearn.decomposition import PCA, RandomizedPCA
import numpy as np
import matplotlib.pyplot as plt 
# Load in the 'digits' data set
digits = datasets.load_digits()

# Print the 'digits' data 
print(digits)

#Alternate way to retrieve data:
# Import the `pandas` library as `pd`
#import pandas as pd

# Load in the data with `read_csv()`
#digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)

# Print out `digits`
#print(digits)

#Note that if you download the data like this, the data is already split 
#up in a training and a test set, indicated by the extensions .tra and 
#.tes. You’ll need to load in both files to elaborate your project. 
#With the command above, you only load in the training set.


#EXPLORE DATA
# Get the keys of the 'digits' data
print(digits.keys())
# dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])


# Print out the data
print(digits.data)

# Print out the target values 
print(digits.target)
# [0 1 2 ... 8 9 8]

# Print out the description of the 'digits' data 
print(digits.DESCR)



# Isolate the 'digits' data 
digits_data = digits.data 

# Inspect the shape
print(digits_data.shape)
# There are 1797 samples/64 features
# (1797, 64)

# Isolate the target values with 'target'
digits_target = digits.target
# 1797 samples means 1797 target values
#(1797,) 

# Inspect the shape 
print(digits_target.shape)
# All target values contain 10 unique values, 0-9
# In other words, all 1979 target vluaes are made of numbers from 0-9
# 10

# Print the number of unique labels
number_digits = len(np.unique(digits.target))
print(number_digits)

# Isolate the 'images'
digits_images = digits.images 

# Inspect the shape 
print(digits_images.shape)
# 1797 images 8x8 pixels in size
# (1797, 8, 8)


# check that the images and the data are related
# With the numpy method all(), you test whether all array elements along 
# a given axis evaluate to True. In this case, you evaluate if it’s true 
# that the reshaped images array equals digits.data. You’ll see that the 
# result will be True in this case.
# print(np.all(digits.images.reshape((1797,64)) == digits.data))


#VISUALIZE DATA IMAGES W/ MATPLOTLIB

# Figure size(width, height) in inches.  This is your blank canvas where
# all subplots with the images will appear
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images, start filling up figure:
for i in range(64):
	# Inititalize the subplots: add a subplot in the grid of 8 by 8 images, 
	# at the i+1-th position
	ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
	# Display an image at the i-th position
	ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
	# label the image with the target value 
	ax.text(0,7, str(digits.target[i]))
	
# Show the plot 
plt.show()


#ALTERNATE VISUALIZATION SETUP:
# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))

# for every element in the list:
for index, (image, label) in enumerate(images_and_labels[:8]):
	# initialize a subplot of 2X4 at the i+1-th position 
	plt.subplot(2, 4, index + 1)
	# Don't plot any axes
	plt.axis('off')
	# Display images in all subplots 
	plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
	# Add a title to each subplot
	plt.title('Training: ' + str(label))

# Show the plot
plt.show()


#VISUALIZE DATA: PRINCIPAL COMPONENT ANALYSIS(PCA)

# Create a Randomized PCA model that takes two components
randomized_pca = RandomizedPCA(n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model 
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape 
reduced_data_pca.shape 

# Print out the data 
print(reduced_data_rpca)
print(reduced_data_pca)


#Build scatterplot to visualize data:
#List a 10 colors, each color representing a label's datapoint
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
	x = reduced_data_rpca[:, 0][digits.target == i]
	y = reduced_data_rpca[:, 1][digits.target == i]
	plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()