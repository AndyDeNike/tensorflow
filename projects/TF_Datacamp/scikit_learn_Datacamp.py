#LOADING DATA SET
# Import 'datasets' from 'sklearn' library 
from sklearn import datasets
from sklearn import cluster
from sklearn import metrics
from sklearn import svm 
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import scale 
from sklearn.cross_validation import train_test_split
from sklearn.manifold import Isomap
from sklearn.metrics import (homogeneity_score, completeness_score, 
v_measure_score, adjusted_rand_score, 
adjusted_mutual_info_score, silhouette_score)
from sklearn.grid_search import GridSearchCV
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

# Print out the target values (target values also known as labels)
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
# Import matplotlib.pyplot as plt 
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
# Import matplotlib.pyplot as plt 
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


# Build scatterplot to visualize data:
# List a 10 colors, each color representing a label's datapoint
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
	# From the r_d_rpca arrays, you will select all elements at index
	# 0 and 1 respectivly 
	x = reduced_data_rpca[:, 0][digits.target == i]
	y = reduced_data_rpca[:, 1][digits.target == i]
	plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()


#PREPROCESSING KMEANS DATA 
# By scaling data, we shift distribution of each attribute to mean of 0
# and standard deviation of 1 (unit variance)
# Import 'scale()' from sklearn.preprocessing
data = scale(digits.data)


#SPLITTING DATA INTO TRAINING/TEST SETS
# Import 'train_test_split' from sklearn.cross_validation
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, 
digits.target, digits.images, test_size=0.25, random_state=42)

# Number of training features
n_samples, n_features = X_train.shape 

# Print out 'n_samples'
print(n_samples)

# Print out 'n_features'
print(n_features)

# Number of Training labels 
n_digits = len(np.unique(y_train))

# Inspect 'y_train'
print(len(y_train))


#CLUSTERING THE digits DATA 
# Create the KMeans model 
# Import the 'cluster' module from sklearn 
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data 'X_train' to the model 
clf.fit(X_train)


# Visualize images that make up cluster centers:
# Figure size in inches
# Import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(8,3))

# Add title 
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For al labels (0-9)
for i in range(10):
	# Initialize subplots in a grid of 2X5, at i+1th position
	ax = fig.add_subplot(2, 5, 1 + i)
	# Display images 
	ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
	# Don't show the axes 
	plt.axis('off')

# Show the plot
plt.show()


# Next step, predict the labels of the test set: 
# Predict the labels for 'X_test'
y_pred=clf.predict(X_test)

# Print out the first 100 instances of 'y_pred'
print(y_pred[:100])

# Print out the first 100 instances of 'y_test'
print(y_test[:100])

# Study the shape of the cluster centers 
clf.cluster_centers_.shape


# Visualize predicted labels:
# Create an isomap and fit the 'digits' data to it
# Import 'Isomap()' from sklearn.manifold
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster indexes for each samples
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()


#VS PCA VERSION
# Model and fit the `digits` data to the PCA model
# Import 'PCA()' from sklearn.decomposition
X_pca = PCA(n_components=2).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()


#EVALUATION OF CLUSTER MODEL 
# Print out the confusion matrix with 'confusion_matrix()'

# Imported 'metrics' from 'sklearn'
print(metrics.confusion_matrix(y_test, y_pred))

print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(clf.inertia_,
      # homogeneity_score tells what extent of clusters contain only 
      # data points belonging to a single class
      homogeneity_score(y_test, y_pred),
      # measures extent of data points that are members of a given 
      # class and are also elements of the same cluster
      completeness_score(y_test, y_pred),
      # harmonic mean between homogeneity/completeness 
      v_measure_score(y_test, y_pred),
      # rand measures similarity between 2 clusterings and considers 
      # all pairs of samples and counting pairs that are assigned in 
      # the same or different clusters in the predicted/true clusterings
      adjusted_rand_score(y_test, y_pred),
      # mutual is used to compare clusters.  Measures similarity between
      # data points that are in clusterings, accounting for change groupings
      # and takes maximum value of 1 when clusterings are equivalent 
      adjusted_mutual_info_score(y_test, y_pred),
      # silhou score measures how simliar an object is to its own cluster
      # compared to other clusters.  Scores range from -1 to 1, higher value
      # indicates the object is better matched to its own cluster and worse
      # matched to neighboring clusters.  If many points have high value
      # clustering config is good
      silhouette_score(X_test, y_pred, metric='euclidean')))


#TRYING ANOTHER SUPPORT MODEL: SUPPORT VECTOR MACHINES 
# Split the data into training and test sets
# Import 'train_test_split' from sklearn.cross_validation
X_train, X_test, y_train, y_test, images_train, 
images_test = train_test_split(digits.data, digits.target, digits.images,
test_size=0.25, random_state=42)

# Imported the 'svm' model from sklearn
# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model 
svc_model.fit(X_train, y_train)


# setting value of gamma manually VS grid search/cross validation:
# Split the 'digits' data into two equal sets 
X_train, X_test, y_train, y_test = train_test_split(digits.data, 
	digits.target, test_size=0.5, random_state=0)


#Import GridSearch CV from sklearn.grid_search
# Set the parameter candidates 
parameter_candidates = [
	{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
	{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 
	'kernel': ['rbf']},
	]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates,
	n_jobs=-1)

# Train the classifier on training data 
clf.fit(X_train, y_train)

# Print out the results 
print('Best score for training data:', clf.best_score_)
print('Best `C`:', clf.best_estimator_.kernel)
print('Best `gamma`:', clf.best_estimator_.gamma)

# Apply the classifier to the test data, and view the accuracy score
clf.score(X_test, y_test)

# Train and score a new classifier with the grid search parameteres
svm.SVC(C=10, kernal='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)

