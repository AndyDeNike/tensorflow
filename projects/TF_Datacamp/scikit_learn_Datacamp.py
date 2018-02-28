# Import 'datasets' from 'sklearn'
from sklearn import datasets

# Load in the 'digits' data
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