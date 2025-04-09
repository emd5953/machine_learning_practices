# -*- coding: utf-8 -*-
"""KNN-v1_modified.ipynb

Modified version of the KNN example to test with new test points and different values of k.
"""

print("Hello world")

a = 10
b = 40
c = a + b
print("Sum:", c)

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

"""# K Nearest Neighbor

This algorithm selects k nearest neighbors from a given data point and assigns labels according to the neighborhood.

**Advantages:**
*   No assumption about data
*   Insensitive to outliers

**Disadvantages**
*   Requires huge memory
*   Requires computations

Often it is called ***instance based*** or ***lazy method***. It saves all the instances and searches for neighbors or closest elements.
K is a very important hyper-parameter. After finding the labels of K nearest neighbor it then uses some aggregating technique. 
*   Majority Voting (classification)
*   Weighted Voting (classification)
*   Uniform (regression)
*   Distance weighted (regression)

## Let's create a dummy dataset and see how it works
"""

feature_data = np.asarray([[0.0, 1.0],
                           [-0.01, 1.1],
                           [1.1, 0.01],
                           [0.99, -0.01]])

labels = np.asarray([1, 1, 0, 0])

"""## Visualize the data"""

plt.scatter(feature_data[:, 0], feature_data[:, 1], s=(labels + 1) * 20, c=(labels + 1) * 20)
plt.title("Dummy Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

"""## Implementation of KNN"""

from numpy import *
import operator

def classifyKNN(test_x, X, y, k):
  # Ensure the y labels are in vector format
  y = np.reshape(y, (y.shape[0],))
  
  dataSetSize = X.shape[0]
  diffMat = np.tile(test_x, (dataSetSize, 1)) - X
  sqDiffMat = diffMat ** 2
  sqDistances = sqDiffMat.sum(axis=1)
  distances = sqDistances ** 0.5

  sortedDistIndices = distances.argsort()

  classCount = {}
  for i in range(k):
    voteIlabel = y[sortedDistIndices[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

"""## Test with a simple point"""

result1 = classifyKNN([0.8, 0], feature_data, labels, 3)
print("Prediction for [0.8, 0] with k=3:", result1)

result2 = classifyKNN([0.2, 0.9], feature_data, labels, 3)
print("Prediction for [0.2, 0.9] with k=3:", result2)

"""## Modified tests: Changing test data and the value of k"""

# Define new test points and different k values to experiment with
new_test_points = [[0.5, 0.5], [0.8, 0.2]]
k_values = [1, 3, 4]

for test in new_test_points:
    for k in k_values:
        prediction = classifyKNN(test, feature_data, labels, k)
        print("Prediction for", test, "with k=", k, "is:", prediction)
