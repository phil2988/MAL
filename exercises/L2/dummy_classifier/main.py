from binaryClassifier import DummyClassifier
from mnistUtils import *
import numpy as np

X, y = MNIST_GetDataSetXy()

print("Fetching a test digit")
some_digit = X.to_numpy()[0]

# MNIST_PlotDigit(some_digit)

print("Splitting data into train-test...")
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train.astype(np.int8) == 5) 
y_test_5 = (y_test.astype(np.int8) == 5)

# print(f"Label for digit is: {y_train[0]}")

# clf = MNIST_CreateSGDClassifier(X_train, y_train_5)

# print("Predicting...")
# prediction = clf.predict([some_digit])

# print(f"Prediction is {prediction}")

from sklearn.model_selection import cross_val_score

print("Creating classifier")
never_5_clf = DummyClassifier()

print("Score from dummy classifier: " + str(never_5_clf.score(X_train,y_train_5)))
