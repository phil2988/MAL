import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from ministUtils import *

X, y = MNIST_GetDataSet()
print (X)
print(y)

if X.ndim==3:
    print(f"X.shape={X.shape}") # print X.shape= (70000, 28, 28)
    print("reshaping X..")
    assert y.ndim==1
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    print(f"X.shape={X.shape}") # X.shape= (70000, 784) 
assert X.ndim==2

print("Splitting data into train-test...")
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

print("Fetching a test digit")
some_digit = np.array(X[:0])

print("Converting vector to 28x28 array")
some_digit_image = some_digit.reshape(28, 28)

print("Plotting digit")
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

sgd_clf = SGDClassifier(random_state=42)

print("Fitting data...")
sgd_clf_fit = sgd_clf.fit(X_train, y_train)

print("Predicting...")
sgd_clf_predict = sgd_clf.predict(some_digit)
