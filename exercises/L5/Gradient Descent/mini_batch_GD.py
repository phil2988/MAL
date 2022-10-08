from sklearn.linear_model import SGDRegressor
from utilsModule import *
import numpy as np

X, X_b, y = GenerateData()

m = len(X_b)

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

plotMyMBSGDRegressor(X, y, X_b, theta, n_iterations, minibatch_size, m)