from sklearn.linear_model import SGDRegressor
from utilsModule import *
import numpy as np

X, X_b, y = GenerateData()

m = len(X_b)
n_epochs = 50
np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
print(f'stochastic gradient descent theta={theta.ravel()}')
print(f'Scikit-learn SGDRegressor "thetas": sgd_reg.intercept_={sgd_reg.intercept_}, sgd_reg.coef_={sgd_reg.coef_}')

plotMySGDRegressor(X, y, X_b, X_new, X_new_b, theta, n_epochs, m)