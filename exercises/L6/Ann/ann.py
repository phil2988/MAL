# Data and plot generation for ANN Exercise..

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import numpy as np

X = np.linspace(-10,10,1000)
y = (np.tanh(X * -2.00 + 12.00)) * -2.00 - 3.05e-5 + np.tanh(X * -2.00 + 4.00)  * 3.00 - 3.05e-5

plt.plot(X, y, 'b.')

X = X.reshape(-1,1) # Scikit-algoritmer kræver (:,1)-format

mlp = MLPRegressor(activation = 'tanh',    # aktiveringsfunktion 
                   hidden_layer_sizes = 5, # antal skjulte neuroner
                   alpha = 1e-5,           # regulariseringsparameter, her meget lille
                   solver = 'lbfgs',       # quasi-Newton solver
                   max_iter=1000,
                   verbose = True)
mlp.fit(X, y)

plt.plot(X, mlp.predict(X), 'rx', ms=1)
plt.xlabel('X')
plt.ylabel('y')
plt.title("ANN, MLPRegressor fit")

plt.show()

# print("weights", mlp.coefs_)
# print("biases", mlp.intercepts_)

# y = (np.tanh(X * -2.00 + 12.00)) * -2.00 - 3.05e-5 + np.tanh(X * -2.00 + 4.00)  * 3.00 - 3.05e-5
# plt.plot(X, y, "b-")
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title("MLP fitted function plot")

# plt.show()

# y1 = (np.tanh(X * -2.00 + 12.00)) * -2.00 - 3.05e-5 
# y2 = (np.tanh(X * -2.00 + 4.00))  * 3.00 - 3.05e-5

# plt.plot(X, y1, "g-")
# plt.plot(X, y2, "r-")
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title("MLP fitted function plot")

# plt.show()

# X = np.linspace(-3,3,1000)
# y = np.sinc(X)

# plt.plot(X, y, '.')

# X = X.reshape(-1,1) # Scikit-algoritmer kræver (:,1)-format

# plt.show()