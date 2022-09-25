import numpy as np
from minMaxScaler import *
from plotting import *
from sklearn.linear_model import LinearRegression


X, y = loadExerciseDataAsXY()

X_scaled, y_scaled = dummyScaler(X, y)

linregNotScaled = LinearRegression()
linregScaled = LinearRegression()

linregNotScaled.fit(X, y)
linregScaled.fit(X_scaled, y_scaled)

plotModel([X, X_scaled], [y, y_scaled], ["Not Scaled", "Scaled"])
plotModel(X_scaled, y_scaled, "Scaled")
