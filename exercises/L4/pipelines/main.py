import numpy as np
from minMaxScaler import *
from plotting import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

X, y = loadExerciseDataAsXY()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# X_scaled, y_scaled = dummyScaler(X_test, y_test)

print("Fitting regressors...")
reg_mlp = MLPRegressor(random_state=42, max_iter=10000).fit(X_train, y_train)
reg_lin = LinearRegression().fit(X_train, y_train)
reg_tree = RandomForestRegressor().fit(X_train, y_train)
print("Done!\n")

print("MLP regression score: ", reg_mlp.score(X_train, y_train))
print("Lin regression score: ", reg_lin.score(X_train, y_train))
print("Tree regression score: ", reg_tree.score(X_train, y_train), "\n")

print("MLP cross val score")
scores = cross_val_score( reg_mlp , X_train , y_train , scoring = "neg_mean_squared_error" , cv = 10)
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std(), "\n")

print("Lin cross val score")
scores = cross_val_score( reg_lin , X_train , y_train , scoring = "neg_mean_squared_error" , cv = 10)
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std(), "\n")

print("Tree cross val score")
scores = cross_val_score( reg_tree , X_train , y_train , scoring = "neg_mean_squared_error" , cv = 10)
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std(), "\n")

params = dict({'hidden_layer_sizes': range(2, 100)})

grid_search = RandomizedSearchCV(reg_mlp, params, random_state=42)
grid_search.fit(X_train, y_train)

print("Grid Search") 
print("Best params:", grid_search.best_params_)
print("Score, train: ", grid_search.score(X_train, y_train))
print("Score, test: ", grid_search.score(X_test, y_test))

# X_scaled, y_scaled = dummyScaler(X, y)

# scikitScaler = MinMaxScaler()
# scikitScalerY = MinMaxScaler()
# scikitScaler.fit(X, y)
# scikitScalerY.fit(y.reshape(-1, 1))

# print(np.array(y).shape)

# X_scaled = scikitScaler.transform(X)
# y_scaled = scikitScalerY.transform(y.reshape(-1, 1))

# linregNotScaled = LinearRegression()
# linregScaled = LinearRegression()

# linregNotScaled.fit(X, y)
# linregScaled.fit(X_scaled, y_scaled)

# plotModel([X, X_scaled], [y, y_scaled], ["Not Scaled", "Scaled"])

# neuralNetwork = MLP()