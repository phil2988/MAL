# import numpy as np
from minMaxScaler import *
from plotting import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

X, y = loadExerciseDataAsXY()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train_scaled = dummyScaler(X_train)

print("Fitting regressors...")
reg_mlp = MLPRegressor(random_state=42, max_iter=10000).fit(X_train_scaled, y_train)
reg_lin = LinearRegression().fit(X_train, y_train)
reg_tree = RandomForestRegressor().fit(X_train, y_train)
print("Done!\n")

print("MLP regression score: ", reg_mlp.score(X_train_scaled, y_train))
print("Lin regression score: ", reg_lin.score(X_train, y_train))
print("Tree regression score: ", reg_tree.score(X_train, y_train), "\n")

print("MLP cross val score")
scores = cross_val_score( reg_mlp , X_train_scaled , y_train , scoring = "neg_mean_squared_error" , cv = 10)
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

grid_search = RandomizedSearchCV(reg_mlp, params, random_state=1)
grid_search.fit(X_train_scaled, y_train)

print("Grid Search") 
print("Best params:", grid_search.best_params_)
print("Score, train: ", grid_search.score(X_train_scaled, y_train))
print("Score, test: ", grid_search.score(X_train_scaled, y_train))


# scikitScaler = MinMaxScaler()
# scikitScalerY = MinMaxScaler()
# scikitScaler.fit(X, y)
# scikitScalerY.fit(y.reshape(-1, 1))

# print(np.array(y).shape)

# X_scaled = scikitScaler.transform(X)
# y_scaled = scikitScalerY.transform(y.reshape(-1, 1))

# X_train_scaled = dummyScaler(X_train)
# X_test_scaled = dummyScaler(X_test)

# linregNotScaled = LinearRegression()
# linregScaled = LinearRegression()

# y_notScaled_pred = linregNotScaled.fit(X_train, y_train).predict(X_test)
# y_scaled_pred = linregScaled.fit(X_train_scaled, y_train).predict(X_test_scaled)

# print(f"Not Scaled Linear Regressor Score: {linregNotScaled.score(X_train, y_train)}")
# print(f"Scaled Linear Regressor Score: {linregScaled.score(X_train_scaled, y_train)}")

# label = ["Not Scaled", "Scaled"]

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(2)

# axs[0].plot()

# axs[0].plot(X_train, y_train, 'b.')
# axs[0].plot(X_test, y_test, 'g.')
# axs[0].plot(X_test, y_notScaled_pred, 'r-')
# axs[0].set_title([label[0]])
# axs[0].set_xlabel("GDP per capita")
# axs[0].set_ylabel("Life satisfaction")

# axs[1].plot(X_train_scaled, y_train, 'b.')
# axs[1].plot(X_test_scaled, y_test, 'g.')
# axs[1].plot(X_test_scaled, y_scaled_pred, 'r-')
# axs[1].set_title([label[1]])
# axs[1].set_xlabel("Scaled GDP per capita")
# axs[1].set_ylabel("Life satisfaction")

# fig.tight_layout(pad=1)
# plt.show()


# neuralNetwork = MLP()