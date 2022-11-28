from utils import *
from sklearn.neural_network import MLPClassifier

tmp = getAllData()
print("Loading data...")
(X_train, y_train), (X_test, y_test) = getTestTrainSplit()
print("Done!")

print("Creating model...")
model = MLPClassifier(activation="relu", solver="adam", learning_rate="adaptive", random_state="42")
print("Done!")

print("Fitting model...")
model.fit(X_train, y_train)
print("Done!")

print("Predicting...")
predictions = model.predict(X_test)
print("Done!")

n = 100
print("Checking result...")
print("Looking at card: ", X_test[n])
print("Labeled value: ", y_test[n])
print("Predicted value: ", predictions[n])