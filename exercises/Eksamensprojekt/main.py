from modelgeneration import createModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *

units, labels = getCardsAsDataFrame("sigurdslabels")

units = onlyCostAttackAndHealth(units)

X_train, X_test, y_train, y_test = getTrainTestSplit(units, labels)

model = createModel()

model, hist = trainModel(model, X_train, y_train, epochs=50)

printTrainingResults(model, X_test, y_test, hist)
