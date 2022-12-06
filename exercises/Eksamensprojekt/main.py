from modelgeneration import createSequentialModel, doGridSearchCVWithSequentialModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *

units, labels = getCardsAsDataFrame("sigurdslabels")

units = onlyCostAttackAndHealth(units)

X_train, X_test, y_train, y_test = getTrainTestSplit(units, labels)

model = createSequentialModel()

# doGridSearchCVWithSequentialModel(X_train, y_train)

model, hist = trainModel(model, X_train, y_train, epochs=200)

printTrainingResults(model, X_test, y_test, hist)