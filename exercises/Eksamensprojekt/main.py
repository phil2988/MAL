from modelgeneration import createModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *

cards = getCardsAsDataFrameByPath()

units = removeNonUnits(cards)

units = onlyCostAttackAndHealth(units)

X_train, X_test, y_train, y_test = getTrainTestSplit_test(units)

model = createModel()

model, _ = trainModel(model, X_train, y_train, X_test, y_test)

printTrainingResults(model, X_test, y_test)