from modelgeneration import createSequentialModel, doGridSearchCVWithSequentialModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *
import keras.api._v2.keras as keras

units, labels = getCardsAsDataFrame("sigurdslabels_done")

balance = getLabelBalance(labels)

units = onlyCostAttackAndHealth(units)

print3dPlotOfData(units, labels)

X_train, X_test, y_train, y_test = getTrainTestSplit(units, labels)

model = createSequentialModel(
    optimizer=keras.optimizers.SGD(learning_rate=2.101),
    loss=keras.losses.MeanAbsolutePercentageError(),
    hiddenlayerSize=9,
    hiddenlayerAmount=474,
    neuronActivationFunc="tanh",
)

# doGridSearchCVWithSequentialModel(X_train, y_train)

model, hist = trainModel(model, X_train, y_train, epochs=500)

printTrainingResults(model, X_test, y_test, hist)
