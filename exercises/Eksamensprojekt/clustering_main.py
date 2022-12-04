import clustering_model_generation
import clustering_model_training
from preprocessing import *

units, labels = getCardsAsDataFrame("sigurdslabels")

units = onlyCostAttackAndHealth(units)

X_train, X_test, y_train, y_test = getTrainTestSplit(units, labels)

model = clustering_model_generation.CreateSpectralClustering()

print(model)

model = clustering_model_training.TrainSpectralClustering(model, X_train, y_train, X_test)

print(model)
