import preprocessing as pp
from modelgeneration import createModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *

# Train models
def TrainSpectralClustering(SpectralModel, X_train, y_train, X_test):
    SpectralModel.fit(X_train, y_train)
    result_of_training = SpectralModel.fit_predict(X_test) #Bare noget at se om det virker. y ikke implementeret, er der bare for konvention
    print(X_train)
    print(result_of_training)
    return SpectralModel # Returnerer en liste af 

# Plot models

# Plot capacity fitting over epochs (maybe in another file?)

# Plot confusion matrix information (maybe in another file?)