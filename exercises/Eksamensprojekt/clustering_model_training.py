import preprocessing as pp
from modelgeneration import createSequentialModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 
import pandas as pd 

# Train models
def TrainSpectralClustering(SpectralModel, X_train, y_train):
    SpectralModel.fit(X_train, y_train)
    #result_of_training = SpectralModel.fit_predict(X_test) #Bare noget at se om det virker. y ikke implementeret, er der bare for konvention
    #print(X_train)
    #print(result_of_training)
    return SpectralModel # Returnerer en liste af 

# Plot models

# Plot capacity fitting over epochs (maybe in another file?)

# Plot confusion matrix information (maybe in another file?)