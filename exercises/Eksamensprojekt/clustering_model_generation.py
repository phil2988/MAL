import preprocessing as pp
from modelgeneration import createModel
from modeltraining import printTrainingResults, trainModel
from preprocessing import *
#from modelgeneration import createModel
#from modeltraining import printTrainingResults, trainModel

print("Loading data...")
cards = pp.getCardsAsDataFrame()
units = pp.removeNonUnits(cards)
units = pp.onlyCostAttackAndHealth(units)

print("Data loaded!")

print("Making training and testing set")
X_train, X_test, y_train, y_test = getTrainTestSplit_test(units)

print("Creating models")

n_clusters = 3

#Remove this, it cannot take variable clusters... Test it out first just to see what happens...
#def CreateMeanShiftModel():
#    model = 0
#    return model

# https://scikit-learn.org/stable/modules/grid_search.html
# Foreslår selv halved
# https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6
# For at finde score
# Score på unsupervised er svært, da man ofte bare kigger på dem og ser om det giver mening.
# Men det giver første sådan rigtig mening at kigge på når man anvender alle dimensionerne...
# Prøv at lavet det og se om man kan se noget som helst

def CreateKMeansModel():
    model = 0
    return model

def CreateHierarchicalClustering():
    model = 0
    return model

def CreateWardHierarchicalClustering():
    model = 0
    return model

def CreateAgglomerativeClustering():
    model = 0
    return model

def CreateOPTICS():
    model = 0
    return model

def CreateBisectingKMeans():
    model = 0
    return model

def CreateBIRCH(): # Unknown if this works
    model = 0
    return model

def CreateSpectralClustering():
    # Parameter search
    # Grid search?
    model = 0
    return model

# Træn modellerne
# Plot dataen