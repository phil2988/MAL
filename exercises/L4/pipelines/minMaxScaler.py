from minMaxScaler import *

def loadExerciseDataAsXY():
    print("Loading data...")
    import pickle
    filename = "Data/itmal_l01_data.pkl"
    with open(f"{filename}", "rb") as f:
        (X, y) = pickle.load(f)
        print("Done!")
        return X, y

def dummyScaler(X_pre, y_pre):
    import numpy as np
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    X_scaled = (X_pre-min(X_pre))/(max(X_pre)-min(X_pre))
    y_scaled = (y_pre-min(y_pre))/(max(y_pre)-min(y_pre))
    return X_scaled, y_scaled
