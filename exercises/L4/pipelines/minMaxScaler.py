def loadExerciseDataAsXY():
    print("Loading data...")
    import pickle
    filename = "Data/itmal_l01_data.pkl"
    with open(f"{filename}", "rb") as f:
        (X, y) = pickle.load(f)
        print("Done!")
        return X, y

def dummyScaler(X_pre):
    import numpy as np
    # Inspiration for scaler "algorithm": https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    X_scaled = (X_pre-min(X_pre))/(max(X_pre)-min(X_pre))
    return X_scaled
