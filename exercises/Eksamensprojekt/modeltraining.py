from enum import IntEnum, auto
from strenum import StrEnum

class Outputs(StrEnum):
    control = auto()
    aggro = auto()
    tempo = auto()

def outputStringNumberConvert(output):
    if(type(output) == Outputs):
        if(output == "control"):
            return 0
        if(output == "aggro"):
            return 1
        if(output == "tempo"):
            return 2
    if(type(output) == int):
        if(output == 0):
            return "control"
        if(output == 1):
            return "aggro"
        if(output == 2):
            return "tempo"

def trainModel(model, X_train, y_train, X_test, y_test):
    hist = model.fit(
        X_train, 
        y_train, 
        epochs=10, 
        shuffle=True,
        validation_data=(X_test, y_test), 
        verbose=1, 
    )
    return model, hist


def printTrainingResults(model, X, y):
    print("Evaluating the model...")
    history = model.evaluate(X, y)
    print("===================================")
    print("Best Accuracy: ", history[1])
    print("Best Loss: ", history[0])
    print("===================================")

    
