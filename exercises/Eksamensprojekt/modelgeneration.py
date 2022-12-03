import keras.api._v2.keras as keras
from keras import layers
import numpy as np

from modeltraining import Outputs, outputEnumNumberConvert

def generateFakeLabels(size):
    y= []
    for i in range(0, size):
        y.append(np.random.randint(0, 3))
    return y


def createModel():
    print("Creating model...")
    model = keras.Sequential()

    model.add(keras.layers.Dense(3))
    model.add(keras.layers.BatchNormalization())

    for i in range(0, 20):
        model.add(keras.layers.Dense(1000))

    model.add(keras.layers.Dense(3))
    print("Done!\n")

    print("Compiling model...")
    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(), 
        optimizer = keras.optimizers.Adam(learning_rate=1e-3), 
        metrics=["accuracy"])
    print("Done!\n")

    return model
