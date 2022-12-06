import keras.api._v2.keras as keras
from keras import layers
import numpy as np

from modeltraining import Outputs, outputEnumNumberConvert

def generateFakeLabels(size):
    y= []
    for i in range(0, size):
        y.append(np.random.randint(0, 3))
    return y


def createSequentialModel(
    inputSize=3, 
    outputSize=3, 
    hiddenlayerSize = 10, 
    hiddenlayerAmount = 10,
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(),
    metrics = ["accuracy"]
):
    print("Creating model...")
    model = keras.Sequential()

    model.add(keras.layers.Dense(inputSize, input_shape=(3,)))
    model.add(keras.layers.BatchNormalization())

    for _ in range(0, hiddenlayerAmount):
        model.add(keras.layers.Dense(hiddenlayerSize))

    model.add(keras.layers.Dense(outputSize))
    print("Done!\n")

    print("Compiling model...")
    model.compile(
        loss = loss, 
        optimizer = optimizer, 
        metrics = metrics)
    print("Done!\n")

    return model

def doGridSearchCVWithSequentialModel(X_train, y_train):
    from scikeras.wrappers import KerasClassifier
    from sklearn.model_selection import GridSearchCV

    model = KerasClassifier(model=createModel, loss="binary_crossentropy", epochs=50)
    
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    
    neurons = [10, 25, 50, 100, 200, 500]
    hidden_layer_size= [10, 15, 20]

    activation = ['softmax', 'softplus', 'relu', 'tanh', 'sigmoid']
    optimizer = ['SGD', 'Adam', 'Adamax']
    
    param_grid = dict(
        optimizer__learning_rate = learn_rate, 
        model__activation = activation,
        model__hidden_layer_size=hidden_layer_size,
        model__neurons=neurons,
        optimizer = optimizer
    )
 
    grid = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        n_jobs=-1,
        cv=5
    )
    
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def createModel(activation='relu', neurons=10, hidden_layer_size=10):
    model = keras.Sequential()

    model.add(keras.layers.Dense(3, input_shape=(3,)))
    model.add(keras.layers.BatchNormalization())

    for _ in range(0, hidden_layer_size):
        model.add(keras.layers.Dense(neurons, activation=activation))

    model.add(keras.layers.Dense(3))

    return model