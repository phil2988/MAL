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
    metrics = ["accuracy"],
    neuronActivationFunc = "tanh"
):
    print("Creating model...")
    model = keras.Sequential()

    model.add(keras.layers.Dense(inputSize, input_shape=(3,)))
    model.add(keras.layers.BatchNormalization())

    for _ in range(0, hiddenlayerAmount):
        model.add(keras.layers.Dense(hiddenlayerSize, activation=neuronActivationFunc))

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
    from sklearn.model_selection import RandomizedSearchCV

    model = KerasClassifier(model=createModel, epochs=50)
    
    learn_rate = []
    for i in range(1, 5000, 10):
        learn_rate.append(i/1000)

    neurons = range(10, 500)
    hidden_layer_size= range(5, 50)

    activation = [
        'softmax', 
        'softplus', 
        'relu', 
        'tanh', 
        'sigmoid'
    ]
    
    optimizer = [
        'SGD', 
        'Adam', 
        'Adamax',
    ]

    loss = [
        "binary_crossentropy", 
        "categorical_crossentropy", 
        "sparse_categorical_crossentropy", 
        "poisson", 
        "kl_divergence",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_logarithmic_error",
        "cosine_similarity",
        "huber_loss",
        "log_cosh"
    ]
    
    param_grid = dict(
        optimizer__learning_rate = learn_rate, 
        model__activation = activation,
        model__hidden_layer_size=hidden_layer_size,
        model__neurons=neurons,
        optimizer = optimizer,
        loss = loss
    )
 
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid, 
        n_iter=200,
        n_jobs=5,
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