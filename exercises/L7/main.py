from helpers import getMnistDataSet, Dataloader, ResidualUnit, DefaultConv2D, compileFitAndPredict
import numpy as np
import keras.api._v2.keras as keras

print("=========================================================")
dl = Dataloader('dataset/train', 'dataset/test')

(X_train, y_train), (X_test, y_test) = dl.get()

print("=========================================================")

print("Creating model...")    

model = keras.models.Sequential()

model.add(DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[28, 28, 1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))

prev_filters = 64
for filters in [64*4] :
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))


print("Model created!")
print("=========================================================")

compileFitAndPredict(X_train, y_train, X_test, y_test, model=model, optimizer="adam")

compileFitAndPredict(X_train, y_train, X_test, y_test, model=model, optimizer="SGD")

compileFitAndPredict(X_train, y_train, X_test, y_test, model=model, optimizer="Adadelta")
