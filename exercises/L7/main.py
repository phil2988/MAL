from helpers import getMnistDataSet, Dataloader, ResidualUnit, DefaultConv2D, compileFitAndPredict
import numpy as np
import keras.api._v2.keras as keras
import matplotlib.pyplot as plt
import keras

print("=========================================================")
dl = Dataloader('dataset/train', 'dataset/test')

(X_train, y_train), (X_test, y_test) = dl.get()

print("=========================================================")

print("Creating model...")    

model = keras.models.Sequential()

model.add(DefaultConv2D(10, kernel_size=7, strides=2, input_shape=[28, 28, 1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))

prev_filters = 10
for filters in [10] :
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()
compileFitAndPredict(X_train, y_train, X_test, y_test, model, optimizer="adam", loss="sparse_categorical_crossentropy", epochs=100)

model.save("final")

print("Model created and saved!")
print("=========================================================")

# model1 = keras.models.load_model("model")
# compileFitAndPredict(X_train, y_train, X_test, y_test, model1, optimizer="adam", loss="sparse_categorical_crossentropy")

# model2 = keras.models.load_model("model")
# compileFitAndPredict(X_train, y_train, X_test, y_test, model2, optimizer="adadelta", loss="sparse_categorical_crossentropy")

# model3 = keras.models.load_model("model")
# compileFitAndPredict(X_train, y_train, X_test, y_test, model3, optimizer="sgd", loss="sparse_categorical_crossentropy")

# model1 = keras.models.load_model("model")
# compileFitAndPredict(X_train, y_train, X_test, y_test, model1, optimizer="adam", loss="sparse_categorical_crossentropy")

# model2 = keras.models.load_model("model")
# compileFitAndPredict(X_train, y_train, X_test, y_test, model2, optimizer="adam", loss="mean_squared_error")

# model3 = keras.models.load_model("model")
# compileFitAndPredict(X_train, y_train, X_test, y_test, model3, optimizer="adam", loss="poisson")

plt.title("Learning curve")
plt.ylabel("Accuracy")
plt.xlabel("Epoc")
plt.legend()
plt.show()