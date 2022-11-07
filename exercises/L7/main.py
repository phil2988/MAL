from helpers import getMnistDataSet, Dataloader, ResidualUnit, DefaultConv2D
import numpy as np
import keras.api._v2.keras as keras

print("=========================================================")

(X_train, y_train), (X_test, y_test) = Dataloader('dataset/train', 'dataset/test').get()

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
print("Compiling model...")
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
print("Model compiled!")
print("=========================================================")
print("Fitting model...")
hist = model.fit(X_train, y_train, epochs=5, batch_size=100, validation_data=(X_test, y_test))
print("Model fitted!")
print("=========================================================")
print("Predicting with model...")
predictions = model.predict(X_test)
print("Done!")
print("=========================================================")
print('|FIT RESULTS|')
print('Fit loss: ', hist.history['loss'] )
print('Fit acc: ', hist.history['accuracy'] )
print('Fit val_loss: ', hist.history['val_loss'] )
print('Fit val_acc: ', hist.history['val_accuracy'] )
print("=========================================================")
print("Testing model....")

testIndexes = [142, 1200, 2412, 6623, 9552, 5522, 144, 402, 1422]

for i in range(len(testIndexes)):
    highest = np.argmax(predictions[testIndexes[i]])
    print(f"Prediction: {highest}. Actual value: {y_test[testIndexes[i]]}")

print("=========================================================")
    