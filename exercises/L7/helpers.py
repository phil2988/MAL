import csv
from keras.datasets import mnist
from PIL import Image
import os
import numpy as np
import threading 
import keras.api._v2.keras as keras
from functools import partial
import matplotlib.pyplot as plt



def getMnistDataSet():
    return mnist.load_data()

def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

def make_containing_dirs(path_list):
    for path in path_list:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

class Dataloader():
    def __init__(self, train_path, test_path):
        
        self._image_format = '.png'
        self.paths = [train_path, test_path]
             
        # Load MNIST dataset
        file_dir_name = str(self.paths[0] + "/0.png")
        file = os.path.exists(file_dir_name)
        
        X_train = []
        y_train = []  
        X_test = []
        y_test = []

        if(not file):
            def download():
                self.data = mnist.load_data() 

            print("DATALOADER: Did not find local data. Downloading it now...")
            mnistThread = threading.Thread(download())
            mnistThread.start()
            mnistThread.join()            

            print("DATALOADER: Done!")
        else:
            print("DATALOADER: Files found in dataset folder. Skipping download")
            make_dirs(self.paths)
            def tf1():
                print("Reading data from: ", self.paths[0])
                for _, __, files in os.walk(self.paths[0]):
                    for file in files:
                        X_train.append(np.array(Image.open(self.paths[0] + "/" + file)))
                print("DATALOADER: Done reading training data")
            def tf2():
                print("DATALOADER: Reading data from: ", self.paths[1])
                for _, __, files in os.walk(self.paths[1]):
                    for file in files:
                        X_test.append(np.array(Image.open(self.paths[1] + "/" + file)))
                print("DATALOADER: Done reading testing data")

            def tf3():
                with open("dataset/labels_train.csv", "r+") as file:
                    for row in csv.reader(file, delimiter=" "):
                        y_train.extend(row)
                with open("dataset/labels_test.csv", "r+") as file:
                    for row in csv.reader(file, delimiter=" "):
                        y_test.extend(row)
                print("DATALOADER: Done reading labels")

            threads = [
                threading.Thread(tf3()),
                threading.Thread(tf2()), 
                threading.Thread(tf1()), 
            ]

            print("DATALOADER: Starting reading of data...")

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            print("DATALOADER: Done reading!")
            
            assert(len(X_train) == len(y_train) and len(X_train) == len(y_train))
            print("DATALOADER: Loading data into dataloader...")
            self.data = [
                [
                    X_train,
                    y_train
                ], 
                [     
                    X_test,
                    y_test
                ]
            ]

    def save(self):
        # Check if already saved
        file_dir_name = str(self.paths[0] + "/0.png")
        file = os.path.exists(file_dir_name)
        if(not file):
            train, test = self.data
            (data_train, labels_train), (data_test, labels_test) = train, test

            dataToWrite = [
                data_train, 
                data_test, 
                labels_train, 
                labels_test
            ]

            assert(len(data_test) == len(labels_test) and len(data_train) == len(labels_train))

            if(not(os.path.exists(self.paths[0]) and os.path.exists(self.paths[1]))):
                make_dirs(self.paths)
            def tf1():
                print("DATALOADER: Writing data to: ", self.paths[0])
                for index in range(len(dataToWrite[0])):
                    img = Image.fromarray(dataToWrite[0][index])
                    img_name =  self.paths[0] + "/" + str(index) + ".png"
                    img.save( img_name)
                print("DATALOADER: Done writing data to: ", self.paths[1])

            def tf2():
                print("Writing data to: ", self.paths[1])
                for index in range(len(dataToWrite[1])):
                    img = Image.fromarray(dataToWrite[1][index])
                    img_name =  self.paths[1] + "/" + str(index) + ".png"
                    img.save( img_name)
                print("DATALOADER: Done writing data to: ", self.paths[1])
                
            def tf3():
                with open('dataset/labels_train.csv', 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(dataToWrite[2])
                with open('dataset/labels_test.csv', 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(dataToWrite[3])
                print("DATALOADER: Done writing labels")
            
            threads = [
                threading.Thread(tf3()),
                threading.Thread(tf2()),
                threading.Thread(tf1()), 
            ]
            print("DATALOADER: Starting writing to files...")

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            print("DATALOADER: Done!")

        else:
            print("DATALOADER: Found existing data, no need to save again!")
    def get(self):
        return self.data

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def compileFitAndPredict(X_train, y_train, X_test, y_test, model: keras.models.Sequential, loss, optimizer, epochs=15):
    print("Compiling model...")
    model.compile(loss=loss,  optimizer=optimizer, metrics=["accuracy"])
    print("Model compiled!")
    print("=========================================================")
    print("Fitting model...")
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=100, validation_data=(X_test, y_test))
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


    plt.plot(range(0, epochs), hist.history['accuracy'], label="Accuracy")
    plt.plot(range(0, epochs), hist.history['val_accuracy'], label="validation set accuracy")

    # plt.plot(range(0, epochs), hist.history['accuracy'], label= loss + " - accuracy")
    # plt.plot(range(0, epochs), hist.history['val_accuracy'], label= loss + " - val_accuracy")

