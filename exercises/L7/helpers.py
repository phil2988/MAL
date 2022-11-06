import csv
from keras.datasets import mnist
from sklearn.datasets import fetch_openml
from importlib.resources import path
import tensorflow
from PIL import Image
import pandas as pd
import os
import numpy as np
import threading 

def MNIST_GetDataSetXy():
    print("Fetching Data...")
    return fetch_openml('mnist_784', return_X_y=True, cache=False) 

def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

def make_containing_dirs(path_list):
    for path in path_list:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

class SaverMNIST():
    def __init__(self, train_path, test_path):
        
        self._image_format = '.png'
        self.paths = [train_path, test_path]
             
        # Load MNIST dataset
        file_dir_name = str(self.paths[0] + "/0.png")
        file = os.path.exists(file_dir_name)
        if(not file):
            print("Did not find data. Downloading it now...")
            def threadDownload():
                self.data = mnist.load_data()

            mnistThead = threading.Thread(threadDownload())
            mnistThead.start()
            mnistThead.join()            

            print("Done!")
        else: 
            print("Files found in dataset folder. Skipping download")

            X_train = []
            y_train = []  
            X_test = []
            y_test = []
            
            print("Reading data from pc...")

            def tf1(): 
                for _, __, files in os.walk(self.paths[0]):
                    for file in files:
                        X_train.append(np.array(Image.open(self.paths[0] + "/" + file)))
                print("Done reading training data")
            def tf2():
                for _, __, files in os.walk(self.paths[1]):
                    for file in files:
                        X_test.append(np.array(Image.open(self.paths[1] + "/" + file)))
                print("Done reading testing data")

            def tf3():
                with open("dataset/labels_train.csv") as file:
                    for row in csv.reader(file, delimiter=" "):
                        y_train.extend(row)
                with open("dataset/labels_test.csv") as file:
                    for row in csv.reader(file, delimiter=" "):
                        y_test.extend(row)
                print("Done reading labels")

            threads = [
                threading.Thread(tf1()), 
                threading.Thread(tf2()), 
                threading.Thread(tf3())
            ]

            print("Starting threads to read data...")
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    
            print("Done!")
        
        assert(len(X_train) == len(y_train) and len(X_train) == len(y_train))
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
        # Load MNIST dataset
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

            print("Writing data to files...")
            if(not(os.path.exists(self.paths[0]) and os.path.exists(self.paths[1]))):
                make_dirs(self.paths)
            for i in range(len(self.paths)):
                print("Writing data to: ", self.paths[i])
                for index in range(len(dataToWrite[i])):
                    img = Image.fromarray(dataToWrite[i][index])
                    img_name =  self.paths[i] + str(index) + ".png"
                    img.save( img_name)
                print("Done!")
            print("Writing csv files for labels...")
            with open('dataset/labels_train.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(dataToWrite[2])
            with open('dataset/labels_test.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(dataToWrite[3])
            print("Done!")

        else:
            print("Found existing files... skipping save!")
    def get(self):
        return self.data
