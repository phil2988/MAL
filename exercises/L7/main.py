from sklearn.model_selection import train_test_split
from utils import MNIST_GetDataSetXy, SaverMNIST
import numpy as np
# import tensorflow as tf
# from tensorflow import keras

mnist_saver = SaverMNIST(image_train_path='dataset/train', image_test_path='dataset/test', csv_train_path='dataset/train.csv', csv_test_path='dataset/test.csv')

# Write files into disk
mnist_saver.run()

# print("Fetching data...")
# X, y = MNIST_GetDataSetXy()
# print("Done!")

# print("Splitting data into test and train")
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print("Done!")

# classes = np.unique(y_train)
# print("Unique classes for dataset: ", classes)
    
