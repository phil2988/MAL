# from sklearn.model_selection import train_test_split
from helpers import MNIST_GetDataSetXy, SaverMNIST
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

mnist_saver = SaverMNIST('dataset/train', 'dataset/test')
mnist_saver.save()
data = mnist_saver.get()
print()

# print("Fetching data...")
# X, y = MNIST_GetDataSetXy()
# print("Done!")

# print("Splitting data into test and train")
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print("Done!")

# classes = np.unique(y_train)
# print("Unique classes for dataset: ", classes)
    
