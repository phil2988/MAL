from helpers import getMnistDataSet, Dataloader
import numpy as np

dataloader = Dataloader('dataset/train', 'dataset/test')
dataloader.save()

(X_train, y_train), (X_Test, y_test) = dataloader.get()
print()

classes = np.unique(y_train)
print("Unique classes for dataset: ", classes)
    
