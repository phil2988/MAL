from helpers import getMnistDataSet, Dataloader

dataloader = Dataloader('dataset/train', 'dataset/test', getMnistDataSet)
dataloader.save()
(X_train, y_train), (X_Test, y_test) = dataloader.get()
print()

# print("Fetching data...")
# X, y = MNIST_GetDataSetXy()
# print("Done!")

# print("Splitting data into test and train")
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print("Done!")

# classes = np.unique(y_train)
# print("Unique classes for dataset: ", classes)
    
