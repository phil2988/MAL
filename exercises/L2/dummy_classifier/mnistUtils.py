def MNIST_GetDataSetXy():
    from sklearn.datasets import fetch_openml
    print("Fetching Data...")
    return fetch_openml('mnist_784', return_X_y=True, cache=False) 

def MNIST_PlotDigit(data):
    import matplotlib
    import matplotlib.pyplot as plt
    
    print("Plotting data...")
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def MNIST_CreateSGDClassifier(X_train, y_train):
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)

    print("Fitting data...")
    return sgd_clf.fit(X_train, y_train)