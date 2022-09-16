def MNIST_GetDataSet():
    from sklearn.datasets import fetch_openml

    return fetch_openml('mnist_784', return_X_y=True, version=1, cache=False) 

def MNIST_PlotDigit(data):
    import matplotlib
    import matplotlib.pyplot as plt
    
    image = data.values.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

X, y = MNIST_GetDataSet()

row = X[0:1]

MNIST_PlotDigit(row)
