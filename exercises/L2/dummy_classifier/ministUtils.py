def MNIST_GetDataSet():
    from sklearn.datasets import fetch_openml
    print("Fetching Data...")
    return fetch_openml('mnist_784', return_X_y=True, version=1, cache=False) 

def MNIST_PlotDigit(data):
    import matplotlib
    import matplotlib.pyplot as plt
    
    print("Plotting data...")
    image = data.values.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
