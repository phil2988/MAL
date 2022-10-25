def plotModel(dataX, dataY, X, y, label = "No Label"):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(X))
    for i in range(len(X)):
        axs[i].plot(dataX, dataY)
        axs[i].plot(X[i], y[i])
        axs[i].set_title([label[i]])
        axs[i].set_xlabel("GDP per capita")
        axs[i].set_ylabel("Life satisfaction")
    fig.tight_layout(pad=1)
    plt.show()
