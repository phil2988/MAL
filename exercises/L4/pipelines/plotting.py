def plotModel(X, y, label = "No Label"):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(X))

    for i in range(len(X)):
        axs[i].plot(X[i], y[i])
        axs.set_title([label[i]])
 
    # axs.xlabel("GDP per capita")
    # axs.ylabel("Life satisfaction")
    plt.show()
# fig.subTitle("Models plotted side by side")