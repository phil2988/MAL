def GenerateData():
    import numpy as np
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    return X, X_b, y

def plot_gradient_descent(X, y, X_b, X_new, X_new_b, theta, eta, theta_path=None):
    import matplotlib.pyplot as plt
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

def plotFigs(X, y, X_b):
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    theta_path_bgd = []
    theta = np.random.randn(2,1)  # random initialization

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    plt.figure(figsize=(10,4))
    plt.subplot(131); plot_gradient_descent(X, y, X_b, X_new, X_new_b, theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132); plot_gradient_descent(X, y, X_b, X_new, X_new_b, theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133); plot_gradient_descent(X, y, X_b, X_new, X_new_b, theta, eta=0.5)
    plt.show()

def learning_schedule(t):
    t0, t1 = 5, 50  # learning schedule hyperparameters
    return t0 / (t + t1)

def plotMoreFigs(X, X_b, y, theta, n_epochs, m):
    from sklearn.linear_model import SGDRegressor
    import matplotlib.pyplot as plt
    import numpy as np

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    sthocasticGradientDescent(X, y, X_b, X_new, X_new_b, theta, n_epochs, m)

    plt.xlabel("$x_1$", fontsize=18)                     
    plt.ylabel("$y$", rotation=0, fontsize=18)           
    plt.axis([0, 2, 0, 15])                              

    sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
    sgd_reg.fit(X, y.ravel())
    print(f'stochastic gradient descent theta={theta.ravel()}')
    print(f'Scikit-learn SGDRegressor "thetas": sgd_reg.intercept_={sgd_reg.intercept_}, sgd_reg.coef_={sgd_reg.coef_}')
    
    plt.show()        

def sthocasticGradientDescent(X, y, X_b, X_new, X_new_b, theta, n_epochs, m):
    import matplotlib.pyplot as plt
    import numpy as np
    theta_path_sgd = []

    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:
                y_predict = X_new_b.dot(theta) 
                style = "b-" if i > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)        
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
            theta_path_sgd.append(theta)                 

            plt.plot(X, y, "b.")  