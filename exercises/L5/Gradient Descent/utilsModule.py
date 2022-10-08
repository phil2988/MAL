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

def plotMySGDRegressor(X, y, X_b, X_new, X_new_b, theta, n_epochs, m):
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
    print(f'stochastic theta={theta.ravel()}')
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show() 

def plotMyMBSGDRegressor(X, y, X_b, theta, n_iterations, minibatch_size, m):
    import matplotlib.pyplot as plt
    import numpy as np

    theta_path_mgd = []
    
    X_new = np . array ([[0] , [2]])
    X_new_b = np . c_ [ np . ones ((2 , 1) ) , X_new ]

    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
    
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, X_new_b.dot(theta), style)
    plt.plot(X, y, "b.")
    print(f'mini-batch theta={theta.ravel()}')
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show() 

def plotAllThetaPaths():
    import matplotlib.pyplot as plt
    import numpy as np

    X, X_b, y = GenerateData()

    m = len(X_b)

    np.random.seed(42)
    theta1 = np.random.randn(2,1)  # random initialization
    theta2 = np.random.randn(2,1)  # random initialization
    theta3 = np.random.randn(2,1)  # random initialization

    eta = 0.1
    n_iterations = 50
    minibatch_size = 20
    n_epochs = 50

    theta_path_bgd = []
    theta_path_mgd = []
    theta_path_sgd = []

    X_new = np . array ([[0] , [2]])
    X_new_b = np . c_ [ np . ones ((2 , 1) ) , X_new ]
    
    n_iterations = 1000
    for iteration in range(n_iterations):
        # if iteration < 10:
        #     y_predict = X_new_b.dot(theta)
        
        gradients = 2/m * X_b.T.dot(X_b.dot(theta1) - y)
        theta1 = theta1 - eta * gradients
        if theta_path_bgd is not None:
            theta_path_bgd.append(theta1)

    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta2) - yi)
            eta = learning_schedule(t)
            theta2 = theta2 - eta * gradients
            theta_path_mgd.append(theta2)
    

    for epoch in range(n_epochs):
        for i in range(m):
            # if epoch == 0 and i < 20:
            #     y_predict = X_new_b.dot(theta3) 
            
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            gradients = 2 * xi.T.dot(xi.dot(theta3) - yi)        
            eta = learning_schedule(epoch * m + i)
            theta3 = theta3 - eta * gradients
            theta_path_sgd.append(theta3)       
    
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)

    plt.figure(figsize=(7,4))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    plt.show()
    print('OK')