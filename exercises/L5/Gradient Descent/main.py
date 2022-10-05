from utilsModule import *
import matplotlib.pyplot as plt
import numpy as np

X, X_b, y = GenerateData()

np.random.seed(42)

theta = np.random.randn(2,1)  

plotMoreFigs(X, X_b, y, theta, 50 , len(X_b))