from math import sqrt
import numpy as np
from euclidianDistances import L2
def RMSE(X, y):
    print(f"X is {X}")
    print(f"y is {y}")
    l2 = L2(X-np.transpose(y))
    print(f"L2 is {l2}")
    
    return 0