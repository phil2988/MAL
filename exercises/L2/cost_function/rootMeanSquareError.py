from math import sqrt
import numpy as np
from euclidianDistances import L2

def RMSE(y_pred, y_true):
    assert y_pred.ndim == 1, f"Expected a vector, y_pred is a {y_pred.ndim}d array"
    assert y_true.ndim == 1, f"Expected a vector, y_true is a {y_true.ndim}d array"
    assert len(y_true) == len(y_pred), f"Vectors are not same length. y_true is {len(y_true)} and y_pred is {len(y_pred)}"

    print(f"y_pred is {y_pred}")
    print(f"y_true is {y_true}")

    l2 = L2(y_true-y_pred)
    print(f"L2 is {l2}")
    
    squared = l2 ** 2
    print(f"Squared is {squared}")

    mean = squared/len(y_pred)
    print(f"Mean is {mean}")
    
    squared = sqrt(mean)
    print(f"Squared is {squared}")
    
    return squared