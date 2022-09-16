import numpy as np

def L1(vector) :
    assert vector.ndim == 1, f"Expected a vector, got {vector.ndim}d array"
    print(f"Vector to do L1 on is {vector}")
    sumVal = 0
    for i in range(len(vector)):
        if(vector[i] < 0):
            sumVal += -vector[i]
        else:
            sumVal += vector[i]
    return sumVal

def L2(vector):
    assert vector.ndim == 1, f"Expected a vector, got {vector.ndim}d array"
    print(f"Vector to do L2 on is {vector}")
    sumVal = 0
    for i in range(len(vector)):
        if(vector[i] < 0):
            sumVal += (-vector[i]) ** 2
        else:
            sumVal += vector[i] ** 2
    sumVal = sumVal ** 0.5
    return sumVal

import numpy as np

def L2Dot(vector):
    sumVal = 0
    vectorTransposed = np.transpose(vector)
    sumVal = np.dot(vectorTransposed, vector) ** 0.5
    return sumVal