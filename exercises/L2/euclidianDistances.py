from xml.etree.ElementInclude import include

import numpy as np

def L1(vector) :
    sumVal = 0

    for i in vector:
        sumVal += vector[i]
    return sumVal