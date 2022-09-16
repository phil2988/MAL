from rootMeanSquareError import *
from euclidianDistances import *
from meanAbsoluteError import *
from math import fabs
import numpy as np

tx=np.array([1, 2, 3, -1])
ty=np.array([3,-1, 4,  1])

expected_d1=8.0
expected_d2=4.242640687119285

d1=L1(tx-ty)
d2=L2(tx-ty)

print(f"tx - ty = {tx-ty} | d1 - expected_d1 = {d1-expected_d1} | d2 - expected_d2 = {d2-expected_d2}")

eps=1E-9

assert fabs(d1-expected_d1)<eps, "L1 dist seems to be wrong" 
assert fabs(d2-expected_d2)<eps, "L2 dist seems to be wrong" 

print("OK(part-1)")

d2dot=L2Dot(tx-ty)
print("d2dot-expected_d2=",d2dot-expected_d2)
assert fabs(d2dot-expected_d2)<eps, "L2Ddot dist seem to be wrong" 
print("OK(part-2)")


X = np.array([
    [1, 2, 3],
    [4, 2, 1],
    [3, 8, 5],
    [-9, -1, 0]
]
)

y = np.array(
    [1,2,3,4]
)

print(f"X is {X}")
print(f"y is {y}")
# Dummy h function:
def h(X):    
    if X.ndim!=2:
        raise ValueError("excpeted X to be of ndim=2, got ndim=",X.ndim)
    if X.shape[0]==0 or X.shape[1]==0:
        raise ValueError("X got zero data along the 0/1 axis, cannot continue")
    return X[:,0]

# # Calls your RMSE() function:
r=RMSE(h(X),y)

# # TEST vector:
eps=1E-9
expected=6.57647321898295
print(f"RMSE={r}, diff={r-expected}")
assert fabs(r-expected)<eps, "your RMSE dist seems to be wrong" 

print("OK")


# Calls your MAE function:
r=MAE(h(X), y)

# TEST vector:
expected=3.75
print(f"MAE={r}, diff={r-expected}")
assert fabs(r-expected)<eps, "MAE dist seems to be wrong" 

print("OK")