# import numpy as np

# X = np.array([
#     [1, 2, 3], 
#     [4, 2, 1], 
#     [3, 8, 5], 
#     [-9, -1, 0]
# ])

# print(X)

import numpy as np
import euclidianDistances as eds

# TEST vectors: here I test your implementation...calling your L1() and L2() functions
tx=np.array([1, 2, 3, -1])
ty=np.array([3,-1, 4,  1])

expected_d1=8.0
expected_d2=4.242640687119285

d1=eds.L1(tx-ty)
# d2=L2(tx-ty)

print(f"tx-ty={tx-ty}, d1-expected_d1={d1-expected_d1}")

# eps=1E-9 
# # NOTE: remember to import 'math' for fabs for the next two lines..
# assert fabs(d1-expected_d1)<eps, "L1 dist seems to be wrong" 
# assert fabs(d2-expected_d2)<eps, "L2 dist seems to be wrong" 

print("OK(part-1)")
