import numpy as np 

# 1)
A = np.zeros((9,6))
# set the ones in three seperate blocks
A[0:2, 1:5] = 1
A[2:7, 2:4] = 1
A[7:9, 1:5] = 1
print("A:\n", A, "\n")

B = np.zeros((11,6))
B[1:10, :] = A
print("B:\n", B, "\n")

# 2)
C = np.arange(66).reshape(11,6) + 1
D = B * C
print("C:\n", C, '\nD:\n', D, '\n')

E = np.array([x for x in D.reshape(-1) if x > 0])

# normalize E
max, min = E.max(), E.min()
F = (E - min) / (max - min)
print("E:\n", E, "\nF:\n", F, '\n')

# find the index by performing elementwise subtraction and finding the min difference
closest_val = F[(np.abs(F - .25)).argmin()]
print("Closest Value:", closest_val)