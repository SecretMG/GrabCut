import numpy as np

matrixSize = 2
A = np.random.randint(0, 2, (matrixSize, matrixSize))
B = np.dot(A, A.transpose())
C = B+B.T   # makesure symmetric
# test whether C is definite
D = np.linalg.cholesky(C)   # if there is no error, C is definite
print(C)