import numpy as np
matrixSize = 2
A = np.random.randint(0, 2, (matrixSize, matrixSize))
B = np.dot(A, A.transpose())
print(B)    # 生成一个随机矩阵，乘以它的转置