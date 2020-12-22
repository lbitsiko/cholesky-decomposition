import numpy as np

from cholesky_functions import solve_cholesky

A = np.zeros((500,500))

d = np.float64(np.arange(1001,1502))
e = np.float64(np.arange(3,502))

n = len(A)
A[0,0] = d[0]
for i in range(1,n):
    A[i,i] = d[i]
    A[i,i-1] = e[i-1]
    A[i-1,i] = e[i-1]

b = np.arange(2, 502)


my_solution = solve_cholesky(A,b)
np_solution = np.linalg.solve(A,b)
print(np.allclose(my_solution, np_solution))