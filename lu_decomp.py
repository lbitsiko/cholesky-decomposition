import numpy as np

# decompose a tridiagonal matrix into its LU decomposition (A=LU)
def decomp_LU_tridiag(A):
    Ls = []
    Us = []
    n = A.shape[0]

    ds = A.diagonal()
    es = A.diagonal(-1)
    fs = A.diagonal(1)

    # first l1, u1 = f1/l1 (in python start with 0)
    Ls.append(ds[0])
    Us.append(fs[0]/Ls[0])
    for i in range(1, n-1): # i=2~n-1  (in python i=1~n-2)
        Ls.append(ds[i]-es[i]*Us[i-1])
        Us.append(fs[i]/Ls[i])
    Ls.append(ds[n-1] - es[n-2] * Us[n-2])
    Us = np.insert(Us,0,0.0) # necessary for spdiags

    Ls = np.array(Ls)
    Us = np.array(Us)

    es = np.insert(es, len(es), 0.0) #necessary for spdiags

    #array Ls to diagonal matrix L
    diags_forL = np.array([Ls, es])
    positions_of_diags_forL = np.array([0, -1])
    L = spdiags(diags_forL, positions_of_diags_forL, n, n)

    #array Us to diagonal matrix U
    diags_forU = np.array([Us, np.ones(n)])
    positions_of_diags_forU = np.array([1, 0])
    U = spdiags(diags_forU, positions_of_diags_forU, n, n)

    return L, U

#store matrix A as a tridiagonal matrix
d = np.float64(np.arange(1001,1101)) #principal diagonal elements, d_i
es = np.float64(np.arange(3, 103)) #lower diagonal elements, e_i
fs = np.float64(np.arange(2, 102)) #upper diagonal elements, f_i = e_i

from scipy.sparse import spdiags
diagonal_elements = np.array([d, es, fs])
diagonal_positions = np.array([0,-1,1])
A = spdiags(diagonal_elements, diagonal_positions, 100, 100)

#solve AX=b where A is a triadiagonal matrix
def solve_tridiag_LU(A,b):
    L, U = decomp_LU_tridiag(A)

    Ls = L.diagonal()
    Us = U.diagonal(1)
    es = L.diagonal(-1)

    y = [b[0]/Ls[0]]
    n=len(b)
    for i in range(1,n):
        y.append((b[i]-es[i-1]*y[i-1])/Ls[i])
    y = np.array(y)

    x = [y[len(y)-1]]
    count = 0
    for i in range(n-2,-1,-1):
        x.append(y[i] - Us[i]*x[count])
        count += 1
    return np.array(x[::-1])

# Solve Ax=b
b = np.arange(2, 102)
solution = solve_tridiag_LU(A, b)
np_solution = np.linalg.solve(A.toarray(), b)

print(np.allclose(solution, np_solution, rtol=1e-04))

import matplotlib.pyplot as plt
plt.close('all')
# plt.rcParams['figure.figsize'] = [10, 6]
# plt.rcParams['figure.dpi'] = 100
fig = plt.figure()
ax = plt.axes()
plt.grid(ls='--')
plt.plot(solution,'.',label='solution')
plt.plot(np_solution, '--',label = 'numpy solution')
plt.legend()
ax.set_xlabel("i")
ax.set_ylabel("$x_{i}$")
plt.show()