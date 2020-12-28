import numpy as np

def lu_decomp(A):
    Ls = []
    Us = []
    n = len(A)

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
    L = spdiags(diags_forL, positions_of_diags_forL, 100, 100)

    #array Us to diagonal matrix U
    diags_forU = np.array([Us, np.ones(100)])
    positions_of_diags_forU = np.array([1, 0])
    U = spdiags(diags_forU, positions_of_diags_forU, 100, 100)

    return L, U

d = np.float64(np.arange(1001,1101))
es = np.float64(np.arange(3, 103))
fs = np.float64(np.arange(2, 102))


from scipy.sparse import spdiags
diagonal_elements = np.array([d, es, fs])
diagonal_positions = np.array([0,-1,1])
A = spdiags(diagonal_elements, diagonal_positions, 101, 101).toarray()
A= A[:100,:100]
L, U = lu_decomp(A)

# import  scipy.linalg
# a1, a2,a3 = scipy.linalg.lu(A)

b = np.arange(2, 102)

def solve_LU(A,b):
    L, U = lu_decomp(A)

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

solution = solve_LU(A,b)
np_solution = np.linalg.solve(A,b)

print(np.allclose(solution, np_solution, rtol=1e-04 ))
# import  matplotlib.pyplot as plt
# plt.plot(solution,'o')
# plt.plot(np_solution,'o')
# plt.show()
