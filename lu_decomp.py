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

    return np.array(Ls), np.array(Us)

# # d = np.float64(np.arange(1001,1102))
# e1 = np.float64(np.arange(3,104))
# e2 = np.float64(np.arange(2,103))

d = np.float64(np.arange(1001,1101))
es = np.float64(np.arange(3, 103))
fs = np.float64(np.arange(2, 102))
# b = np.arange(2, 502)

from scipy.sparse import spdiags
diagonal_elements = np.array([d, es, fs])
diagonal_positions = np.array([0,-1,1])
A = spdiags(diagonal_elements, diagonal_positions, 101, 101).toarray()
A= A[:100,:100]
Ls,Us = lu_decomp(A)

diags_forL = np.array([Ls, es])
positions_of_diags_forL = np.array([0, -1])
L = spdiags(diags_forL, positions_of_diags_forL, 100, 100)  # .toarray()

diags_forU = np.array([Us, np.ones(100)])
positions_of_diags_forU = np.array([1, 0])
U = spdiags(diags_forU, positions_of_diags_forU, 100, 100).toarray()


es_lu_method = A.diagonal(-1)
fs_lu_method = A.diagonal(1)

es_lu_method = np.insert(es_lu_method, len(es_lu_method), 0.0)

diags_forL2 = np.array([Ls, es_lu_method])
positions_of_diags_forL2 = np.array([0, -1])
L2 = spdiags(diags_forL2, positions_of_diags_forL2, 100, 100)  # .toarray()

# import  scipy.linalg
# a1, a2,a3 = scipy.linalg.lu(A)
