import numpy as np
from scipy.sparse import spdiags


def decomp_cholesky_tridiag(A):
    l_diag = []
    l_offdiag = []
    n = A.shape[0]

    ds = A.diagonal() #d_i
    es = A.diagonal(-1) #e_i

    l_diag.append(np.sqrt(ds[0])) # l_11 = sqrt(d_1)
    for i in range(1, n): # i=2~n-1  (in python i=1~n-2)
        l_offdiag.append(es[i-1]/l_diag[i-1]) #l_{i,i-1} = e_i / l_{i-1,i-1}
        l_diag.append(np.sqrt(ds[i]-np.square(l_offdiag[i-1]))) # l_{ii} = sqrt(d_i - (l_{i,i-1})^2)
    l_offdiag = np.insert(l_offdiag,n-1,0.0) # necessary for spdiags

    l_diag = np.array(l_diag)
    l_offdiag = np.array(l_offdiag)

    #l_diag, l_offdiag transformed to diagonal matrix L
    diags_forL = np.array([l_diag, l_offdiag])
    positions_of_diags_forL = np.array([0, -1])
    L = spdiags(diags_forL, positions_of_diags_forL, n, n)

    return L

M = np.float64(np.array([[2.,-1.,0.],
              [-1.,2.,-1.],
              [0.,-1.,2.]]))
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

print("M is positive def?", is_pos_def(M))
my_cholesky_decomposition = decomp_cholesky_tridiag(M)
python_cholesky_decomposition = np.linalg.cholesky(M)
print("is the result of my function the same as numpy's ?")
print(np.allclose(my_cholesky_decomposition.toarray(), python_cholesky_decomposition))
print("")


# algorithms for solving the system Ax =b
# using the Cholesky decomposition algorithm

def solve_cholesky_forwards(L, b):
    n = len(b)
    l_diag = L.diagonal() # l_{i,i}
    l_offfdiag = L.diagonal(-1) # l_{i,i-1}
    y = [b[0] / l_diag[0]] #y_1 = b_1/l_{11}
    for i in range(1,n): #i=2,..,n
        y.append((b[i] - l_offfdiag[i-1]*y[i-1])/l_diag[i])
    return np.array(y)

def solve_cholesky_backwards(L, y):
    n = len(y)
    l_diag = L.diagonal()
    l_off_diag = L.diagonal(1)

    x = [y[n-1]/l_diag[n-1]] # that's the last element, so will have to reverse x in the end
    count = 0
    for i in range(n-2,-1,-1):
        count += 1
        x.append((y[i] - l_off_diag[i]*x[count -1 ])/l_diag[i])
    return np.array(x[::-1])

def solve_cholesky(A,b):
    L = decomp_cholesky_tridiag(A)
    y = solve_cholesky_forwards(L,b)
    x = solve_cholesky_backwards(L.T,y)
    return x

# Test that solving algorithms are correct
M = np.array([[2.,-1.,0.],
              [-1.,2.,-1.],
              [0.,-1.,2.]])
b = np.array([2.,1.,0.])
print(is_pos_def(M))
np_solution = np.linalg.solve(M,b)
my_solution = solve_cholesky(M, b)
print(np.allclose(my_solution, np_solution))
print(my_solution)
print(np_solution)