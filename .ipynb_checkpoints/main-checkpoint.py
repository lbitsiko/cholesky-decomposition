import numpy as np

def cholesky_decomp(A):
    L = np.zeros_like(A)
    n = len(L)
    for i in range(n):
        for j in range(i+1):#algo for j=1~i-1, but python starts with 0 so it's till i, but range does not include i+1
            if i==j:
                L[i,i] = np.sqrt(A[i,i]-np.sum(np.square(L[i,:i]))) #array[:i] means elements till i-1
            else:
                L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j]))/L[j,j]
    return L

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def solve_cholesky_forwards(A, b):
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        if i==0:
            x[0] = b[0] / A[0, 0]
        else:
            x[i] = (b[i] - A[i, i - 1] * x[i - 1]) / A[i, i]
    return x

def solve_cholesky_backwards(A, b):
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n-1,-1,-1):
        if i == n-1:
            x[n-1] = b[n - 1] / A[n - 1, n - 1]
        else:
            x[i] = (b[i] - A[i, i + 1] * x[i + 1]) / A[i, i]
    return x

def solve_cholesky(A,b):
    L = cholesky_decomp(A)
    y = solve_cholesky_forwards(L,b)
    x = solve_cholesky_backwards(L.T,y)
    return x


# test that my cholesky decomosition is correct
A = np.random.randn(10,10)
A = A.dot(A.T) # now A is positive definite
my_cholesky_decomposition = cholesky_decomp(A)
python_cholesky_decomposition = np.linalg.cholesky(A)
print("is the result of my function the same as numpy's ?")
print(np.allclose(my_cholesky_decomposition, python_cholesky_decomposition))
print("")

#test that my solving algorithm is correct
M = np.array([[2.,-1.,0.],
              [-1.,2.,-1.],
              [0.,-1.,2.]])
const = np.array([2.,1.,0.])
print("Mx = b")
print("Matrix M", M)
print("b ", const)
print("Is Matrix M is positive definite?", is_pos_def(M))
np_solution = np.linalg.solve(M,const)
my_solution = solve_cholesky(M, const)
print("Is my solution the same as numpy's?")
print(np.allclose(my_solution, np_solution))
print("")
print("my solution", my_solution)
print("numpy solution", np_solution)
