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
    x = [b[0] / A[0, 0]]
    for i in range(1,n):
        x.append((b[i] - A[i, i - 1] * x[i - 1]) / A[i, i])
    return np.array(x)

def solve_cholesky_backwards(A, b):
    n = len(b)
    x = [b[n - 1] / A[n - 1, n - 1]] # that's the last element, so will have to reverse x in the end
    count = 0
    for i in range(n-2,-1,-1):
        count += 1
        x.append((b[i] - A[i, i + 1] * x[count - 1]) / A[i, i])
    return np.array(x[::-1])

def solve_cholesky(A,b):
    L = cholesky_decomp(A)
    y = solve_cholesky_forwards(L,b)
    x = solve_cholesky_backwards(L.T,y)
    return x
