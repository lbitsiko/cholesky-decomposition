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

#test
A = np.random.randn(10,10)
A = A.dot(A.T) # now A is positive definite
my_cholesky_decomposition = cholesky_decomp(A)
python_cholesky_decomposition = np.linalg.cholesky(A)
print("is the result of my function the same as numpy's ?")
print(np.allclose(my_cholesky_decomposition, python_cholesky_decomposition))

def solve_cholesky(A,b):
    L = cholesky_decomp(A)
    y = solve_backwards(L, b)
    x = solve_backwards(L.T, y)
    return x
    # n = len(b)
    # y_2 = np.zeros_like(b)
    # # todo: remove the test if both methods are the same
    # for i in range(n):
    #     #alternative version, but unpythonic
    #     if  (i == 0): y_2[i] = b[i]/L[i,i]
    #     sumation = 0
    #     for k in range(i):
    #         sumation += L[i,k]*y_2[k]
    #     y_2[i] = (b[i] - sumation)/L[i,i]
    #     # #the pythonic version that is in the function I made below
    #     # y[i] = (b[i] - np.sum(L[i,:i]*y[:i]))/L[i,i]
    # print("Are two methods the same?")
    # print(np.allclose(y,y_2))
    # # L_transpose = L.T
    # # for i in range(n):
    # #     x[i] = (y[i] - np.sum(L_transpose[i, :i] * x[:i])) / L_transpose[i, i]

def solve_backwards(A,b):
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        x[i] = (b[i] - np.sum(L[i, :i] * x[:i])) / A[i, i]
    return x