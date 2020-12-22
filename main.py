import numpy as np

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
