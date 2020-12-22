def solve_backwards_lower_diag(A, b):
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        x[i] = (b[i] - np.sum(A[i, :i] * x[:i])) / A[i, i]
    return x

def solve_backwards_lower_diag_unpythonic(A, b):
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        if (i == 0): x[i] = b[i] / A[i, i]
        sumation = 0
        for k in range(i):
            sumation += A[i, k] * x[k]
        x[i] = (b[i] - sumation) / A[i, i]
    return x
