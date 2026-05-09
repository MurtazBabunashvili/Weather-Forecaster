import numpy as np

def lu_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n-1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k+1, n):
            if U[k, k] == 0:
                continue
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor*U[k,k:]
    return L, U, P

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1] @ x[i+1:]) / U[i,i]
    return x

def lu_solve(A, b):
    L, U, P = lu_decomposition(A)
    pb = P @ np.asarray(b, dtype=float)
    y = forward_substitution(L, pb)
    x = backward_substitution(U, y)
    return x


def cholesky(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = A[i,j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive-definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j,j]
    return L


def cholesky_solve(A, b):
    L = cholesky(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return x

def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    A = np.array(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    residuals = []

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = A[i, :i] @ x[:i] + A[i, i+1] @ x_old[i+1]
            x[i] = (b[i] - sigma) / A[i,i]
        res = np.linalg.norm(b - A@x)
        residuals.append(res)
        if res < tol:
            break
    return x, residuals

def condition_number(A, norm_type=2):
    A = np.array(A, dtype=float)
    return np.linalg.cond(A, p=norm_type)

def is_well_conditioned(A, threshold=1e6):
    kappa = condition_number(A)
    return kappa < threshold, kappa