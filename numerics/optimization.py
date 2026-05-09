import numpy as np

def gradient_descent(grad_fn, x0, lr=1e-3, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = []
    for _ in range(max_iter):
        g = grad_fn(x)
        x = x - lr*g
        gnorm = np.linalg.norm(g)

        history.append(gnorm)
        if gnorm < tol:
            break
    return x, history

def back_tracking_line_search(f, grad, x, direction, alpha=1.0, rho=0.5, c=1e-4):
    #Applies armijo condition
    fx = f(x)
    gd = grad(x) @ direction
    while f(x + alpha * direction) > fx + c * alpha * gd:
        alpha *= rho
        if alpha < 1e-12:
            break
    return alpha

def adam(grad_fn, x0, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=2000, tol=1e-6):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []

    for t in range(1, max_iter + 1):
        g = grad_fn(x)
        m = beta1 * m + (1-beta1) * g
        v = beta2*v + (1-beta2) * g**2
        m_hat = m/(1-beta1**t)
        v_hat = v / (1-beta2**t)

        x = x - lr * m_hat/ (np.sqrt(v_hat) + eps)
        gnorm = np.linalg.norm(g)

        history.append(gnorm)
        if gnorm < tol:
            break
    return x, history


def least_squares_normal(X, y, lam=0.0):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    XtX = X.T @ X + lam*np.eye(X.shape[1])
    Xty = X.T @ y
    try:
        L = _cholesky(XtX)
        w = _cholesky_solve(L, Xty)
    except (np.linalg.LinAlgError, ValueError):
        w, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
    return w

def _choleksy(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s/L[j,j]
    return L

def _cholesky_solve(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros_like(b, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x


def mse_loss(X, y, w):
    r = X @ w - y
    return np.dot(r, r) / len(y)


def mse_grad(X, y, w):
    r = X @ w - y
    return 2 * X.T @ r / len(y)