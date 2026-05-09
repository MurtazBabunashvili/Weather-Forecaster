import numpy as np

def svd(A, full_matrices=False):
    A = np.array(A, dtype=float)
    U, sigma, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    return U, sigma, Vt

def truncated_svd(A,k):
    U, sigma, Vt = svd(A, full_matrices=False)
    k = min(k, len(sigma))
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    Vt_k = Vt[:k, :]
    A_k = (U_k * sigma_k) @ Vt_k
    return A_k, (U_k, sigma_k, Vt_k)

class PCA:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.sigma_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        U, sigma, Vt = svd(X_c, full_matrices=False)
        k = min(self.n_components, len(sigma))

        self.components_ = Vt[:k]
        self.sigma_ = sigma[:k]

        total_var = np.sum(sigma**2)
        self.explained_variance_ratio_ = (sigma[:k] ** 2)/total_var if total_var > 0 else sigma[:k] * 0

        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_c = np.array(X, dtype=float) - self.mean_
        return X_c @ self.components_.T

    def inverse_transformation(self, Z: np.ndarray) -> np.ndarray:
        return np.array(Z, dtype=float) @ self.components_ + self.mean_

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)

    def choose_k(self, threshold: float = 0.95) -> int:
        cum = self.cumulative_variance()
        idx = np.searchsorted(cum, threshold)
        return int(idx) + 1

def flatten_fields(data_dict: dict) -> np.ndarray:
    arrays = []
    for arr in data_dict.values():
        T = arr.shape[0]
        arrays.append(arr.reshape(T, -1))
    return np.concatenate(arrays, axis=1)