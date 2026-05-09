import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numerics.convolution      import gaussian_kernel, convolve2d_separable, convolve2d, laplacian_kernel
from numerics.finite_differences import differentiate
from numerics.interpolation    import cubic_spline, cubic_spline_evaluate


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_field(field_3d: np.ndarray, sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:

    G   = gaussian_kernel(kernel_size, sigma)
    g1d = G[kernel_size // 2, :]
    row_k = g1d.reshape(1, -1)
    col_k = g1d.reshape(-1, 1)
    out = np.empty_like(field_3d)
    for t in range(field_3d.shape[0]):
        out[t] = convolve2d_separable(field_3d[t], row_k, col_k)
    return out



def spatial_gradients(field_3d: np.ndarray, dlat_km: float = 111.0, dlon_km: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    T, nlat, nlon = field_3d.shape
    grad_lat = np.empty_like(field_3d)
    grad_lon = np.empty_like(field_3d)

    for t in range(T):
        for j in range(nlon):
            grad_lat[t, :, j] = differentiate(field_3d[t, :, j], dlat_km)
        for i in range(nlat):
            grad_lon[t, i, :] = differentiate(field_3d[t, i, :], dlon_km)

    return grad_lat, grad_lon


def laplacian_field(field_3d: np.ndarray) -> np.ndarray:
    K = laplacian_kernel()
    out = np.empty_like(field_3d)
    for t in range(field_3d.shape[0]):
        out[t] = convolve2d(field_3d[t], K)
    return out



def wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    return np.sqrt(u10**2 + v10**2)

def wind_direction(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    return (270 - np.degrees(np.arctan2(v10, u10))) % 360

def temperature_celsius(t2m_K: np.ndarray) -> np.ndarray:
    return t2m_K - 273.15



def interpolate_missing_times(times: np.ndarray, field_3d: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    T, nlat, nlon = field_3d.shape
    T_new = len(query_times)
    out   = np.empty((T_new, nlat, nlon), dtype=float)

    for i in range(nlat):
        for j in range(nlon):
            col    = field_3d[:, i, j].astype(float)
            spdata = cubic_spline(times, col, mode="natural")
            out[:, i, j] = cubic_spline_evaluate(spdata, query_times)

    return out

class Normalizer:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0   # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)



def build_features(data: dict, smooth_sigma: float = 1.0) -> np.ndarray:

    t2m = data.get("t2m")
    msl = data.get("msl")
    u10 = data.get("u10")
    v10 = data.get("v10")

    features = {}

    if t2m is not None:
        t2m_s = smooth_field(t2m, sigma=smooth_sigma)
        features["t2m_mean"]         = t2m_s.mean(axis=(1, 2))
        features["t2m_std"]          = t2m_s.std(axis=(1, 2))
        gl, glon = spatial_gradients(t2m_s)
        features["t2m_gradlat_mean"] = gl.mean(axis=(1, 2))
        features["t2m_gradlon_mean"] = glon.mean(axis=(1, 2))

    if msl is not None:
        msl_s = smooth_field(msl, sigma=smooth_sigma)
        features["msl_mean"]         = msl_s.mean(axis=(1, 2))
        features["msl_std"]          = msl_s.std(axis=(1, 2))
        gl, glon = spatial_gradients(msl_s)
        features["msl_gradlat_mean"] = gl.mean(axis=(1, 2))
        features["msl_gradlon_mean"] = glon.mean(axis=(1, 2))
        lap = laplacian_field(msl_s)
        features["msl_laplacian"]    = lap.mean(axis=(1, 2))

    if u10 is not None and v10 is not None:
        ws = wind_speed(u10, v10)
        wd = wind_direction(u10, v10)
        features["wind_speed_mean"]  = ws.mean(axis=(1, 2))
        features["wind_dir_mean"]    = wd.mean(axis=(1, 2))
        features["u10_mean"]         = u10.mean(axis=(1, 2))
        features["v10_mean"]         = v10.mean(axis=(1, 2))

    if not features:
        found = [k for k in data if k not in ("time", "lat", "lon")]
        raise ValueError(
            f"build_features: no recognised variables found.\n"
            f"  Keys in data: {found}\n"
            f"  Expected any of: t2m, msl, u10, v10"
        )
    X = np.column_stack(list(features.values()))
    feature_names = list(features.keys())
    return X, feature_names