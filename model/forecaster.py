import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_era5, summary
from data.preprocess import build_features, Normalizer, interpolate_missing_times
from numerics.svd_pca import PCA, flatten_fields
from numerics.optimization import least_squares_normal, adam, mse_grad
from numerics.linear_solvers import condition_number, is_well_conditioned
from numerics.interpolation import cubic_spline, cubic_spline_evaluate



class WeatherForecaster:
    def __init__(self, nc_path: str, target_var: str = "t2m", n_lags: int = 6, n_pca: int = 8, lead_hours: int = 6, lam: float = 1e-3, smooth_sigma: float = 1.0):
        self.nc_path = nc_path
        self.target_var = target_var
        self.n_lags = n_lags
        self.n_pca = n_pca
        self.lead_hours = lead_hours
        self.lam = lam
        self.smooth_sigma = smooth_sigma

        self.pca_ = None
        self.normalizer_ = None
        self.weights_ = None
        self.bias_ = None
        self.feature_names_ = None
        self.data_ = None
        self.X_norm_ = None
        self.y_norm_ = None
        self.condition_number_ = None

    def fit(self, optimizer: str = "lstsq"):
        print("─" * 55)
        print(f"[1/5] Loading ERA5: {self.nc_path}")
        self.data_ = load_era5(self.nc_path)
        summary(self.data_)

        print("[2/5] Engineering spatial features …")
        X_raw, self.feature_names_ = build_features(self.data_, smooth_sigma=self.smooth_sigma)

        print("[3/5] PCA compression …")
        self.pca_ = PCA(n_components=self.n_pca)
        self.pca_.fit(X_raw)
        Z = self.pca_.transform(X_raw)
        k_needed = self.pca_.choose_k(0.95)
        print(f"      {self.n_pca} components explain "
              f"{100 * self.pca_.cumulative_variance()[self.n_pca - 1]:.1f}% variance "
              f"(95% threshold needs {k_needed})")

        print("[4/5] Building lag-feature matrix …")
        X_lag, y = self._make_lag_matrix(Z,target_col=0, target_raw=self._get_target_series())

        self.normalizer_ = Normalizer()
        X_norm = self.normalizer_.fit_transform(X_lag)

        well, kappa = is_well_conditioned(X_norm.T @ X_norm)
        self.condition_number_ = kappa
        print(f"      κ(XᵀX) = {kappa:.3e}  → {'well' if well else 'ILL'}-conditioned")

        print(f"[5/5] Fitting with optimizer='{optimizer}' …")
        X_b = np.column_stack([X_norm, np.ones(len(X_norm))])

        if optimizer == "lstsq":
            w = least_squares_normal(X_b, y, lam=self.lam)
            self.weights_ = w[:-1]
            self.bias_ = w[-1]

        elif optimizer == "adam":
            w0 = np.zeros(X_b.shape[1])
            grad_fn = lambda w: mse_grad(X_b, y, w) + 2 * self.lam * w
            w, hist = adam(grad_fn, w0, lr=1e-3, max_iter=3000)
            self.weights_ = w[:-1]
            self.bias_ = w[-1]
            print(f"      Adam converged in {len(hist)} steps, "
                  f"final ‖g‖ = {hist[-1]:.3e}")
        else:
            raise ValueError("optimizer must be 'lstsq' or 'adam'")

        train_pred = X_b @ np.append(self.weights_, self.bias_)
        rmse = np.sqrt(np.mean((train_pred - y) ** 2))
        print(f"      Train RMSE = {rmse:.4f}  (normalised units)")
        print("─" * 55)
        self.X_norm_ = X_norm
        self.y_norm_ = y
        return self


    def predict(self, n_steps: int = 24) -> dict:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")

        Z = self.pca_.transform(
            build_features(self.data_, self.smooth_sigma)[0])
        target = self._get_target_series()
        _, y_all = self._make_lag_matrix(Z, target_col=0, target_raw=target)

        lag_window = self.X_norm_[-1].copy()
        preds = []

        for _ in range(n_steps):
            x_b = np.append(lag_window, 1.0)
            y_hat = x_b @ np.append(self.weights_, self.bias_)
            preds.append(y_hat)
            lag_window = np.roll(lag_window, -self.n_pca)
            lag_window[-self.n_pca:] = y_hat  # simplified: repeat scalar

        preds = np.array(preds)
        steps = np.arange(n_steps)

        if n_steps > 3:
            spdata = cubic_spline(steps.astype(float), preds, mode="natural")
            fine = np.linspace(0, n_steps - 1, n_steps * 4)
            smooth = cubic_spline_evaluate(spdata, fine)
        else:
            fine, smooth = steps.astype(float), preds

        return {"steps": steps, "forecast": preds, "fine_steps": fine, "forecast_smooth": smooth}


    def _get_target_series(self) -> np.ndarray:
        arr = self.data_[self.target_var]
        return arr.mean(axis=(1, 2)).astype(float)

    def _make_lag_matrix(self, Z: np.ndarray, target_col: int,  target_raw: np.ndarray) -> tuple:
        T = len(Z)
        start = self.n_lags
        end = T - self.lead_hours
        rows = []
        ys = []
        for t in range(start, end):
            row = Z[t - self.n_lags:t].ravel()
            rows.append(row)
            ys.append(target_raw[t + self.lead_hours])
        return np.array(rows), np.array(ys)
