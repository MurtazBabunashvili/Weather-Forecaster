# Weather Forecaster

A numerical weather forecasting system built on ERA5 reanalysis data. Uses spatial feature engineering, PCA compression, and linear regression with lag features to forecast atmospheric variables over the Georgia / Caucasus region.

---

## Project Structure

```
weather_forecaster/
├── data/
│   ├── loader.py           # ERA5 NetCDF loading and variable aliasing
│   └── preprocess.py       # Feature engineering, smoothing, normalisation
├── model/
│   └── forecaster.py       # WeatherForecaster: fit / predict pipeline
├── numerics/
│   ├── convolution.py      # 2D convolution, Gaussian/Laplacian kernels
│   ├── finite_differences.py  # Forward, backward, central, Richardson
│   ├── interpolation.py    # Lagrange, Newton, Hermite, cubic spline
│   ├── linear_solvers.py   # LU, Cholesky, Gauss-Seidel, condition number
│   ├── optimization.py     # Gradient descent, Adam, least squares normal equations
│   └── svd_pca.py          # SVD, truncated SVD, PCA
├── visualization/
│   └── plot_forecast.py    # Dashboard, forecast, PCA variance, gradient plots
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11+
- numpy
- xarray
- netCDF4
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Expects ERA5 reanalysis data in NetCDF format (`.nc`). The following variables are supported and will be aliased automatically:

| Variable | Aliases accepted |
|---|---|
| `t2m` | `2m_temperature` |
| `msl` | `mean_sea_level_pressure`, `sp`, `surface_pressure` |
| `u10` | `10m_u_component_of_wind` |
| `v10` | `10m_v_component_of_wind` |
| `tp` | `total_precipitation` |

ERA5 data can be downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

---

## Usage

```bash
python main.py --nc <path_to_nc_file> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--nc` | required | Path to ERA5 `.nc` file |
| `--target` | `t2m` | Variable to forecast: `t2m`, `msl`, `u10`, `v10` |
| `--lags` | `6` | Number of lag time steps |
| `--pca` | `8` | Number of PCA components |
| `--lead` | `6` | Forecast lead time in hours |
| `--steps` | `24` | Number of forecast steps to produce |
| `--optimizer` | `lstsq` | Solver: `lstsq` (normal equations) or `adam` |
| `--lam` | `1e-3` | L2 regularisation lambda |
| `--sigma` | `1.0` | Gaussian smoothing sigma in grid cells |
| `--save` | `None` | Directory to save output plots |

### Examples

Basic forecast of 2m temperature, 24 steps ahead:

```bash
python main.py --nc era5_georgia_2023.nc --target t2m --steps 24 --save ./outputs
```

Use Adam optimiser with stronger regularisation:

```bash
python main.py --nc era5_georgia_2023.nc --target t2m --optimizer adam --lam 1e-2 --steps 24
```

Forecast sea-level pressure with more lag context:

```bash
python main.py --nc era5_georgia_2023.nc --target msl --lags 12 --pca 4 --lead 6 --steps 48
```

---

## Pipeline

```
ERA5 .nc file
     │
     ▼
[1] Load & alias variables (loader.py)
     │
     ▼
[2] Spatial feature engineering (preprocess.py)
     │  Gaussian smoothing, spatial gradients,
     │  Laplacian, wind speed/direction, t2m/msl stats
     ▼
[3] PCA compression (svd_pca.py)
     │  SVD-based dimensionality reduction
     │  Retains configurable number of components
     ▼
[4] Lag feature matrix construction (forecaster.py)
     │  Sliding window of n_lags PCA vectors → X
     │  Target: domain-mean of target variable at t + lead_hours → y
     │  Both X and y z-score normalised independently
     ▼
[5] Linear regression fit (optimization.py)
     │  lstsq: L2-regularised normal equations via Cholesky
     │  adam:  stochastic gradient descent with momentum
     ▼
[6] Autoregressive prediction (forecaster.py)
     │  Rolls lag window forward n_steps times
     │  Output inverse-transformed to physical units
     ▼
[7] Cubic spline smoothing + plots (interpolation.py, plot_forecast.py)
```

---

## Outputs

Three plots are saved when `--save` is specified:

**`pca_variance.png`** — Individual and cumulative explained variance per principal component, with 95% threshold line.

**`forecast.png`** — Predicted values at each step (scatter) with cubic spline interpolation (line).

**`dashboard.png`** — 2×2 panel showing the last time step of: temperature (°C), sea-level pressure (hPa), wind speed (m/s), and the forecast series.

---

## Numerics

All numerical methods are implemented from scratch in the `numerics/` package without relying on scipy or sklearn.

| Module | Methods |
|---|---|
| `convolution.py` | 2D convolution, separable convolution, Gaussian, Sobel, Laplacian kernels |
| `finite_differences.py` | Forward / backward / central differences, Richardson extrapolation |
| `interpolation.py` | Lagrange, Newton divided differences, Hermite, natural and clamped cubic spline |
| `linear_solvers.py` | LU decomposition with partial pivoting, Cholesky, Gauss-Seidel, condition number |
| `optimization.py` | Gradient descent, backtracking line search, Adam, L2-regularised least squares |
| `svd_pca.py` | Full and truncated SVD, PCA with explained variance and component selection |

---

## Known Limitations

- The autoregressive rollout feeds the last known PCA observation into the lag window at each step rather than re-running the full preprocessing pipeline. Forecast skill degrades for longer horizons.
- The model predicts the spatial domain mean, not per-gridpoint values.
- No train/validation split is enforced; RMSE reported is in-sample.
- PCA with 8 components on ~14 engineered features provides minimal compression. Using `--pca 2` or `--pca 3` is sufficient for this feature set.
