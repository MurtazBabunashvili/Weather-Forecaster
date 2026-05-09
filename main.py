#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.forecaster      import WeatherForecaster
from visualization.plot_forecast     import plot_dashboard, plot_pca_variance, plot_forecast


def main():
    parser = argparse.ArgumentParser(description="Numerical Weather Forecaster")
    parser.add_argument("--nc",        type=str,   required=True, help="Path to ERA5 .nc file")
    parser.add_argument("--target",    type=str,   default="t2m", help="Variable to forecast: t2m | msl | u10 | v10")
    parser.add_argument("--lags",      type=int,   default=6, help="Number of lag time steps (default 6)")
    parser.add_argument("--pca",       type=int,   default=8, help="Number of PCA components (default 8)")
    parser.add_argument("--lead",      type=int,   default=6,  help="Forecast lead time in hours (default 6)")
    parser.add_argument("--steps",     type=int,   default=24,  help="Number of forecast steps to predict (default 24)")
    parser.add_argument("--optimizer", type=str,   default="lstsq", choices=["lstsq", "adam"], help="Optimizer: lstsq (normal eqs) | adam (Ch. 22)")
    parser.add_argument("--lam",       type=float, default=1e-3, help="L2 regularisation lambda (default 1e-3)")
    parser.add_argument("--sigma",     type=float, default=1.0, help="Gaussian smoothing sigma in grid cells (default 1.0)")
    parser.add_argument("--save",      type=str,   default=None, help="Directory to save plots (optional)")
    args = parser.parse_args()

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    fc = WeatherForecaster(
        nc_path      = args.nc,
        target_var   = args.target,
        n_lags       = args.lags,
        n_pca        = args.pca,
        lead_hours   = args.lead,
        lam          = args.lam,
        smooth_sigma = args.sigma,
    )

    fc.fit(optimizer=args.optimizer)
    result = fc.predict(n_steps=args.steps)

    print(f"\nForecast produced: {args.steps} steps, lead={args.lead}h")
    print(f"Condition number κ(XᵀX) = {fc.condition_number_:.3e}")

    plot_pca_variance(fc.pca_, save_path=str(save_dir / "pca_variance.png") if save_dir else None)
    plot_forecast(
        result,
        title  = f"ERA5 {args.target.upper()} forecast  |  lead={args.lead}h",
        ylabel = f"{args.target} (normalised)",
        save_path=str(save_dir / "forecast.png") if save_dir else None
    )
    plot_dashboard(fc.data_, result, save_path=str(save_dir / "dashboard.png") if save_dir else None)


if __name__ == "__main__":
    main()