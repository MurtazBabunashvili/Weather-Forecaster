import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def plot_forecast(result: dict, title: str = "Weather Forecast", ylabel: str = "Value (normalised)", save_path: str | None = None):

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(result["steps"], result["forecast"], color="steelblue", s=40, zorder=3, label="Predicted (step)")
    ax.plot(result["fine_steps"], result["forecast_smooth"], color="tomato", lw=2, label="Spline smooth (Ch. 28)")
    ax.set_xlabel("Forecast step (hours ahead)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def plot_field(field_2d: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str = "Field", cmap: str = "RdYlBu_r", units: str = "", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(lon, lat, field_2d, levels=20, cmap=cmap)
    ax.contour(lon, lat, field_2d, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(units)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def plot_gradient_magnitude(field_2d: np.ndarray, grad_lat: np.ndarray, grad_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str = "Gradient magnitude (fronts)", save_path: str | None = None):

    mag = np.sqrt(grad_lat ** 2 + grad_lon ** 2)
    fig, ax = plt.subplots(figsize=(9, 5))
    cf = ax.contourf(lon, lat, field_2d, levels=20, cmap="coolwarm", alpha=0.8)
    plt.colorbar(cf, ax=ax)

    step = max(1, min(field_2d.shape) // 8)
    LON, LAT = np.meshgrid(lon, lat)
    ax.quiver(LON[::step, ::step], LAT[::step, ::step],
              grad_lon[::step, ::step], grad_lat[::step, ::step],
              color="k", scale=None, width=0.003)
    ax.set_title(title)
    ax.set_xlabel("Longitude");
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def plot_pca_variance(pca, save_path: str | None = None):
    evr = pca.explained_variance_ratio_
    cum = pca.cumulative_variance()
    k = len(evr)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(range(1, k + 1), evr * 100, color="steelblue", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")

    ax2 = ax1.twinx()
    ax2.plot(range(1, k + 1), cum * 100, "ro-", lw=2, label="Cumulative")
    ax2.axhline(95, color="grey", ls="--", lw=1, label="95% threshold")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("SVD / PCA explained variance  (Ch. 16–17)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def plot_loss(history: list, title: str = "Adam optimizer — gradient norm (Ch. 22)", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(history, color="darkorange", lw=1.5)
    ax.set_xlabel("Iteration");
    ax.set_ylabel("‖∇L‖₂")
    ax.set_title(title);
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def plot_dashboard(data: dict, result: dict | None = None,  save_path: str | None = None):
    lat = data["lat"]
    lon = data["lon"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("ERA5 – Georgia / Caucasus  (lat 41–44, lon 40–47)", fontsize=13)

    def _plot(ax, field, title, cmap, units):
        cf = ax.contourf(lon, lat, field, levels=16, cmap=cmap)
        plt.colorbar(cf, ax=ax).set_label(units)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lon");
        ax.set_ylabel("lat")

    t_idx = -1

    if "t2m" in data:
        _plot(axes[0, 0], data["t2m"][t_idx] - 273.15,
              "Temperature (°C)", "RdYlBu_r", "°C")
    if "msl" in data:
        _plot(axes[0, 1], data["msl"][t_idx] / 100,
              "Sea-level pressure (hPa)", "PuBu", "hPa")
    if "u10" in data and "v10" in data:
        ws = np.sqrt(data["u10"][t_idx] ** 2 + data["v10"][t_idx] ** 2)
        _plot(axes[1, 0], ws, "Wind speed (m/s)", "YlOrRd", "m/s")
    else:
        axes[1, 0].axis("off")

    if result is not None:
        axes[1, 1].scatter(result["steps"], result["forecast"],
                           s=30, color="steelblue", label="Forecast")
        axes[1, 1].plot(result["fine_steps"], result["forecast_smooth"],
                        "r-", lw=2, label="Spline")
        axes[1, 1].set_title("Forecast (normalised)", fontsize=10)
        axes[1, 1].set_xlabel("Step");
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
