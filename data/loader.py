import numpy as np
import xarray as xr
from pathlib import Path

VARIABLE_ALIASES = {
    "t2m":                "t2m",
    "2m_temperature":     "t2m",
    "msl":                "msl",
    "mean_sea_level_pressure": "msl",
    "sp": "msl",
    "surface_pressure": "msl",
    "u10":                "u10",
    "10m_u_component_of_wind": "u10",
    "v10":                "v10",
    "10m_v_component_of_wind": "v10",
    "tp":                 "tp",
    "total_precipitation":"tp",
}


def load_era5(path: str | Path, variables: list[str] | None = None) -> dict:
    ds = xr.open_dataset(path, engine="netcdf4")

    lat_key = "latitude" if "latitude" in ds.coords else "lat"
    lon_key = "longitude" if "longitude" in ds.coords else "lon"
    time_key = "time" if "time" in ds.coords else "valid_time"

    out = {
        "time": ds[time_key].values,
        "lat": ds[lat_key].values,
        "lon": ds[lon_key].values,
    }

    for raw_name, ds_var in ds.data_vars.items():
        internal = VARIABLE_ALIASES.get(raw_name, raw_name)
        if variables is not None and internal not in variables:
            continue
        arr = ds_var.values.astype(np.float32)

        dims = list(ds_var.dims)
        time_idx = next((i for i, d in enumerate(dims) if "time" in d or "valid" in d), 0)
        lat_idx = next((i for i, d in enumerate(dims) if "lat" in d), 1)
        lon_idx = next((i for i, d in enumerate(dims) if "lon" in d), 2)
        arr = np.transpose(arr, (time_idx, lat_idx, lon_idx))
        if internal not in out:
            out[internal] = arr
    ds.close()
    return out


def load_era5_multi(paths: list[str | Path], **kwargs) -> dict:
    parts = [load_era5(p, **kwargs) for p in paths]
    merged = {"lat": parts[0]["lat"], "lon": parts[0]["lon"]}
    merged["time"] = np.concatenate([p["time"] for p in parts])
    for key in parts[0]:
        if key in ("time", "lat", "lon"):
            continue
        merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged


def summary(data: dict):
    print(f"  time steps : {len(data['time'])}")
    print(f"  lat range  : {data['lat'].min():.2f} – {data['lat'].max():.2f}")
    print(f"  lon range  : {data['lon'].min():.2f} – {data['lon'].max():.2f}")
    for k, v in data.items():
        if k in ("time", "lat", "lon"):
            continue
        print(f"  {k:6s} : shape {v.shape},  "
              f"min={float(np.nanmin(v)):.2f},  max={float(np.nanmax(v)):.2f}")
