from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought").resolve()

DATA_DIR = PROJECT_ROOT / "data"
RASTERS_DIR = PROJECT_ROOT / "rasters"

CMIP6_SCENARIOS_DIR = DATA_DIR / "scenarios"
ERA5_D2M_DIR = DATA_DIR / "simulated" / "era5_land" / "d2m"

ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

OUTPUT_ROOT = RASTERS_DIR / "corrected" / "cmip6"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SCENARIOS = ["ssp245", "ssp370", "ssp585"]
FUTURE_START = 2026
FUTURE_END = 2050

RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID = f"correct_cmip6_d2m_{RUN_TS}"

# latest trained model from previous step
MODEL_PATH = MODELS_DIR / "cmip6_bias_model_d2m_20260321T120959Z.joblib"

# use one ERA5 raster as the target output grid template
ERA5_TEMPLATE_PATH = ERA5_D2M_DIR / "d2m_201412.tif"

MANIFEST_PATH = MANIFESTS_DIR / f"{RUN_ID}_manifest.json"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def nc_time_to_yyyymm(time_values) -> list[str]:
    t = pd.to_datetime(time_values)
    return [f"{x.year}{x.month:02d}" for x in t]


def specific_humidity_to_vapor_pressure(q: np.ndarray, p_pa: float = 101325.0) -> np.ndarray:
    """
    q (kg/kg) -> vapor pressure e (Pa)
    """
    q = np.asarray(q, dtype=np.float64)
    return (q * p_pa) / (0.622 + 0.378 * q)


def vapor_pressure_to_dewpoint_kelvin(e_pa: np.ndarray) -> np.ndarray:
    """
    vapor pressure (Pa) -> dewpoint (K)
    """
    e_pa = np.asarray(e_pa, dtype=np.float64)
    e_pa = np.clip(e_pa, 1.0, None)
    ln_ratio = np.log(e_pa / 611.2)
    td_c = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    return td_c + 273.15


def prepare_lon_lat_and_array(
    arr: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert lon from 0..360 to -180..180, sort longitudes,
    and flip latitude to descending order for raster writing.
    """
    lon_180 = np.where(lon > 180.0, lon - 360.0, lon)
    sort_idx = np.argsort(lon_180)

    lon_180 = lon_180[sort_idx]
    arr = arr[:, sort_idx]

    if lat[0] < lat[-1]:
        lat = lat[::-1]
        arr = arr[::-1, :]

    return arr, lat, lon_180


def build_src_transform(lat_desc: np.ndarray, lon_180: np.ndarray, height: int, width: int):
    lon_res = float(np.abs(np.diff(lon_180).mean()))
    lat_res = float(np.abs(np.diff(lat_desc).mean()))

    left = float(lon_180.min() - lon_res / 2.0)
    right = float(lon_180.max() + lon_res / 2.0)
    bottom = float(lat_desc.min() - lat_res / 2.0)
    top = float(lat_desc.max() + lat_res / 2.0)

    return from_bounds(left, bottom, right, top, width, height)


# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------
print("=" * 60)
print("CMIP6 D2M FUTURE CORRECTION")
print("=" * 60)
print(f"Run ID      : {RUN_ID}")
print(f"Model path  : {MODEL_PATH}")
print(f"Grid source : {ERA5_TEMPLATE_PATH}")

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------
# LOAD TARGET GRID TEMPLATE
# ---------------------------------------------------------------------
with rasterio.open(ERA5_TEMPLATE_PATH) as ref:
    dst_meta = ref.meta.copy()
    dst_transform = ref.transform
    dst_crs = ref.crs
    dst_height = ref.height
    dst_width = ref.width

dst_meta.update(
    dtype="float32",
    count=1,
    compress="lzw",
)

# ---------------------------------------------------------------------
# PROCESS EACH SCENARIO
# ---------------------------------------------------------------------
manifest = {
    "run_id": RUN_ID,
    "model_path": str(MODEL_PATH),
    "era5_template_path": str(ERA5_TEMPLATE_PATH),
    "future_period": [FUTURE_START, FUTURE_END],
    "scenarios": {},
}

for scenario in SCENARIOS:
    print("\n" + "-" * 60)
    print(f"SCENARIO: {scenario}")
    print("-" * 60)

    huss_dir = CMIP6_SCENARIOS_DIR / scenario / "huss"
    out_dir = OUTPUT_ROOT / scenario / "d2m"
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(
        fp for fp in huss_dir.glob("*.nc")
        if "Amon" in fp.name
    )

    if not nc_files:
        print("No valid huss NetCDF file found. Skipping.")
        manifest["scenarios"][scenario] = {
            "status": "missing_huss_nc",
            "months_written": 0,
        }
        continue

    nc_path = nc_files[0]
    print(f"Using file: {nc_path.name}")

    ds = xr.open_dataset(nc_path, engine="netcdf4")
    months = nc_time_to_yyyymm(ds["time"].values)

    written = 0
    first_month = None
    last_month = None

    for time_idx, yyyymm in enumerate(months):
        year = int(yyyymm[:4])
        if not (FUTURE_START <= year <= FUTURE_END):
            continue

        out_path = out_dir / f"d2m_corrected_{yyyymm}.tif"
        if out_path.exists():
            print(f"Already exists, skipping: {out_path.name}")
            continue

        q = ds["huss"].isel(time=time_idx).values.astype(np.float32)
        lat = ds["lat"].values
        lon = ds["lon"].values

        # prepare grid orientation
        q, lat_desc, lon_180 = prepare_lon_lat_and_array(q, lat, lon)

        # physics conversion
        td_src = vapor_pressure_to_dewpoint_kelvin(
            specific_humidity_to_vapor_pressure(q)
        ).astype(np.float32)

        # source transform
        src_transform = build_src_transform(
            lat_desc=lat_desc,
            lon_180=lon_180,
            height=td_src.shape[0],
            width=td_src.shape[1],
        )

        # reproject to ERA5 grid
        td_regridded = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=td_src,
            destination=td_regridded,
            src_transform=src_transform,
            src_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=np.nan,
        )

        # bias correction
        flat = td_regridded.reshape(-1)
        valid_mask = np.isfinite(flat)

        corrected_flat = np.full(flat.shape, np.nan, dtype=np.float32)

        if valid_mask.any():
            X = pd.DataFrame({
                "d2m_cmip6": flat[valid_mask].astype(np.float32)
            })
            corrected_flat[valid_mask] = model.predict(X).astype(np.float32)

        corrected = corrected_flat.reshape(dst_height, dst_width)

        # write raster
        with rasterio.open(out_path, "w", **dst_meta) as dst:
            dst.write(corrected, 1)

        written += 1
        first_month = first_month or yyyymm
        last_month = yyyymm

        print(
            f"Wrote {out_path.name} | "
            f"min={np.nanmin(corrected):.3f} "
            f"max={np.nanmax(corrected):.3f} "
            f"mean={np.nanmean(corrected):.3f}"
        )

    ds.close()

    manifest["scenarios"][scenario] = {
        "status": "ok",
        "source_nc": str(nc_path),
        "output_dir": str(out_dir),
        "months_written": written,
        "first_month": first_month,
        "last_month": last_month,
    }

# ---------------------------------------------------------------------
# SAVE MANIFEST
# ---------------------------------------------------------------------
save_json(manifest, MANIFEST_PATH)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Manifest saved: {MANIFEST_PATH}")