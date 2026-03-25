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
ERA5_PET_DIR = DATA_DIR / "simulated" / "era5_land" / "pet"

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
RUN_ID = f"correct_cmip6_pet_{RUN_TS}"

MODEL_PATH = MODELS_DIR / "cmip6_bias_model_pet_20260322T115100Z.joblib"
ERA5_TEMPLATE_PATH = ERA5_PET_DIR / "pet_201412.tif"

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


def get_valid_nc(folder: Path, token: str = "Amon") -> Path:
    files = sorted(fp for fp in folder.glob("*.nc") if token in fp.name)
    if not files:
        raise FileNotFoundError(f"No valid NetCDF file found in {folder}")
    return files[0]


def get_month_slice(ds: xr.Dataset, var_name: str, time_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = ds[var_name].isel(time=time_idx).values.astype(np.float32)
    lat = ds["lat"].values
    lon = ds["lon"].values
    return arr, lat, lon


def prep_cmip6_grid(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray):
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

    return from_bounds(
        float(lon_180.min() - lon_res / 2.0),
        float(lat_desc.min() - lat_res / 2.0),
        float(lon_180.max() + lon_res / 2.0),
        float(lat_desc.max() + lat_res / 2.0),
        width,
        height,
    )


def regrid_to_template(
    arr: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    arr, lat_desc, lon_180 = prep_cmip6_grid(arr, lat, lon)
    src_transform = build_src_transform(lat_desc, lon_180, arr.shape[0], arr.shape[1])

    out = np.full(dst_shape, np.nan, dtype=np.float32)

    reproject(
        source=arr,
        destination=out,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    return out


# ---------------------------------------------------------------------
# LOAD MODEL + TEMPLATE
# ---------------------------------------------------------------------
print("=" * 60)
print("CMIP6 PET FUTURE CORRECTION")
print("=" * 60)
print(f"Run ID      : {RUN_ID}")
print(f"Model path  : {MODEL_PATH}")
print(f"Grid source : {ERA5_TEMPLATE_PATH}")

model = joblib.load(MODEL_PATH)

with rasterio.open(ERA5_TEMPLATE_PATH) as ref:
    dst_meta = ref.meta.copy()
    dst_transform = ref.transform
    dst_crs = ref.crs
    dst_shape = (ref.height, ref.width)

dst_meta.update(
    dtype="float32",
    count=1,
    compress="lzw",
)


# ---------------------------------------------------------------------
# PROCESS SCENARIOS
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

    tas_nc_path = get_valid_nc(CMIP6_SCENARIOS_DIR / scenario / "tas")
    rsds_nc_path = get_valid_nc(CMIP6_SCENARIOS_DIR / scenario / "rsds")
    huss_nc_path = get_valid_nc(CMIP6_SCENARIOS_DIR / scenario / "huss")

    out_dir = OUTPUT_ROOT / scenario / "pet"
    out_dir.mkdir(parents=True, exist_ok=True)

    tas_ds = xr.open_dataset(tas_nc_path, engine="netcdf4")
    rsds_ds = xr.open_dataset(rsds_nc_path, engine="netcdf4")
    huss_ds = xr.open_dataset(huss_nc_path, engine="netcdf4")

    months = nc_time_to_yyyymm(tas_ds["time"].values)

    written = 0
    first_month = None
    last_month = None

    for time_idx, yyyymm in enumerate(months):
        year = int(yyyymm[:4])
        if not (FUTURE_START <= year <= FUTURE_END):
            continue

        out_path = out_dir / f"pet_corrected_{yyyymm}.tif"
        if out_path.exists():
            print(f"Already exists, skipping: {out_path.name}")
            continue

        tas_src, tas_lat, tas_lon = get_month_slice(tas_ds, "tas", time_idx)
        rsds_src, rsds_lat, rsds_lon = get_month_slice(rsds_ds, "rsds", time_idx)
        huss_src, huss_lat, huss_lon = get_month_slice(huss_ds, "huss", time_idx)

        rsds_src = np.where(rsds_src < 0, 0.0, rsds_src)

        tas = regrid_to_template(tas_src, tas_lat, tas_lon, dst_shape, dst_transform, dst_crs)
        rsds = regrid_to_template(rsds_src, rsds_lat, rsds_lon, dst_shape, dst_transform, dst_crs)
        huss = regrid_to_template(huss_src, huss_lat, huss_lon, dst_shape, dst_transform, dst_crs)

        tas_c = tas - 273.15
        humidity_penalty = 1.0 - np.clip(huss / 0.03, 0.0, 0.95)

        pet_proxy = np.where(
            np.isfinite(tas_c) & np.isfinite(rsds) & np.isfinite(huss),
            np.maximum(0.0, rsds) * np.maximum(0.0, tas_c + 5.0) * humidity_penalty,
            np.nan
        ).astype(np.float32)

        flat_tas = tas.reshape(-1)
        flat_rsds = rsds.reshape(-1)
        flat_huss = huss.reshape(-1)
        flat_proxy = pet_proxy.reshape(-1)

        valid_mask = (
            np.isfinite(flat_tas) &
            np.isfinite(flat_rsds) &
            np.isfinite(flat_huss) &
            np.isfinite(flat_proxy)
        )

        corrected_flat = np.full(flat_proxy.shape, np.nan, dtype=np.float32)

        if valid_mask.any():
            X = pd.DataFrame({
                "tas_k": flat_tas[valid_mask].astype(np.float32),
                "rsds_wm2": flat_rsds[valid_mask].astype(np.float32),
                "huss": flat_huss[valid_mask].astype(np.float32),
                "pet_proxy": flat_proxy[valid_mask].astype(np.float32),
            })
            corrected_flat[valid_mask] = model.predict(X).astype(np.float32)

        corrected = corrected_flat.reshape(dst_shape)

        # PET should not be negative after correction
        corrected = np.where(np.isfinite(corrected), np.maximum(corrected, 0.0), np.nan).astype(np.float32)

        with rasterio.open(out_path, "w", **dst_meta) as dst:
            dst.write(corrected, 1)

        written += 1
        first_month = first_month or yyyymm
        last_month = yyyymm

        print(
            f"Wrote {out_path.name} | "
            f"min={np.nanmin(corrected):.8f} "
            f"max={np.nanmax(corrected):.8f} "
            f"mean={np.nanmean(corrected):.8f}"
        )

    tas_ds.close()
    rsds_ds.close()
    huss_ds.close()

    manifest["scenarios"][scenario] = {
        "status": "ok",
        "tas_nc": str(tas_nc_path),
        "rsds_nc": str(rsds_nc_path),
        "huss_nc": str(huss_nc_path),
        "output_dir": str(out_dir),
        "months_written": written,
        "first_month": first_month,
        "last_month": last_month,
    }

save_json(manifest, MANIFEST_PATH)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Manifest saved: {MANIFEST_PATH}")