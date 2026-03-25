from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

from rozvidrought_datasets.grid import RozviGrid


PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought")

SCENARIOS = ["ssp245", "ssp370", "ssp585"]
SCENARIO_ROOT = PROJECT_ROOT / "data" / "scenarios"

MODELS_DIR = PROJECT_ROOT / "data" / "artifacts" / "models"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "manifests"
OUT_ROOT = PROJECT_ROOT / "rasters" / "corrected" / "cmip6"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# update these if you train newer models
TAS_MODEL_PATH = MODELS_DIR / "cmip6_bias_model_tas_20260321T102243Z.joblib"
MRSOS_MODEL_PATH = MODELS_DIR / "cmip6_bias_model_mrsos_20260321T102243Z.joblib"

TAS_FILE_GLOB = "tas_*.nc"
MRSOS_FILE_GLOB = "mrsos_*.nc"


@dataclass
class RunManifest:
    run_id: str
    created_at_utc: str
    script_name: str
    scenarios: List[str]
    tas_model_path: str
    mrsos_model_path: str
    output_root: str
    outputs_written: List[str]
    notes: List[str]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def get_grid():
    grid = RozviGrid()
    tl = grid.pixel_bounds(0)
    br = grid.pixel_bounds(grid.width * grid.height - 1)
    west = float(tl[0])
    north = float(tl[3])
    east = float(br[2])
    south = float(br[1])
    transform = from_bounds(west, south, east, north, grid.width, grid.height)
    return grid, transform


def save_raster(out_path: Path, arr: np.ndarray, transform, crs: str = "EPSG:4326") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def month_str_from_time(t) -> str:
    ts = np.datetime_as_string(t, unit="M")
    return ts.replace("-", "")


def open_var_file(scenario: str, var: str, pattern: str) -> xr.DataArray:
    files = list((SCENARIO_ROOT / scenario / var).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for {scenario}/{var}")
    ds = xr.open_dataset(files[0], engine="netcdf4")
    return ds[var]


def resample_cmip6_slice_to_rozvi(da: xr.DataArray, dst_shape, dst_transform) -> np.ndarray:
    arr = da.values.astype(np.float32)

    lon_vals = da.lon.values
    lat_vals = da.lat.values

    west = float(lon_vals.min())
    east = float(lon_vals.max())
    south = float(lat_vals.min())
    north = float(lat_vals.max())

    src_transform = from_bounds(west, south, east, north, da.sizes["lon"], da.sizes["lat"])

    dst = np.full(dst_shape, np.nan, dtype=np.float32)

    rasterio.warp.reproject(
    source=arr,
    destination=dst,
    src_transform=src_transform,
    src_crs="EPSG:4326",
    dst_transform=dst_transform,
    dst_crs="EPSG:4326",
    src_nodata=np.nan,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear,
)
    return dst

def main() -> None:
    run_id = f"cmip6_tempsoil_correction_{utc_stamp()}"
    manifest_path = MANIFESTS_DIR / f"{run_id}.json"

    tas_model = joblib.load(TAS_MODEL_PATH)
    mrsos_model = joblib.load(MRSOS_MODEL_PATH)

    grid, transform = get_grid()
    shape = (grid.height, grid.width)

    outputs_written: List[str] = []

    for scenario in SCENARIOS:
        print(f"\nProcessing scenario: {scenario}")

        tas_da = open_var_file(scenario, "tas", TAS_FILE_GLOB)
        mrsos_da = open_var_file(scenario, "mrsos", MRSOS_FILE_GLOB)

        if tas_da.sizes["time"] != mrsos_da.sizes["time"]:
            raise ValueError(f"Time mismatch in {scenario}")

        for i in range(tas_da.sizes["time"]):
            yyyymm = month_str_from_time(tas_da.time.values[i])

            tas_out = OUT_ROOT / scenario / "tas" / f"tas_corrected_{yyyymm}.tif"
            sm_out = OUT_ROOT / scenario / "sm" / f"sm_corrected_{yyyymm}.tif"

            if tas_out.exists() and sm_out.exists():
                print(f"Skipping existing month: {scenario} {yyyymm}")
                continue

            tas_arr = resample_cmip6_slice_to_rozvi(tas_da.isel(time=i), shape, transform)
            mrsos_arr = resample_cmip6_slice_to_rozvi(mrsos_da.isel(time=i), shape, transform)

            # convert mrsos kg/m² -> volumetric fraction for 0–10 cm layer
            mrsos_vol = mrsos_arr / (1000.0 * 0.1)

            tas_flat = tas_arr.reshape(-1)
            sm_flat = mrsos_vol.reshape(-1)

            tas_valid = np.isfinite(tas_flat)
            sm_valid = np.isfinite(sm_flat)

            tas_pred = np.full_like(tas_flat, np.nan, dtype=np.float32)
            sm_pred = np.full_like(sm_flat, np.nan, dtype=np.float32)

            if tas_valid.any():
                tas_pred[tas_valid] = tas_model.predict(
                    tas_flat[tas_valid].reshape(-1, 1)
                ).astype(np.float32)

            if sm_valid.any():
                sm_pred[sm_valid] = mrsos_model.predict(
                    sm_flat[sm_valid].reshape(-1, 1)
                ).astype(np.float32)

            tas_corrected = tas_pred.reshape(shape)
            sm_corrected = sm_pred.reshape(shape)

            sm_corrected = np.clip(sm_corrected, 0.0, 1.0)

            if not tas_out.exists():
                save_raster(tas_out, tas_corrected, transform)
                outputs_written.append(str(tas_out))
                print(f"Saved: {tas_out}")

            if not sm_out.exists():
                save_raster(sm_out, sm_corrected, transform)
                outputs_written.append(str(sm_out))
                print(f"Saved: {sm_out}")

    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        script_name="cmip6_tempsoil_correction.py",
        scenarios=SCENARIOS,
        tas_model_path=str(TAS_MODEL_PATH),
        mrsos_model_path=str(MRSOS_MODEL_PATH),
        output_root=str(OUT_ROOT),
        outputs_written=outputs_written,
        notes=[
            "Applies trained CMIP6->ERA5 temperature correction.",
            "Applies trained CMIP6->corrected-soil-moisture correction.",
            "CMIP6 mrsos converted from kg/m² to volumetric fraction before prediction.",
            "Outputs written on RozviGrid.",
        ],
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    print("\nDone.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()