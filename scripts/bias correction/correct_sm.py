from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

from rozvidrought_datasets.grid import RozviGrid


PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought")

# Inputs
SIM_DIRS = {
    "pet_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "pet",
    "t2m_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "t2m",
    "d2m_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "d2m",
    "sm_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "sm",
}
OBS_DIRS = {
    "ndvi_obs": PROJECT_ROOT / "data" / "real_observations" / "ndvi",
    "tws_obs": PROJECT_ROOT / "data" / "real_observations" / "tws",
}

# Model
MODEL_PATH = PROJECT_ROOT / "data" / "artifacts" / "models" / "sm_bias_model_20260319T135227Z.joblib"

# Outputs
OUT_DIR = PROJECT_ROOT / "rasters" / "corrected" / "sm"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "manifests"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

FILE_RE = re.compile(r"^(?P<var>[A-Za-z0-9_]+)_(?P<yyyymm>\d{6})\.tif$", re.IGNORECASE)

FEATURE_COLUMNS = [
    "sm_sim",
    "pet_sim",
    "t2m_sim",
    "d2m_sim",
    "ndvi_obs",
    "tws_obs",
]


@dataclass
class RunManifest:
    run_id: str
    created_at_utc: str
    script_name: str
    model_path: str
    output_dir: str
    months_written: List[str]
    source_directories: Dict[str, str]
    notes: List[str]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def discover_files(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in folder.glob("*.tif"):
        m = FILE_RE.match(p.name)
        if m:
            out[m.group("yyyymm")] = p
    return out


def build_index(source_dirs: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
    return {name: discover_files(folder) for name, folder in source_dirs.items()}


def intersect_months(index: Dict[str, Dict[str, Path]]) -> List[str]:
    if not index:
        return []
    month_sets = [set(v.keys()) for v in index.values()]
    return sorted(set.intersection(*month_sets)) if month_sets else []


def get_rozvi_global_bounds(grid: RozviGrid) -> Tuple[float, float, float, float]:
    tl = grid.pixel_bounds(0)
    br = grid.pixel_bounds(grid.width * grid.height - 1)
    west = float(tl[0])
    north = float(tl[3])
    east = float(br[2])
    south = float(br[1])
    return west, south, east, north


def get_rozvi_transform(grid: RozviGrid, bounds: Tuple[float, float, float, float]):
    west, south, east, north = bounds
    return from_bounds(west, south, east, north, grid.width, grid.height)


def read_resample(src_path: Path, dst_shape: Tuple[int, int], dst_transform, dst_crs: str) -> np.ndarray:
    dst = np.full(dst_shape, np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return dst


def save_raster(out_path: Path, arr: np.ndarray, transform, crs: str):
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


def main():
    run_id = f"correct_sm_{utc_stamp()}"
    manifest_path = MANIFESTS_DIR / f"{run_id}.json"

    model = joblib.load(MODEL_PATH)

    grid = RozviGrid()
    bounds = get_rozvi_global_bounds(grid)
    transform = get_rozvi_transform(grid, bounds)
    crs = "EPSG:4326"
    shape = (grid.height, grid.width)

    all_dirs = {**SIM_DIRS, **OBS_DIRS}
    index = build_index(all_dirs)

    required_for_prediction = ["sm_sim", "pet_sim", "t2m_sim", "d2m_sim"]
    candidate_months = intersect_months({k: index[k] for k in required_for_prediction})

    months_written: List[str] = []

    for yyyymm in candidate_months:
        out_path = OUT_DIR / f"sm_corrected_{yyyymm}.tif"
        if out_path.exists():
            print(f"Skipping existing: {out_path.name}")
            continue

        arrays: Dict[str, np.ndarray] = {}

        for var_name in required_for_prediction:
            arrays[var_name] = read_resample(index[var_name][yyyymm], shape, transform, crs)

        for obs_name in OBS_DIRS.keys():
            src_path = index[obs_name].get(yyyymm)
            if src_path is None:
                arrays[obs_name] = np.full(shape, np.nan, dtype=np.float32)
            else:
                arrays[obs_name] = read_resample(src_path, shape, transform, crs)

        n = shape[0] * shape[1]
        feature_data = {}
        for col in FEATURE_COLUMNS:
            feature_data[col] = arrays[col].reshape(-1).astype(np.float32)

        # median fill like training
        for col in FEATURE_COLUMNS:
            vals = feature_data[col]
            valid = np.isfinite(vals)
            if valid.any():
                fill_value = np.nanmedian(vals)
            else:
                fill_value = 0.0
            vals = np.where(np.isfinite(vals), vals, fill_value)
            feature_data[col] = vals

        X = np.column_stack([feature_data[col] for col in FEATURE_COLUMNS])
        preds = model.predict(X).astype(np.float32)

        corrected = preds.reshape(shape)

        # optional clamp for physical sanity
        corrected = np.clip(corrected, 0.0, 1.0)

        save_raster(out_path, corrected, transform, crs)
        months_written.append(yyyymm)
        print(f"Saved: {out_path}")

    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        script_name="correct_sm.py",
        model_path=str(MODEL_PATH),
        output_dir=str(OUT_DIR),
        months_written=months_written,
        source_directories={k: str(v) for k, v in all_dirs.items()},
        notes=[
            "Corrected soil moisture created from trained XGBoost model.",
            "Rozvi grid used as output grid.",
            "Observed features missing for a month were median-filled at inference time.",
            "Output naming: sm_corrected_yyyymm.tif",
        ],
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    print("Done.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()