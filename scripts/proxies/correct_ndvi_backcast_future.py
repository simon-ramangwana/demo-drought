from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought").resolve()

DATA_DIR = PROJECT_ROOT / "data"
RASTERS_DIR = PROJECT_ROOT / "rasters"

NDVI_OBS_DIR = DATA_DIR / "real_observations" / "ndvi"

ERA5_DIR = DATA_DIR / "simulated" / "era5_land"
ERA5_T2M_DIR = ERA5_DIR / "t2m"
ERA5_D2M_DIR = ERA5_DIR / "d2m"
ERA5_SM_DIR = ERA5_DIR / "sm"
ERA5_PET_DIR = ERA5_DIR / "pet"

CMIP6_CORRECTED_DIR = RASTERS_DIR / "corrected" / "cmip6"

ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

OUTPUT_DIR = RASTERS_DIR / "corrected" / "proxies" / "ndvi"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID = f"correct_ndvi_backcast_future_{RUN_TS}"

MODEL_PATH = MODELS_DIR / "ndvi_proxy_model_20260322T130508Z.joblib"
ERA5_TEMPLATE_PATH = ERA5_T2M_DIR / "t2m_201412.tif"

BACKCAST_START = "198109"
BACKCAST_END = "200001"

OBS_START = "200002"
OBS_END = "202602"

FUTURE_START = "202601"
FUTURE_END = "205012"

SCENARIOS = ["ssp245", "ssp370", "ssp585"]

MANIFEST_PATH = MANIFESTS_DIR / f"{RUN_ID}_manifest.json"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def extract_yyyymm(path: Path):
    stem = path.stem
    for token in stem.split("_"):
        if token.isdigit() and len(token) == 6:
            return token
    return None


def index_by_yyyymm(folder: Path, pattern: str = "*.tif"):
    out = {}
    if not folder.exists():
        return out
    for fp in sorted(folder.glob(pattern)):
        ym = extract_yyyymm(fp)
        if ym is not None:
            out[ym] = fp
    return out


def read_array(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def normalize_pet(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), -1.0 * arr, np.nan).astype(np.float32)


def clean_ndvi(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.where(
        np.isfinite(arr) & (arr >= -0.2) & (arr <= 1.0),
        arr,
        np.nan
    ).astype(np.float32)


def clamp_ndvi(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.where(np.isfinite(arr), np.clip(arr, -0.2, 1.0), np.nan).astype(np.float32)


def resample_ndvi_to_template(
    ndvi_path: Path,
    template_shape,
    template_transform,
    template_crs,
) -> np.ndarray:
    with rasterio.open(ndvi_path) as src:
        ndvi_src = clean_ndvi(src.read(1).astype(np.float32))
        out = np.full(template_shape, np.nan, dtype=np.float32)

        reproject(
            source=ndvi_src,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_transform,
            dst_crs=template_crs,
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    return out.astype(np.float32)


def regrid_raster_to_template(
    src_path: Path,
    template_shape,
    template_transform,
    template_crs,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        out = np.full(template_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_transform,
            dst_crs=template_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )

    return out.astype(np.float32)


def write_raster(out_path: Path, arr: np.ndarray, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------
# LOAD MODEL + TEMPLATE
# ---------------------------------------------------------------------
print("=" * 60)
print("CORRECT NDVI BACKCAST + FUTURE")
print("=" * 60)
print(f"Run ID      : {RUN_ID}")
print(f"Model path  : {MODEL_PATH}")
print(f"Output dir  : {OUTPUT_DIR}")

model = joblib.load(MODEL_PATH)

with rasterio.open(ERA5_TEMPLATE_PATH) as ref:
    template_meta = ref.meta.copy()
    template_transform = ref.transform
    template_crs = ref.crs
    template_shape = (ref.height, ref.width)

template_meta.update(
    dtype="float32",
    count=1,
    compress="lzw",
)


# ---------------------------------------------------------------------
# INDEX INPUTS
# ---------------------------------------------------------------------
ndvi_obs_index = index_by_yyyymm(NDVI_OBS_DIR)

t2m_hist_index = index_by_yyyymm(ERA5_T2M_DIR)
d2m_hist_index = index_by_yyyymm(ERA5_D2M_DIR)
sm_hist_index = index_by_yyyymm(ERA5_SM_DIR)
pet_hist_index = index_by_yyyymm(ERA5_PET_DIR)

future_indexes = {}
for scenario in SCENARIOS:
    future_indexes[scenario] = {
        "t2m": index_by_yyyymm(CMIP6_CORRECTED_DIR / scenario / "tas"),
        "d2m": index_by_yyyymm(CMIP6_CORRECTED_DIR / scenario / "d2m"),
        "sm": index_by_yyyymm(CMIP6_CORRECTED_DIR / scenario / "sm"),
        "pet": index_by_yyyymm(CMIP6_CORRECTED_DIR / scenario / "pet"),
    }


# ---------------------------------------------------------------------
# 1) BACKCAST NDVI
# ---------------------------------------------------------------------
backcast_months = sorted(
    ym for ym in set(t2m_hist_index) & set(d2m_hist_index) & set(sm_hist_index) & set(pet_hist_index)
    if BACKCAST_START <= ym <= BACKCAST_END
)

backcast_written = 0

print("\n" + "-" * 60)
print("BACKCAST NDVI")
print("-" * 60)
print(f"Months to write: {len(backcast_months)}")

for ym in backcast_months:
    out_path = OUTPUT_DIR / "historical" / f"ndvi_proxy_{ym}.tif"

    t2m = read_array(t2m_hist_index[ym])
    d2m = read_array(d2m_hist_index[ym])
    sm = read_array(sm_hist_index[ym])
    pet = normalize_pet(read_array(pet_hist_index[ym]))

    flat_t2m = t2m.reshape(-1)
    flat_d2m = d2m.reshape(-1)
    flat_sm = sm.reshape(-1)
    flat_pet = pet.reshape(-1)

    valid_mask = (
        np.isfinite(flat_t2m) &
        np.isfinite(flat_d2m) &
        np.isfinite(flat_sm) &
        np.isfinite(flat_pet)
    )

    pred_flat = np.full(flat_t2m.shape, np.nan, dtype=np.float32)

    if valid_mask.any():
        X = pd.DataFrame({
            "t2m": flat_t2m[valid_mask].astype(np.float32),
            "d2m": flat_d2m[valid_mask].astype(np.float32),
            "sm": flat_sm[valid_mask].astype(np.float32),
            "pet": flat_pet[valid_mask].astype(np.float32),
        })
        pred_flat[valid_mask] = model.predict(X).astype(np.float32)

    pred = clamp_ndvi(pred_flat.reshape(template_shape))
    write_raster(out_path, pred, template_meta)
    backcast_written += 1

    print(
        f"Wrote {out_path.name} | "
        f"min={np.nanmin(pred):.4f} "
        f"max={np.nanmax(pred):.4f} "
        f"mean={np.nanmean(pred):.4f}"
    )


# ---------------------------------------------------------------------
# 2) OBSERVED NDVI COPY/RESAMPLE
# ---------------------------------------------------------------------
obs_months = sorted(
    ym for ym in ndvi_obs_index
    if OBS_START <= ym <= OBS_END
)

obs_written = 0

print("\n" + "-" * 60)
print("OBSERVED NDVI COPY/RESAMPLE")
print("-" * 60)
print(f"Months to write: {len(obs_months)}")

for ym in obs_months:
    out_path = OUTPUT_DIR / "historical" / f"ndvi_observed_{ym}.tif"

    arr = resample_ndvi_to_template(
        ndvi_obs_index[ym],
        template_shape=template_shape,
        template_transform=template_transform,
        template_crs=template_crs,
    )

    arr = clamp_ndvi(arr)
    write_raster(out_path, arr, template_meta)
    obs_written += 1

    print(
        f"Wrote {out_path.name} | "
        f"min={np.nanmin(arr):.4f} "
        f"max={np.nanmax(arr):.4f} "
        f"mean={np.nanmean(arr):.4f}"
    )


# ---------------------------------------------------------------------
# 3) FUTURE NDVI BY SCENARIO
# ---------------------------------------------------------------------
future_written = {}

for scenario in SCENARIOS:
    print("\n" + "-" * 60)
    print(f"FUTURE NDVI: {scenario}")
    print("-" * 60)

    idx = future_indexes[scenario]

    months = sorted(
        ym for ym in set(idx["t2m"]) & set(idx["d2m"]) & set(idx["sm"]) & set(idx["pet"])
        if FUTURE_START <= ym <= FUTURE_END
    )

    count = 0

    for ym in months:
        out_path = OUTPUT_DIR / scenario / f"ndvi_proxy_{ym}.tif"

        t2m = regrid_raster_to_template(
            idx["t2m"][ym],
            template_shape=template_shape,
            template_transform=template_transform,
            template_crs=template_crs,
        )

        d2m = regrid_raster_to_template(
            idx["d2m"][ym],
            template_shape=template_shape,
            template_transform=template_transform,
            template_crs=template_crs,
        )

        sm = regrid_raster_to_template(
            idx["sm"][ym],
            template_shape=template_shape,
            template_transform=template_transform,
            template_crs=template_crs,
        )

        pet = regrid_raster_to_template(
            idx["pet"][ym],
            template_shape=template_shape,
            template_transform=template_transform,
            template_crs=template_crs,
        )

        flat_t2m = t2m.reshape(-1)
        flat_d2m = d2m.reshape(-1)
        flat_sm = sm.reshape(-1)
        flat_pet = pet.reshape(-1)

        valid_mask = (
            np.isfinite(flat_t2m) &
            np.isfinite(flat_d2m) &
            np.isfinite(flat_sm) &
            np.isfinite(flat_pet)
        )

        pred_flat = np.full(flat_t2m.shape, np.nan, dtype=np.float32)

        if valid_mask.any():
            X = pd.DataFrame({
                "t2m": flat_t2m[valid_mask].astype(np.float32),
                "d2m": flat_d2m[valid_mask].astype(np.float32),
                "sm": flat_sm[valid_mask].astype(np.float32),
                "pet": flat_pet[valid_mask].astype(np.float32),
            })

            pred_flat[valid_mask] = model.predict(X).astype(np.float32)

        pred = clamp_ndvi(pred_flat.reshape(template_shape))
        write_raster(out_path, pred, template_meta)
        count += 1

        print(
            f"Wrote {out_path.name} | "
            f"min={np.nanmin(pred):.4f} "
            f"max={np.nanmax(pred):.4f} "
            f"mean={np.nanmean(pred):.4f}"
        )

    future_written[scenario] = count


# ---------------------------------------------------------------------
# SAVE MANIFEST
# ---------------------------------------------------------------------
manifest = {
    "run_id": RUN_ID,
    "model_path": str(MODEL_PATH),
    "output_root": str(OUTPUT_DIR),
    "template_path": str(ERA5_TEMPLATE_PATH),
    "periods": {
        "backcast": [BACKCAST_START, BACKCAST_END],
        "observed": [OBS_START, OBS_END],
        "future": [FUTURE_START, FUTURE_END],
    },
    "written": {
        "backcast_months": backcast_written,
        "observed_months": obs_written,
        "future_months_by_scenario": future_written,
    },
}

save_json(manifest, MANIFEST_PATH)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Manifest saved: {MANIFEST_PATH}")