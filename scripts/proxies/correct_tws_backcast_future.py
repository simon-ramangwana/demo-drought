# ---------------------------------------------------------------------
# correct_tws_backcast_future.py
# Build continuous TWS proxy stack:
# - backcast      : 198109 -> 200203
# - observed      : 200204 -> 202409
# - bridge proxy  : 202410 -> 202512
# - future proxy  : 202601 -> 205012
#
# NOTE:
# - This version OVERWRITES existing output files.
# ---------------------------------------------------------------------

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# ROOT PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought").resolve()

DATA_DIR = PROJECT_ROOT / "data"
RASTERS_DIR = PROJECT_ROOT / "rasters"

ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

ERA5_DIR = DATA_DIR / "simulated" / "era5_land"
TWS_OBS_DIR = DATA_DIR / "real_observations" / "tws"

CMIP6_CORRECTED_DIR = RASTERS_DIR / "corrected" / "cmip6"
OUTPUT_DIR = RASTERS_DIR / "corrected" / "proxies" / "tws"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "historical").mkdir(parents=True, exist_ok=True)
for scenario_name in ["ssp245", "ssp370", "ssp585"]:
    (OUTPUT_DIR / scenario_name).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID = f"correct_tws_backcast_future_{RUN_TS}"

SCENARIOS = ["ssp245", "ssp370", "ssp585"]

BACKCAST_START = "198109"
BACKCAST_END = "200203"

OBS_START = "200204"
OBS_END = "202409"

BRIDGE_START = "202410"
BRIDGE_END = "202512"

FUTURE_START = "202601"
FUTURE_END = "205012"

MODEL_PATH = MODELS_DIR / "tws_proxy_model_20260322T135040Z.joblib"
MANIFEST_PATH = MANIFESTS_DIR / f"{RUN_ID}_manifest.json"

ERA5_T2M_DIR = ERA5_DIR / "t2m"
ERA5_D2M_DIR = ERA5_DIR / "d2m"
ERA5_SM_DIR = ERA5_DIR / "sm"
ERA5_PET_DIR = ERA5_DIR / "pet"

# Use a stable ERA5 raster as template
ERA5_TEMPLATE_PATH = sorted((ERA5_SM_DIR).glob("*.tif"))[0]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
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


def month_range(start: str, end: str):
    months = []
    y = int(start[:4])
    m = int(start[4:])

    while True:
        ym = f"{y:04d}{m:02d}"
        months.append(ym)

        if ym == end:
            break

        m += 1
        if m > 12:
            m = 1
            y += 1

    return months


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_array(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    return arr


def regrid_raster_to_template(
    src_path: Path,
    template_shape,
    template_transform,
    template_crs,
    resampling=Resampling.bilinear,
):
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nodata = src.nodata

        if src_nodata is not None:
            src_arr = np.where(src_arr == src_nodata, np.nan, src_arr)

        dst_arr = np.full(template_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_transform,
            dst_crs=template_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    return dst_arr


def resample_obs_to_template(
    src_path: Path,
    template_shape,
    template_transform,
    template_crs,
):
    return regrid_raster_to_template(
        src_path=src_path,
        template_shape=template_shape,
        template_transform=template_transform,
        template_crs=template_crs,
        resampling=Resampling.bilinear,
    )


def normalize_pet(arr: np.ndarray):
    # Historical ERA5 PET used negative sign convention in this pipeline.
    # Convert to positive magnitude for modeling consistency.
    return np.where(np.isfinite(arr), -1.0 * arr, np.nan).astype(np.float32)


def clamp_tws(arr: np.ndarray):
    # Keep finite values; clamp only absurd extremes.
    arr = arr.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.clip(arr, -1.0e6, 1.0e6)
    return arr


def write_raster(out_path: Path, arr: np.ndarray, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr.astype(np.float32), 1)


def predict_from_hist_predictors(ym: str, template_shape, template_meta):
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

    pred = clamp_tws(pred_flat.reshape(template_shape))
    return pred


def predict_from_future_predictors(ym: str, scenario: str, idx: dict, template_shape):
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

    pred = clamp_tws(pred_flat.reshape(template_shape))
    return pred


# ---------------------------------------------------------------------
# LOAD MODEL + TEMPLATE
# ---------------------------------------------------------------------
print("=" * 60)
print("CORRECT TWS BACKCAST + BRIDGE + FUTURE")
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
    nodata=np.nan,
)

# ---------------------------------------------------------------------
# INDEX INPUTS
# ---------------------------------------------------------------------
tws_obs_index = index_by_yyyymm(TWS_OBS_DIR)

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
# 1) BACKCAST TWS
# ---------------------------------------------------------------------
backcast_months = sorted(
    ym for ym in set(t2m_hist_index) & set(d2m_hist_index) & set(sm_hist_index) & set(pet_hist_index)
    if BACKCAST_START <= ym <= BACKCAST_END
)

backcast_written = 0

print("\n" + "-" * 60)
print("BACKCAST TWS")
print("-" * 60)
print(f"Months to write: {len(backcast_months)}")

for ym in backcast_months:
    out_path = OUTPUT_DIR / "historical" / f"tws_proxy_{ym}.tif"

    pred = predict_from_hist_predictors(
        ym=ym,
        template_shape=template_shape,
        template_meta=template_meta,
    )

    write_raster(out_path, pred, template_meta)
    backcast_written += 1

    print(
        f"Wrote {out_path.name} | "
        f"min={np.nanmin(pred):.4f} "
        f"max={np.nanmax(pred):.4f} "
        f"mean={np.nanmean(pred):.4f}"
    )

# ---------------------------------------------------------------------
# 2) OBSERVED TWS COPY/RESAMPLE
# ---------------------------------------------------------------------
obs_months = sorted(
    ym for ym in tws_obs_index
    if OBS_START <= ym <= OBS_END
)

obs_written = 0

print("\n" + "-" * 60)
print("OBSERVED TWS COPY/RESAMPLE")
print("-" * 60)
print(f"Months to write: {len(obs_months)}")

for ym in obs_months:
    out_path = OUTPUT_DIR / "historical" / f"tws_observed_{ym}.tif"

    arr = resample_obs_to_template(
        tws_obs_index[ym],
        template_shape=template_shape,
        template_transform=template_transform,
        template_crs=template_crs,
    )

    arr = clamp_tws(arr)
    write_raster(out_path, arr, template_meta)
    obs_written += 1

    print(
        f"Wrote {out_path.name} | "
        f"min={np.nanmin(arr):.4f} "
        f"max={np.nanmax(arr):.4f} "
        f"mean={np.nanmean(arr):.4f}"
    )

# ---------------------------------------------------------------------
# 2B) FILL MISSING HISTORICAL OBSERVED TWS MONTHS (198109 -> 202409)
# ---------------------------------------------------------------------
print("\n" + "-" * 60)
print("FILL MISSING HISTORICAL TWS MONTHS")
print("-" * 60)

expected_hist_months = set(month_range(BACKCAST_START, OBS_END))

existing_hist_months = {
    extract_yyyymm(fp)
    for fp in (OUTPUT_DIR / "historical").glob("*.tif")
    if extract_yyyymm(fp) is not None
}

missing_hist_months = sorted(expected_hist_months - existing_hist_months)

print(f"Missing months detected : {len(missing_hist_months)}")

filled_count = 0

for ym in missing_hist_months:
    if (
        ym not in t2m_hist_index
        or ym not in d2m_hist_index
        or ym not in sm_hist_index
        or ym not in pet_hist_index
    ):
        continue

    out_path = OUTPUT_DIR / "historical" / f"tws_proxy_{ym}.tif"

    pred = predict_from_hist_predictors(
        ym=ym,
        template_shape=template_shape,
        template_meta=template_meta,
    )

    write_raster(out_path, pred, template_meta)
    filled_count += 1

    print(
        f"Filled {ym} | "
        f"min={np.nanmin(pred):.4f} "
        f"max={np.nanmax(pred):.4f} "
        f"mean={np.nanmean(pred):.4f}"
    )

# ---------------------------------------------------------------------
# 2C) BRIDGE TWS FOR 202410 -> 202512 USING HISTORICAL PREDICTORS
# ---------------------------------------------------------------------
print("\n" + "-" * 60)
print("BRIDGE TWS")
print("-" * 60)

bridge_months = sorted(
    ym for ym in set(t2m_hist_index) & set(d2m_hist_index) & set(sm_hist_index) & set(pet_hist_index)
    if BRIDGE_START <= ym <= BRIDGE_END
)

bridge_written = 0

print(f"Months to write: {len(bridge_months)}")

for ym in bridge_months:
    out_path = OUTPUT_DIR / "historical" / f"tws_proxy_{ym}.tif"

    pred = predict_from_hist_predictors(
        ym=ym,
        template_shape=template_shape,
        template_meta=template_meta,
    )

    write_raster(out_path, pred, template_meta)
    bridge_written += 1

    print(
        f"Wrote {out_path.name} | "
        f"min={np.nanmin(pred):.4f} "
        f"max={np.nanmax(pred):.4f} "
        f"mean={np.nanmean(pred):.4f}"
    )

# ---------------------------------------------------------------------
# 3) FUTURE TWS BY SCENARIO
# ---------------------------------------------------------------------
print("\n" + "-" * 60)
print("FUTURE TWS BY SCENARIO")
print("-" * 60)

future_months = month_range(FUTURE_START, FUTURE_END)
future_written = {}

for scenario in SCENARIOS:
    print("\n" + "." * 60)
    print(f"SCENARIO: {scenario}")
    print("." * 60)

    idx = future_indexes[scenario]

    valid_future_months = sorted(
        ym for ym in future_months
        if ym in idx["t2m"]
        and ym in idx["d2m"]
        and ym in idx["sm"]
        and ym in idx["pet"]
    )

    print(f"Months to write: {len(valid_future_months)}")

    count = 0

    for ym in valid_future_months:
        out_path = OUTPUT_DIR / scenario / f"tws_proxy_{ym}.tif"

        pred = predict_from_future_predictors(
            ym=ym,
            scenario=scenario,
            idx=idx,
            template_shape=template_shape,
        )

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
        "bridge": [BRIDGE_START, BRIDGE_END],
        "future": [FUTURE_START, FUTURE_END],
    },
    "written": {
        "backcast_months": backcast_written,
        "observed_months": obs_written,
        "filled_missing_historical_months": filled_count,
        "bridge_months": bridge_written,
        "future_months_by_scenario": future_written,
    },
    "notes": {
        "overwrite_mode": True,
        "historical_proxy_method": "tws predicted from t2m, d2m, sm, pet using trained proxy model",
        "observed_method": "observed TWS resampled to ERA5 inference template",
        "future_method": "tws predicted from future climate inputs by scenario",
    },
}

save_json(manifest, MANIFEST_PATH)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Manifest saved: {MANIFEST_PATH}")