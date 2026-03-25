from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

from rozvidrought_datasets.grid import RozviGrid


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought")

SIMULATED_DIRS = {
    "pet_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "pet",
    "t2m_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "t2m",
    "d2m_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "d2m",
    "sm_sim": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "sm",
}

OBSERVED_DIRS = {
    "ndvi_obs": PROJECT_ROOT / "data" / "real_observations" / "ndvi",
    "tws_obs": PROJECT_ROOT / "data" / "real_observations" / "tws",
    "sm_obs": PROJECT_ROOT / "data" / "real_observations" / "sm",
}

TRAINING_DIR = PROJECT_ROOT / "data" / "artifacts" / "training data"
MODELS_DIR = PROJECT_ROOT / "data" / "artifacts" / "models"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "manifests"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRE_ALL_SIMULATED = True
KEEP_ROWS_WITHOUT_OBS = True
PARQUET_COMPRESSION = "zstd"

ALL_COLUMNS = [
    "pixel_id", "yyyymm",
    "pet_sim", "t2m_sim", "d2m_sim", "sm_sim",
    "ndvi_obs", "tws_obs", "sm_obs",
]

FILE_RE = re.compile(r"^(?P<var>[A-Za-z0-9_]+)_(?P<yyyymm>\d{6})\.tif$", re.IGNORECASE)


@dataclass
class RunManifest:
    run_id: str
    created_at_utc: str
    script_name: str
    output_parquet: str
    output_manifest: str
    rozvi_width: int
    rozvi_height: int
    rozvi_bounds: Tuple[float, float, float, float]
    months_written: List[str]
    source_directories: Dict[str, str]
    source_files_used: Dict[str, List[str]]
    notes: List[str]


def utc_now_stamp() -> str:
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


def build_month_index(source_dirs: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
    return {name: discover_files(folder) for name, folder in source_dirs.items()}


def intersect_months(index: Dict[str, Dict[str, Path]]) -> List[str]:
    if not index:
        return []
    month_sets = [set(v.keys()) for v in index.values()]
    if not month_sets:
        return []
    return sorted(set.intersection(*month_sets))


def union_months(index: Dict[str, Dict[str, Path]]) -> List[str]:
    out = set()
    for d in index.values():
        out.update(d.keys())
    return sorted(out)


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


def month_to_arrays(
    yyyymm: str,
    grid: RozviGrid,
    target_transform,
    target_crs: str,
    all_indexes: Dict[str, Dict[str, Path]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    height = grid.height
    width = grid.width

    arrays: Dict[str, np.ndarray] = {}
    files_used: Dict[str, str] = {}

    for var_name, month_files in all_indexes.items():
        src_path = month_files.get(yyyymm)
        if src_path is None:
            continue

        dst = np.full((height, width), np.nan, dtype=np.float32)

        with rasterio.open(src_path) as src:
            src_arr = src.read(1).astype(np.float32)
            src_nodata = src.nodata

            if src_nodata is not None:
                src_arr = np.where(src_arr == src_nodata, np.nan, src_arr)

            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

        arrays[var_name] = dst
        files_used[var_name] = str(src_path)

    return arrays, files_used


def arrays_to_month_df(
    yyyymm: str,
    grid: RozviGrid,
    arrays: Dict[str, np.ndarray],
) -> pd.DataFrame:
    n_pixels = grid.width * grid.height
    data = {
        "pixel_id": np.arange(n_pixels, dtype=np.int64),
        "yyyymm": np.full(n_pixels, int(yyyymm), dtype=np.int32),
    }

    for var_name, arr in arrays.items():
        data[var_name] = arr.reshape(-1).astype(np.float32)

    for col in ALL_COLUMNS:
        if col not in data:
            if col in ("pixel_id", "yyyymm"):
                continue
            data[col] = np.full(n_pixels, np.nan, dtype=np.float32)

    df = pd.DataFrame(data)
    df = df[ALL_COLUMNS]
    return df


def main():
    run_id = f"bias_dataset_{utc_now_stamp()}"
    output_parquet = TRAINING_DIR / f"{run_id}.parquet"
    output_manifest = MANIFESTS_DIR / f"{run_id}.json"

    grid = RozviGrid()
    bounds = get_rozvi_global_bounds(grid)
    transform = get_rozvi_transform(grid, bounds)
    target_crs = "EPSG:4326"

    sim_index = build_month_index(SIMULATED_DIRS)
    obs_index = build_month_index(OBSERVED_DIRS)
    all_index = {**sim_index, **obs_index}

    print("Simulated folders:")
    for k, v in SIMULATED_DIRS.items():
        print(f"  {k}: {v} -> {len(sim_index[k])} files")

    print("Observed folders:")
    for k, v in OBSERVED_DIRS.items():
        print(f"  {k}: {v} -> {len(obs_index[k])} files")

    sim_months = intersect_months(sim_index) if REQUIRE_ALL_SIMULATED else union_months(sim_index)
    obs_months = union_months(obs_index)
    candidate_months = sim_months if KEEP_ROWS_WITHOUT_OBS else sorted(set(sim_months) & set(obs_months))

    print(f"Simulated common months: {len(sim_months)}")
    print(f"Observed union months  : {len(obs_months)}")
    print(f"Candidate months      : {len(candidate_months)}")

    if not candidate_months:
        raise RuntimeError("No candidate months found. Check source folders and filenames.")

    writer: Optional[pq.ParquetWriter] = None
    months_written: List[str] = []
    source_files_used: Dict[str, List[str]] = {k: [] for k in all_index.keys()}

    try:
        for yyyymm in candidate_months:
            arrays, files_used = month_to_arrays(
                yyyymm=yyyymm,
                grid=grid,
                target_transform=transform,
                target_crs=target_crs,
                all_indexes=all_index,
            )

            missing_sim = [k for k in SIMULATED_DIRS.keys() if k not in arrays]
            if REQUIRE_ALL_SIMULATED and missing_sim:
                print(f"Skipping {yyyymm}: missing simulated vars {missing_sim}")
                continue

            df = arrays_to_month_df(yyyymm=yyyymm, grid=grid, arrays=arrays)

            if not KEEP_ROWS_WITHOUT_OBS:
                obs_cols = list(OBSERVED_DIRS.keys())
                if obs_cols:
                    df = df.dropna(subset=obs_cols, how="all")

            if df.empty:
                print(f"Skipping {yyyymm}: empty dataframe after filtering")
                continue

            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    where=str(output_parquet),
                    schema=table.schema,
                    compression=PARQUET_COMPRESSION,
                )

            writer.write_table(table)
            months_written.append(yyyymm)

            for var_name, path_str in files_used.items():
                source_files_used[var_name].append(path_str)

            print(f"Wrote month {yyyymm} with {len(df):,} rows")

    finally:
        if writer is not None:
            writer.close()

    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        script_name="dataset_builder.py",
        output_parquet=str(output_parquet),
        output_manifest=str(output_manifest),
        rozvi_width=int(grid.width),
        rozvi_height=int(grid.height),
        rozvi_bounds=bounds,
        months_written=months_written,
        source_directories={**{k: str(v) for k, v in SIMULATED_DIRS.items()},
                            **{k: str(v) for k, v in OBSERVED_DIRS.items()}},
        source_files_used=source_files_used,
        notes=[
            "Target grid is RozviGrid in EPSG:4326.",
            "All simulated variables are required for a month to be written.",
            "Observed variables may be missing for some months.",
            "Rasters are resampled to the Rozvi grid before tabularization.",
        ],
    )

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    print("\nDone.")
    print(f"Parquet : {output_parquet}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()