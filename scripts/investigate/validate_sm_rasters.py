from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import rasterio


from pathlib import Path

PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought")

# ONLY soil moisture
SIM_DIR = PROJECT_ROOT / "data" / "simulated" / "era5_land" / "sm"
CORR_DIR = PROJECT_ROOT / "rasters" / "corrected" / "sm"

EXPECTED_CRS = "EPSG:4326"
EXPECTED_MIN = 0.0
EXPECTED_MAX = 1.0


def extract_yyyymm(path: Path) -> str | None:
    match = re.search(r"(20\d{2})(0[1-9]|1[0-2])", path.stem)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return None


def index_rasters(folder: Path) -> dict[str, Path]:
    rasters: dict[str, Path] = {}
    for path in sorted(folder.glob("*.tif")):
        yyyymm = extract_yyyymm(path)
        if yyyymm:
            rasters[yyyymm] = path
    return rasters


def read_stats(path: Path) -> dict:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = np.nan

        stats = {
            "path": path,
            "crs": src.crs.to_string() if src.crs else None,
            "shape": (src.height, src.width),
            "transform": src.transform,
            "count": src.count,
            "nodata": nodata,
            "min": float(np.nanmin(arr)) if not np.isnan(arr).all() else np.nan,
            "max": float(np.nanmax(arr)) if not np.isnan(arr).all() else np.nan,
            "mean": float(np.nanmean(arr)) if not np.isnan(arr).all() else np.nan,
            "nan_count": int(np.isnan(arr).sum()),
            "total_cells": int(arr.size),
            "lt_zero": int(np.sum(arr < EXPECTED_MIN)),
            "gt_one": int(np.sum(arr > EXPECTED_MAX)),
        }
        return stats


def compare_pair(sim_path: Path, corr_path: Path) -> dict:
    with rasterio.open(sim_path) as sim_src, rasterio.open(corr_path) as corr_src:
        sim = sim_src.read(1).astype("float32")
        corr = corr_src.read(1).astype("float32")

        if sim_src.nodata is not None:
            sim[sim == sim_src.nodata] = np.nan
        if corr_src.nodata is not None:
            corr[corr == corr_src.nodata] = np.nan

        if sim.shape != corr.shape:
            return {
                "shape_match": False,
                "sim_shape": sim.shape,
                "corr_shape": corr.shape,
                "mean_diff": np.nan,
                "abs_mean_diff": np.nan,
                "changed_cells": 0,
                "valid_cells": 0,
            }

        valid = ~np.isnan(sim) & ~np.isnan(corr)
        if not np.any(valid):
            return {
                "shape_match": True,
                "sim_shape": sim.shape,
                "corr_shape": corr.shape,
                "mean_diff": np.nan,
                "abs_mean_diff": np.nan,
                "changed_cells": 0,
                "valid_cells": 0,
            }

        diff = corr[valid] - sim[valid]
        changed_cells = int(np.sum(np.abs(diff) > 1e-9))

        return {
            "shape_match": True,
            "sim_shape": sim.shape,
            "corr_shape": corr.shape,
            "mean_diff": float(np.mean(diff)),
            "abs_mean_diff": float(np.mean(np.abs(diff))),
            "changed_cells": changed_cells,
            "valid_cells": int(valid.sum()),
        }


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    if not SIM_DIR.exists():
        print(f"Missing simulated folder: {SIM_DIR}")
        return 1

    if not CORR_DIR.exists():
        print(f"Missing corrected folder: {CORR_DIR}")
        return 1

    sim_files = index_rasters(SIM_DIR)
    corr_files = index_rasters(CORR_DIR)

    print_header("1. FILE COMPLETENESS")
    print(f"Simulated files found : {len(sim_files)}")
    print(f"Corrected files found : {len(corr_files)}")

    sim_months = set(sim_files)
    corr_months = set(corr_files)

    missing_in_corr = sorted(sim_months - corr_months)
    extra_in_corr = sorted(corr_months - sim_months)
    common_months = sorted(sim_months & corr_months)

    print(f"Common months       : {len(common_months)}")
    print(f"Missing corrected   : {len(missing_in_corr)}")
    print(f"Extra corrected     : {len(extra_in_corr)}")

    if missing_in_corr:
        print("\nMonths missing in corrected:")
        for m in missing_in_corr[:20]:
            print(f"  - {m}")

    if extra_in_corr:
        print("\nExtra months in corrected:")
        for m in extra_in_corr[:20]:
            print(f"  - {m}")

    if not common_months:
        print("\nNo matching months found. Check filenames.")
        return 1

    print_header("2. STRUCTURE + VALUE CHECKS")
    failed_months = []
    changed_summary = []

    for month in common_months:
        sim_path = sim_files[month]
        corr_path = corr_files[month]

        sim_stats = read_stats(sim_path)
        corr_stats = read_stats(corr_path)
        pair_stats = compare_pair(sim_path, corr_path)

        crs_ok = corr_stats["crs"] == EXPECTED_CRS
        shape_ok = pair_stats["shape_match"]
        range_ok = corr_stats["lt_zero"] == 0 and corr_stats["gt_one"] == 0
        changed_ok = pair_stats["changed_cells"] > 0

        status = "OK" if all([crs_ok, shape_ok, range_ok, changed_ok]) else "CHECK"

        print(
            f"{month} | {status} | "
            f"CRS={corr_stats['crs']} | "
            f"shape={corr_stats['shape']} | "
            f"min={corr_stats['min']:.4f} | "
            f"max={corr_stats['max']:.4f} | "
            f"mean={corr_stats['mean']:.4f} | "
            f"mean_diff={pair_stats['mean_diff']:.6f}"
        )

        changed_summary.append((month, pair_stats["mean_diff"], pair_stats["abs_mean_diff"]))

        if not all([crs_ok, shape_ok, range_ok, changed_ok]):
            failed_months.append(
                {
                    "month": month,
                    "crs_ok": crs_ok,
                    "shape_ok": shape_ok,
                    "range_ok": range_ok,
                    "changed_ok": changed_ok,
                }
            )

    print_header("3. SUMMARY")
    if failed_months:
        print(f"Months needing attention: {len(failed_months)}")
        for item in failed_months[:20]:
            print(
                f"- {item['month']} | "
                f"crs_ok={item['crs_ok']} | "
                f"shape_ok={item['shape_ok']} | "
                f"range_ok={item['range_ok']} | "
                f"changed_ok={item['changed_ok']}"
            )
    else:
        print("All matched months passed the structural and value checks.")

    mean_diffs = [x[1] for x in changed_summary if not np.isnan(x[1])]
    abs_mean_diffs = [x[2] for x in changed_summary if not np.isnan(x[2])]

    if mean_diffs:
        print(f"\nAverage monthly mean_diff     : {float(np.mean(mean_diffs)):.6f}")
        print(f"Average monthly abs_mean_diff : {float(np.mean(abs_mean_diffs)):.6f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())