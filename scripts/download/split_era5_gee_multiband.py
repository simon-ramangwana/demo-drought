from pathlib import Path
import re

import rasterio

MULTI_DIR = Path(r"C:\Projects\Infer RozviDrought\data\era5_land\gee multiband")
OUT_BASE = Path(r"C:\Projects\Infer RozviDrought\data\era5_land")

OUT_DIRS = {
    "pet": OUT_BASE / "pet",
    "t2m": OUT_BASE / "t2m",
    "d2m": OUT_BASE / "d2m",
}

for d in OUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# expects names like: era5_rozvi_202001_multiband.tif
PATTERN = re.compile(r"^era5_rozvi_(\d{6})_multiband\.(tif|tiff)$", re.IGNORECASE)

# band order from the export script:
# 1 = t2m, 2 = d2m, 3 = pet
BANDS = [
    (1, "t2m"),
    (2, "d2m"),
    (3, "pet"),
]

for tif_path in MULTI_DIR.glob("*"):
    m = PATTERN.match(tif_path.name)
    if not m:
        print(f"Skipping unrecognized file: {tif_path.name}")
        continue

    yyyymm = m.group(1)

    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        meta.update(count=1)

        for band_index, short_name in BANDS:
            out_path = OUT_DIRS[short_name] / f"{short_name}_{yyyymm}.tif"

            if out_path.exists():
                print(f"Skipping existing: {out_path}")
                continue

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(src.read(band_index), 1)

            print(f"Saved: {out_path}")

print("Done.")