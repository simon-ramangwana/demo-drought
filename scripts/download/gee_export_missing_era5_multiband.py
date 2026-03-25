import re
from pathlib import Path
from datetime import datetime

import ee

# -----------------------------
# CONFIG
# -----------------------------
PROJECT = "august-analyze"
LOCAL_DIR = Path(r"C:\Projects\Infer RozviDrought\data\era5_land")
DRIVE_FOLDER = "august-analyze"
START_YEAR = 1980

# Rozvi bounds: west, south, east, north
WEST = 25.197743719552577
SOUTH = -22.457882102988037
EAST = 33.05800245559839
NORTH = -15.585770179473698

# Existing local single-band files like t2m_202001.nc, d2m_202001.nc, pet_202001.nc
SINGLE_RE = re.compile(r"^(pet|t2m|d2m)_(\d{6})\.nc$", re.IGNORECASE)

# Existing local multiband files already downloaded from Drive, if any
MULTI_RE = re.compile(r"^era5_rozvi_(\d{6})_multiband\.(tif|tiff)$", re.IGNORECASE)

# -----------------------------
# EE INIT
# -----------------------------
ee.Initialize(project=PROJECT)

# -----------------------------
# HELPERS
# -----------------------------
def existing_months(local_dir: Path) -> set[str]:
    months = set()

    for p in local_dir.glob("*.nc"):
        m = SINGLE_RE.match(p.name)
        if m:
            months.add(m.group(2))

    multiband_dir = local_dir / "gee multiband"
    if multiband_dir.exists():
        for p in multiband_dir.glob("*"):
            m = MULTI_RE.match(p.name)
            if m:
                months.add(m.group(1))

    return months


def expected_months(start_year: int) -> list[str]:
    now = datetime.utcnow()
    out = []
    for year in range(start_year, now.year + 1):
        max_month = now.month if year == now.year else 12
        for month in range(1, max_month + 1):
            out.append(f"{year}{month:02d}")
    return out


def month_image(yyyymm: str) -> ee.Image:
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start, end)
        .select([
            "temperature_2m",
            "dewpoint_temperature_2m",
            "potential_evaporation_hourly",
        ], [
            "t2m",
            "d2m",
            "pet",
        ])
    )

    img = col.mean().toFloat()

    # Keep only Rozvi area
    region = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], proj=None, geodesic=False)
    return img.clip(region), region


def create_task(yyyymm: str):
    img, region = month_image(yyyymm)

    description = f"era5_rozvi_{yyyymm}_multiband"
    file_prefix = description

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_prefix,
        region=region,
        crs="EPSG:4326",
        scale=11132,   # approx ERA5-Land native 0.1 degree
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
    )
    return task


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_DIR / "gee multiband").mkdir(parents=True, exist_ok=True)

    have = existing_months(LOCAL_DIR)
    want = expected_months(START_YEAR)
    missing = [m for m in want if m not in have]

    print(f"Existing months: {len(have)}")
    print(f"Expected months: {len(want)}")
    print(f"Missing months : {len(missing)}")

    if not missing:
        print("Nothing to export.")
        return

    for yyyymm in missing:
        try:
            task = create_task(yyyymm)
            task.start()
            print(f"STARTED: era5_rozvi_{yyyymm}_multiband")
        except Exception as e:
            print(f"FAILED : {yyyymm} -> {e}")

    print("All missing-month export tasks submitted.")


if __name__ == "__main__":
    main()