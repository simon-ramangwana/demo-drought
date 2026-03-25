# C:\Projects\Infer RozviDrought\scripts\download\gee_export_missing_sm_simulated.py

import re
from pathlib import Path
from datetime import datetime
import ee

PROJECT = "august-analyze"
LOCAL_DIR = Path(r"C:\Projects\Infer RozviDrought\data\simulated\sm")
DRIVE_FOLDER = "august-analyze"
START_YEAR = 1980

# Rozvi bounds: west, south, east, north
WEST = 25.197743719552577
SOUTH = -22.457882102988037
EAST = 33.05800245559839
NORTH = -15.585770179473698

FILE_RE = re.compile(r"^sm_(\d{6})\.tif$", re.IGNORECASE)

ee.Initialize(project=PROJECT)

def existing_months(local_dir: Path) -> set[str]:
    local_dir.mkdir(parents=True, exist_ok=True)
    return {
        m.group(1)
        for p in local_dir.glob("*.tif")
        if (m := FILE_RE.match(p.name))
    }

def expected_months(start_year: int) -> list[str]:
    now = datetime.utcnow()
    out = []
    for year in range(start_year, now.year + 1):
        max_month = now.month if year == now.year else 12
        for month in range(1, max_month + 1):
            out.append(f"{year}{month:02d}")
    return out

def month_image(yyyymm: str):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    region = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], proj=None, geodesic=False)

    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start, end)
        .select("volumetric_soil_water_layer_1")
    )

    if col.size().getInfo() == 0:
        return None, None

    img = col.mean().toFloat().rename("sm").clip(region)
    return img, region

def create_task(yyyymm: str):
    img, region = month_image(yyyymm)
    if img is None:
        print(f"NO DATA: {yyyymm}")
        return None

    desc = f"sm_{yyyymm}"
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=DRIVE_FOLDER,
        fileNamePrefix=desc,
        region=region,
        crs="EPSG:4326",
        scale=11132,  # approx 0.1 degree ERA5-Land native scale
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
    )
    return task

def main():
    have = existing_months(LOCAL_DIR)
    want = expected_months(START_YEAR)
    missing = [m for m in want if m not in have]

    print(f"Existing months: {len(have)}")
    print(f"Expected months: {len(want)}")
    print(f"Missing months : {len(missing)}")

    for yyyymm in missing:
        try:
            task = create_task(yyyymm)
            if task is None:
                continue
            task.start()
            print(f"STARTED: sm_{yyyymm}")
        except Exception as e:
            print(f"FAILED : {yyyymm} -> {e}")

if __name__ == "__main__":
    main()