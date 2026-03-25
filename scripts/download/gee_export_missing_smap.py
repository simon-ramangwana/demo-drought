import re
from pathlib import Path
import ee
from datetime import datetime

PROJECT = "august-analyze"
LOCAL_DIR = Path(r"C:\Projects\Infer RozviDrought\data\real_observations\sm")
DRIVE_FOLDER = "august-analyze"

# SMAP starts in 2015
START_YEAR = 2015
START_MONTH = 4

# Rozvi bounds: west, south, east, north
WEST = 25.197743719552577
SOUTH = -22.457882102988037
EAST = 33.05800245559839
NORTH = -15.585770179473698

FILE_RE = re.compile(r"^sm_(\d{6})\.tif$", re.IGNORECASE)

ee.Initialize(project=PROJECT)

def existing_months(local_dir: Path) -> set[str]:
    local_dir.mkdir(parents=True, exist_ok=True)
    out = set()
    for p in local_dir.glob("*.tif"):
        m = FILE_RE.match(p.name)
        if m:
            out.add(m.group(1))
    return out

def expected_months():
    now = datetime.utcnow()
    end_year = now.year
    end_month = now.month

    out = []
    for year in range(START_YEAR, end_year + 1):
        m1 = START_MONTH if year == START_YEAR else 1
        m2 = end_month if year == end_year else 12
        for month in range(m1, m2 + 1):
            out.append(f"{year}{month:02d}")
    return out

def month_image(yyyymm: str):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    region = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], proj=None, geodesic=False)

    # Split across SMAP collection versions
    if start.millis().lt(ee.Date("2023-12-04").millis()).getInfo():
        col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005").filterDate(start, end)
    else:
        col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006").filterDate(start, end)

    # Use AM + PM soil moisture when present, then monthly mean
    col = col.select(["soil_moisture_am", "soil_moisture_pm"])

    if col.size().getInfo() == 0:
        return None, None

    img = col.mean().reduce(ee.Reducer.mean()).rename("sm").toFloat().clip(region)
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
        scale=9000,
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
    )
    return task

def main():
    have = existing_months(LOCAL_DIR)
    want = expected_months()
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