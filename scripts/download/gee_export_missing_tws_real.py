import re
from pathlib import Path
import ee

PROJECT = "august-analyze"
LOCAL_DIR = Path(r"C:\Projects\Infer RozviDrought\data\real_observations\tws")
DRIVE_FOLDER = "august-analyze"

# GRACE availability in this GEE dataset
START_YEAR = 2002
START_MONTH = 4
END_YEAR = 2024
END_MONTH = 9

# Rozvi bounds: west, south, east, north
WEST = 25.197743719552577
SOUTH = -22.457882102988037
EAST = 33.05800245559839
NORTH = -15.585770179473698

FILE_RE = re.compile(r"^tws_(\d{6})\.tif$", re.IGNORECASE)

ee.Initialize(project=PROJECT)

def existing_months(local_dir: Path) -> set[str]:
    local_dir.mkdir(parents=True, exist_ok=True)
    months = set()
    for p in local_dir.glob("*.tif"):
        m = FILE_RE.match(p.name)
        if m:
            months.add(m.group(1))
    return months

def expected_months():
    out = []
    for year in range(START_YEAR, END_YEAR + 1):
        m1 = START_MONTH if year == START_YEAR else 1
        m2 = END_MONTH if year == END_YEAR else 12
        for month in range(m1, m2 + 1):
            out.append(f"{year}{month:02d}")
    return out

def month_image(yyyymm: str):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    region = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], proj=None, geodesic=False)

    col = (
        ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI")
        .filterDate(start, end)
        .select("lwe_thickness")
    )

    if col.size().getInfo() == 0:
        return None, None

    img = col.first().toFloat().rename("tws").clip(region)
    return img, region

def create_task(yyyymm: str):
    img, region = month_image(yyyymm)
    if img is None:
        print(f"NO DATA: {yyyymm}")
        return None

    desc = f"tws_{yyyymm}"
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=DRIVE_FOLDER,
        fileNamePrefix=desc,
        region=region,
        crs="EPSG:4326",
        scale=55660,  # native pixel size from GEE catalog
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
            print(f"STARTED: tws_{yyyymm}")
        except Exception as e:
            print(f"FAILED : {yyyymm} -> {e}")

if __name__ == "__main__":
    main()