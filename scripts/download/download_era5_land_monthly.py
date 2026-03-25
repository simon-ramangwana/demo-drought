import re
from pathlib import Path
from datetime import datetime

import cdsapi

OUT_DIR = Path(r"C:\Projects\Infer RozviDrought\data\era5_land")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rozvi grid CDS area: [North, West, South, East]
AREA = [-15.585770179473698, 25.197743719552577, -22.457882102988037, 33.05800245559839]

START_YEAR = 1980
NOW = datetime.utcnow()
END_YEAR = NOW.year
END_MONTH = NOW.month

VARIABLES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "potential_evaporation": "pet",
}

FILENAME_RE = re.compile(r"^(pet|t2m|d2m)_(\d{6})\.nc$")


def existing_files(out_dir: Path) -> set[str]:
    found = set()
    for p in out_dir.glob("*.nc"):
        if FILENAME_RE.match(p.name):
            found.add(p.name)
    return found


def expected_items(start_year: int, end_year: int, end_month: int):
    items = []
    for year in range(start_year, end_year + 1):
        month_max = end_month if year == end_year else 12
        for month in range(1, month_max + 1):
            yyyymm = f"{year}{month:02d}"
            for short_name in VARIABLES.values():
                items.append((short_name, yyyymm, f"{short_name}_{yyyymm}.nc"))
    return items


def short_to_cds_var(short_name: str) -> str:
    for cds_var, short in VARIABLES.items():
        if short == short_name:
            return cds_var
    raise ValueError(f"Unknown short variable name: {short_name}")


def download_monthly_mean(c: cdsapi.Client, cds_var: str, year: int, month: int, out_file: Path):
    c.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": cds_var,
            "year": str(year),
            "month": f"{month:02d}",
            "time": "00:00",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": AREA,
        },
        str(out_file),
    )


def main():
    existing = existing_files(OUT_DIR)
    expected = expected_items(START_YEAR, END_YEAR, END_MONTH)

    missing = [(short, yyyymm, fname) for short, yyyymm, fname in expected if fname not in existing]

    print(f"Existing files : {len(existing)}")
    print(f"Expected files : {len(expected)}")
    print(f"Missing files  : {len(missing)}")

    if not missing:
        print("Nothing to download.")
        return

    c = cdsapi.Client()

    for short_name, yyyymm, fname in missing:
        year = int(yyyymm[:4])
        month = int(yyyymm[4:6])
        cds_var = short_to_cds_var(short_name)
        out_file = OUT_DIR / fname

        print(f"Downloading {fname} ...")
        try:
            download_monthly_mean(c, cds_var, year, month, out_file)
            print(f"Saved: {out_file}")
        except Exception as e:
            print(f"Failed: {fname} -> {e}")

    print("Done.")


if __name__ == "__main__":
    main()