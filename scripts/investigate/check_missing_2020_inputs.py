from pathlib import Path

PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought")

folders = {
    "pet_nc": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "pet",
    "t2m_nc": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "t2m",
    "d2m_nc": PROJECT_ROOT / "data" / "simulated" / "era5_land" / "d2m",
}

for name, folder in folders.items():
    nc_files = list(folder.glob("*2020*.nc"))
    print(name, "2020 nc files:", len(nc_files))