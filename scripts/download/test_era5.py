import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-land",
    {
        "variable": "2m_temperature",
        "year": "2020",
        "month": "01",
        "day": "01",
        "time": "12:00",
        "data_format": "netcdf",
    },
    r"C:\Projects\Infer RozviDrought\data\era5_land\test.nc"
)