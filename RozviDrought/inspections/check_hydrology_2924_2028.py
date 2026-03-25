import pandas as pd

MASTER_PATH = r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_long_198001_205012.parquet"

df = pd.read_parquet(
    MASTER_PATH,
    columns=["scenario", "yyyymm", "tws"]
)

start = "202603"
end = "202810"

subset = df[
    (df["yyyymm"] >= start) &
    (df["yyyymm"] <= end)
]

check = (
    subset
    .groupby(["scenario", "yyyymm"])["tws"]
    .apply(lambda s: s.notna().sum())
    .reset_index(name="non_null_tws_pixels")
)

print(check.sort_values(["scenario", "yyyymm"]))