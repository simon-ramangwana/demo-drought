import pandas as pd

MASTER_PATH = r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_long_198001_205012.parquet"
PIXELS_PER_BLOCK = 26775

PIXEL_A = 8705
PIXEL_B = 10778
SCENARIO = "historical"


class MasterInputReader:
    def __init__(self, parquet_path: str, pixels_per_block: int):
        self.parquet_path = parquet_path
        self.pixels_per_block = pixels_per_block
        self._df = None

    def _load(self):
        if self._df is None:
            self._df = pd.read_parquet(self.parquet_path)

    def get_pixel_timeseries(self, scenario: str, pixel_id: int) -> pd.DataFrame:
        self._load()

        df = self._df[self._df["scenario"] == scenario].copy()
        if df.empty:
            raise ValueError(f"No data found for scenario={scenario}")

        rows = []
        for yyyymm, block in df.groupby("yyyymm", sort=True):
            block = block.reset_index(drop=True)
            if len(block) != self.pixels_per_block:
                raise ValueError(f"Unexpected block size for {scenario}-{yyyymm}: {len(block)}")
            rows.append(block.iloc[pixel_id])

        return pd.DataFrame(rows).reset_index(drop=True)


reader = MasterInputReader(
    parquet_path=MASTER_PATH,
    pixels_per_block=PIXELS_PER_BLOCK,
)

ts_a = reader.get_pixel_timeseries(scenario=SCENARIO, pixel_id=PIXEL_A).copy()
ts_b = reader.get_pixel_timeseries(scenario=SCENARIO, pixel_id=PIXEL_B).copy()

cols = ["yyyymm", "t2m", "d2m", "sm", "pet", "ndvi", "tws"]

a = ts_a[cols].rename(columns={c: f"{c}_a" for c in cols if c != "yyyymm"})
b = ts_b[cols].rename(columns={c: f"{c}_b" for c in cols if c != "yyyymm"})

merged = a.merge(b, on="yyyymm", how="inner")

print("\nFIRST 12 ROWS")
print(merged.head(12).to_string(index=False))

print("\nLAST 12 ROWS")
print(merged.tail(12).to_string(index=False))

value_cols = ["t2m", "d2m", "sm", "pet", "ndvi", "tws"]

print("\nIDENTICAL COLUMN CHECK")
for col in value_cols:
    s1 = merged[f"{col}_a"]
    s2 = merged[f"{col}_b"]
    same = s1.equals(s2)
    print(f"{col}: identical={same}")

print("\nDIFFERENCE COUNTS")
for col in value_cols:
    s1 = merged[f"{col}_a"]
    s2 = merged[f"{col}_b"]
    diff_mask = ~((s1 == s2) | (s1.isna() & s2.isna()))
    print(f"{col}: different_rows={int(diff_mask.sum())}")

print("\nSAMPLE DIFFERENCES")
for col in value_cols:
    s1 = merged[f"{col}_a"]
    s2 = merged[f"{col}_b"]
    diff_mask = ~((s1 == s2) | (s1.isna() & s2.isna()))
    diff_rows = merged.loc[diff_mask, ["yyyymm", f"{col}_a", f"{col}_b"]]
    print(f"\n--- {col} ---")
    if diff_rows.empty:
        print("No differences found")
    else:
        print(diff_rows.head(10).to_string(index=False))