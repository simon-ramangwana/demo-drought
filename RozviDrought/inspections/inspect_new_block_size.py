import pandas as pd

MASTER_PATH = r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_long_198001_205012.parquet"

df = pd.read_parquet(MASTER_PATH, columns=["scenario", "yyyymm"])

counts = (
    df.groupby(["scenario", "yyyymm"])
      .size()
      .reset_index(name="n")
)

print("UNIQUE BLOCK SIZES:")
print(sorted(counts["n"].unique()))

print("\nFIRST 10 GROUPS:")
print(counts.head(10))

print("\nTOTAL ROWS:", len(df))