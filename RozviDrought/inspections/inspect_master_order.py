import pandas as pd

MASTER_PATH = r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_20260323T112832Z.parquet"  # change if needed

df = pd.read_parquet(MASTER_PATH, columns=["scenario", "yyyymm"])

print("TOTAL ROWS:", len(df))
print("GRID PIXELS:", 5600)
print("ROWS % 5600:", len(df) % 5600)

counts = (
    df.groupby(["scenario", "yyyymm"])
      .size()
      .reset_index(name="n")
)

print("\nGROUP SIZE SUMMARY:")
print(counts["n"].describe())

print("\nUNIQUE GROUP SIZES:")
print(sorted(counts["n"].unique().tolist())[:10])

print("\nFIRST 10 GROUPS:")
print(counts.head(10))

bad = counts[counts["n"] != 5600]
print("\nGROUPS NOT EQUAL TO 5600:", len(bad))
if len(bad) > 0:
    print(bad.head(20))