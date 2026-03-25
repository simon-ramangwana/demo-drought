import pandas as pd

MASTER_PATH = r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_20260323T112832Z.parquet"  # change if needed

df = pd.read_parquet(MASTER_PATH)

print("\nSHAPE:")
print(df.shape)

print("\nCOLUMNS:")
for c in df.columns:
    print(c)

print("\nDTYPES:")
print(df.dtypes)

print("\nHEAD:")
print(df.head(3))

for col in ["scenario", "yyyymm", "pixel_key"]:
    if col in df.columns:
        print(f"\nVALUE COUNTS: {col}")
        print(df[col].value_counts().head(10))