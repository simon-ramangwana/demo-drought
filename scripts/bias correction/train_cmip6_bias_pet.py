from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought").resolve()

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
MODELS_DIR = ARTIFACTS_DIR / "models"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ID = f"train_cmip6_bias_model_pet_{RUN_TS}"

DATASET_PATH = DATASETS_DIR / "cmip6_pet_training_dataset_20260322T113428Z.parquet"

MODEL_PATH = MODELS_DIR / f"cmip6_bias_model_pet_{RUN_TS}.joblib"
METRICS_PATH = MANIFESTS_DIR / f"{RUN_ID}_metrics.json"

FEATURES = [
    "tas_k",
    "rsds_wm2",
    "huss",
    "pet_proxy",
]
TARGET = "pet_era5"
TIME_COL = "yyyymm"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def month_to_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("-", "", regex=False).astype(int)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
print("=" * 60)
print("TRAIN CMIP6 PET BIAS MODEL")
print("=" * 60)
print(f"Run ID       : {RUN_ID}")
print(f"Dataset path : {DATASET_PATH}")

df = pd.read_parquet(DATASET_PATH)

print(f"Rows loaded  : {len(df)}")
print(f"Columns      : {list(df.columns)}")

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET, TIME_COL]).copy()

print(f"Rows valid   : {len(df)}")

df["_yyyymm_int"] = month_to_int(df[TIME_COL])

# same split style used before
train_df = df[df["_yyyymm_int"] <= 201012].copy()
valid_df = df[(df["_yyyymm_int"] >= 201101) & (df["_yyyymm_int"] <= 201412)].copy()

if train_df.empty or valid_df.empty:
    raise ValueError("Train/validation split is empty. Check yyyymm values in dataset.")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET]

print(f"Train rows   : {len(train_df)}")
print(f"Valid rows   : {len(valid_df)}")
print(f"Train months : {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()}")
print(f"Valid months : {valid_df[TIME_COL].min()} -> {valid_df[TIME_COL].max()}")


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

train_pred = model.predict(X_train)
valid_pred = model.predict(X_valid)

metrics = {
    "run_id": RUN_ID,
    "dataset_path": str(DATASET_PATH),
    "model_path": str(MODEL_PATH),
    "features": FEATURES,
    "target": TARGET,
    "train_rows": int(len(train_df)),
    "valid_rows": int(len(valid_df)),
    "train_month_start": str(train_df[TIME_COL].min()),
    "train_month_end": str(train_df[TIME_COL].max()),
    "valid_month_start": str(valid_df[TIME_COL].min()),
    "valid_month_end": str(valid_df[TIME_COL].max()),
    "train_mae": float(mean_absolute_error(y_train, train_pred)),
    "train_rmse": rmse(y_train, train_pred),
    "train_r2": float(r2_score(y_train, train_pred)),
    "valid_mae": float(mean_absolute_error(y_valid, valid_pred)),
    "valid_rmse": rmse(y_valid, valid_pred),
    "valid_r2": float(r2_score(y_valid, valid_pred)),
}

joblib.dump(model, MODEL_PATH)
save_json(metrics, METRICS_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Model saved   : {MODEL_PATH}")
print(f"Metrics saved : {METRICS_PATH}")
print(f"Valid MAE     : {metrics['valid_mae']:.8f}")
print(f"Valid RMSE    : {metrics['valid_rmse']:.8f}")
print(f"Valid R2      : {metrics['valid_r2']:.4f}")