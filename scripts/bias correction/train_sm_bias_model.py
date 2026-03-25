from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(r"C:\Projects\Infer RozviDrought")
TRAINING_DIR = PROJECT_ROOT / "data" / "artifacts" / "training data"
MODELS_DIR = PROJECT_ROOT / "data" / "artifacts" / "models"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "manifests"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATASET = TRAINING_DIR / "bias_dataset_20260319T133647Z.parquet"  # change this to your actual parquet if needed

FEATURE_COLUMNS = [
    "sm_sim",
    "pet_sim",
    "t2m_sim",
    "d2m_sim",
    "ndvi_obs",
    "tws_obs",
]

TARGET_COLUMN = "sm_obs"
GROUP_COLUMN = "yyyymm"

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class TrainManifest:
    run_id: str
    created_at_utc: str
    script_name: str
    training_dataset: str
    model_path: str
    manifest_path: str
    feature_columns: List[str]
    target_column: str
    n_train_rows: int
    n_test_rows: int
    train_months: List[int]
    test_months: List[int]
    mae: float
    rmse: float
    r2: float
    model_params: dict
    notes: List[str]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def find_latest_parquet(training_dir: Path) -> Path:
    files = sorted(training_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {training_dir}")
    return files[-1]


def load_training_data() -> pd.DataFrame:
    parquet_path = TRAINING_DATASET if TRAINING_DATASET.exists() else find_latest_parquet(TRAINING_DIR)
    df = pd.read_parquet(parquet_path)
    df.attrs["training_dataset_path"] = str(parquet_path)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[required].copy()

    # target must exist
    work = work.dropna(subset=[TARGET_COLUMN])

    # fill missing features so schema stays usable
    for col in FEATURE_COLUMNS:
        work[col] = work[col].astype("float32")
        work[col] = work[col].fillna(work[col].median())

    work[TARGET_COLUMN] = work[TARGET_COLUMN].astype("float32")
    work[GROUP_COLUMN] = work[GROUP_COLUMN].astype("int32")

    if work.empty:
        raise ValueError("No rows left after filtering for sm_obs.")

    return work


def split_by_time(df: pd.DataFrame):
    months = sorted(df[GROUP_COLUMN].unique().tolist())
    split_idx = max(1, int(len(months) * 0.8))

    train_months = months[:split_idx]
    test_months = months[split_idx:]

    if not test_months:
        test_months = train_months[-1:]
        train_months = train_months[:-1]

    train_df = df[df[GROUP_COLUMN].isin(train_months)].copy()
    test_df = df[df[GROUP_COLUMN].isin(test_months)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Not enough months with sm_obs.")

    return train_df, test_df, train_months, test_months


def main():
    run_id = f"sm_bias_model_{utc_stamp()}"
    model_path = MODELS_DIR / f"{run_id}.joblib"
    manifest_path = MANIFESTS_DIR / f"{run_id}.json"

    raw_df = load_training_data()
    training_dataset_path = raw_df.attrs["training_dataset_path"]

    df = prepare_dataframe(raw_df)
    train_df, test_df, train_months, test_months = split_by_time(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    joblib.dump(model, model_path)

    manifest = TrainManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        script_name="train_sm_bias_model.py",
        training_dataset=training_dataset_path,
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        n_train_rows=int(len(train_df)),
        n_test_rows=int(len(test_df)),
        train_months=[int(x) for x in train_months],
        test_months=[int(x) for x in test_months],
        mae=mae,
        rmse=rmse,
        r2=r2,
        model_params=MODEL_PARAMS,
        notes=[
            "Target is sm_obs; simulated sm is corrected toward observed sm.",
            "Time-based split used to avoid leakage across months.",
            "Missing feature values were median-filled.",
        ],
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)

    print("Done.")
    print(f"Model    : {model_path}")
    print(f"Manifest : {manifest_path}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R2   : {r2:.6f}")


if __name__ == "__main__":
    main()