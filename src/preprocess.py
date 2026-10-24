"""Dataset loading and leakage-safe preprocessing pipeline construction."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and minimally clean the IBM Telco CSV without fitting transformations."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Place the IBM Telco CSV at "
            f"{DEFAULT_DATA_PATH} or pass its location with --data."
        )

    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    required = {TARGET_COLUMN, ID_COLUMN, "TotalCharges"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    frame = frame.drop_duplicates().copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("string").str.strip()
    if not frame[TARGET_COLUMN].dropna().isin(["Yes", "No"]).all():
        raise ValueError("The Churn column must contain only 'Yes' and 'No'.")

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=[TARGET_COLUMN])
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, excluding the non-predictive customer ID."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Dataset must include a {TARGET_COLUMN!r} column.")
    features = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore").copy()
    target = frame[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype("int8")
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build imputers, scaling, and one-hot encoding for the discovered columns."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def model_feature_columns(model: Any) -> list[str]:
    """Return the feature names expected by a fitted preprocessing pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    return list(preprocessor.feature_names_in_)
