import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

DATA_DIR = Path("Rainfall_Prediction/data")
OUT_DIR = Path("Rainfall_Prediction/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = [
    "Date","Location","WindGustDir","WindDir9am","WindDir3pm"
]
TARGET_COL = "RainTomorrow"

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map target + binary cat to {0,1}
    df["RainToday"] = df["RainToday"].replace({"No": 0, "Yes": 1})
    df["RainTomorrow"] = df["RainTomorrow"].replace({"No": 0, "Yes": 1})
    return df

def fill_categorical_mode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

def label_encode_object_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def mice_impute(df: pd.DataFrame) -> pd.DataFrame:
    imputer = IterativeImputer()
    vals = imputer.fit_transform(df.values)
    return pd.DataFrame(vals, columns=df.columns, index=df.index)

def iqr_outlier_trim(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df.loc[mask]

def preprocess_and_split(csv_path: str = "Rainfall_Prediction/data/weatherAUS.csv",
                         test_size: float = 0.25, random_state: int = 12345):
    # 1) Load
    df = load_raw(Path(csv_path))

    # 2) Fill mode for categoricals
    df = fill_categorical_mode(df)

    # 3) Label encode remaining object columns
    df, encoders = label_encode_object_cols(df)

    # 4) MICE imputation
    df = mice_impute(df)

    # 5) IQR outlier trim
    df = iqr_outlier_trim(df)

    # 6) Save processed snapshot
    processed_path = OUT_DIR / "processed.csv"
    df.to_csv(processed_path, index=False)

    # 7) Train/test split
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # save split
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    # store encoders info
    with open(OUT_DIR / "label_encoders.json", "w") as f:
        json.dump({k: list(v.classes_) for k, v in encoders.items()}, f, indent=2)

    print(f"[OK] Preprocessed & split. Files in {OUT_DIR}")

if __name__ == "__main__":
    preprocess_and_split()
