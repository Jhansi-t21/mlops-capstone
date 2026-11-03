"""Step 2 - DATA CLEANING + PREPROCESSING.

raw CSV  ->  cleaned CSV  ->  stratified train / test split.

Cleaning rules (each one is counted and reported in metrics/data_quality.json):
  1. standardise column names to snake_case
  2. drop exact duplicate rows
  3. drop rows whose target is missing
  4. drop rows with too many missing feature values
  5. drop physically impossible rows (radius <= 0 or area <= 0)
  6. encode the target: malignant = 1, benign = 0

Remaining missing values are NOT filled here. They are filled inside the model
pipeline (SimpleImputer) so the median is learned from training data only, which
prevents data leakage.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, clean_column_name, load_params, resolve


def clean_data(df: pd.DataFrame, target: str = "diagnosis",
               max_missing_ratio: float = 0.30) -> tuple[pd.DataFrame, dict]:
    report = {"rows_raw": int(len(df))}

    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    before = len(df)
    df = df.dropna(subset=[target])
    report["missing_target_removed"] = int(before - len(df))

    before = len(df)
    missing_ratio = df[FEATURE_COLUMNS].isna().mean(axis=1)
    df = df[missing_ratio <= max_missing_ratio]
    report["too_many_missing_removed"] = int(before - len(df))

    before = len(df)
    valid = ~((df["mean_radius"] <= 0) | (df["mean_area"] <= 0))
    df = df[valid]
    report["invalid_rows_removed"] = int(before - len(df))

    df[target] = df[target].map({"malignant": 1, "benign": 0}).astype(int)
    df = df[FEATURE_COLUMNS + [target]].reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    report["missing_cells_left"] = int(df[FEATURE_COLUMNS].isna().sum().sum())
    report["malignant_share"] = round(float(df[target].mean()), 4)
    return df, report


def main() -> None:
    params = load_params()
    d = params["data"]

    raw = pd.read_csv(resolve(d["raw_path"]))
    clean, report = clean_data(raw, d["target"], d["max_missing_ratio_per_row"])

    train, test = train_test_split(
        clean, test_size=d["test_size"], random_state=d["random_state"],
        stratify=clean[d["target"]],
    )
    report["rows_train"], report["rows_test"] = int(len(train)), int(len(test))

    for key, frame in (("train_path", train), ("test_path", test)):
        out = resolve(d[key])
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)

    quality_path = resolve("metrics/data_quality.json")
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(report, indent=2))

    print("[preprocess] data quality report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
