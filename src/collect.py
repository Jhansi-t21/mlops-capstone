"""Step 1 - DATA COLLECTION.

Loads the Breast Cancer Wisconsin (Diagnostic) dataset that ships with scikit-learn
(569 patient records, 30 numeric features, binary target) and writes it to
data/raw/ as a CSV file. DVC tracks this file.

Because the original dataset is already clean, we optionally simulate the data-quality
problems that real ingestion pipelines have (duplicate rows, missing values) so that
the cleaning step in preprocess.py has genuine work to do. This is controlled by
`data.simulate_dirty_data` in params.yaml and is clearly a simulation.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.config import load_params, resolve


def collect_raw_data(simulate_dirty: bool = True, seed: int = 7) -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()  # 30 feature columns + "target" (0 = malignant, 1 = benign)

    # Make the target human readable and unambiguous.
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.drop(columns=["target"])

    if simulate_dirty:
        rng = np.random.default_rng(seed)
        feature_cols = [c for c in df.columns if c != "diagnosis"]

        # (a) ~1.5 % of feature cells become missing
        mask = rng.random((len(df), len(feature_cols))) < 0.015
        values = df[feature_cols].to_numpy(dtype=float)
        values[mask] = np.nan
        df[feature_cols] = values

        # (b) 12 exact duplicate rows
        duplicates = df.sample(12, random_state=seed)
        df = pd.concat([df, duplicates], ignore_index=True)

        # (c) 3 impossible records (non-positive radius / area)
        bad_idx = rng.choice(len(df), size=3, replace=False)
        df.loc[bad_idx, "mean radius"] = -1.0
        df.loc[bad_idx, "mean area"] = 0.0

    return df


def main() -> None:
    params = load_params()
    out = resolve(params["data"]["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    df = collect_raw_data(
        simulate_dirty=params["data"]["simulate_dirty_data"],
        seed=params["data"]["dirty_seed"],
    )
    df.to_csv(out, index=False)
    print(f"[collect] saved {len(df)} rows x {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
