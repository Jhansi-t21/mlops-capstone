"""Shared fixtures. Tests build their own tiny model so they never depend on a prior
`dvc repro` run, which keeps them fast and independent."""
import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.collect import collect_raw_data
from src.config import FEATURE_COLUMNS
from src.features import build_pipeline
from src.preprocess import clean_data


@pytest.fixture(scope="session")
def raw_df():
    return collect_raw_data(simulate_dirty=True, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    df, _ = clean_data(raw_df)
    return df


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, clean_df):
    """Train a small model once and save it the same way src/train.py does."""
    out = tmp_path_factory.mktemp("model")
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(clean_df[FEATURE_COLUMNS], clean_df["diagnosis"])
    joblib.dump(model, out / "model.joblib")
    (out / "model_info.json").write_text(json.dumps({
        "registered_model_name": "test-model", "version": "1", "alias": "champion",
        "algorithm": "logistic_regression", "f1": 0.97}))
    return out


@pytest.fixture(scope="session")
def sample_payloads():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "app" / "sample_payloads.json"
    return json.loads(path.read_text())


@pytest.fixture()
def reference_and_current(clean_df):
    ref = clean_df.sample(frac=0.7, random_state=1)
    cur = clean_df.drop(ref.index)
    return ref, pd.DataFrame(cur)
