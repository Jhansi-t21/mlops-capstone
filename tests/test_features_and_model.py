"""Tests for feature engineering and the model pipeline."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS
from src.features import FeatureEngineer, build_pipeline


def test_feature_engineer_adds_columns(clean_df):
    out = FeatureEngineer().transform(clean_df[FEATURE_COLUMNS])
    for name in FeatureEngineer.NEW_FEATURES:
        assert name in out.columns
    assert len(out) == len(clean_df)


def test_feature_engineer_handles_zero_division(clean_df):
    X = clean_df[FEATURE_COLUMNS].head(5).copy()
    X["mean_perimeter"] = 0.0
    out = FeatureEngineer().transform(X)
    assert not np.isinf(out["area_perimeter_ratio"]).any()


def test_pipeline_handles_missing_values(clean_df):
    X = clean_df[FEATURE_COLUMNS].copy()
    X.iloc[0, 0] = np.nan
    model = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, clean_df["diagnosis"])
    assert model.predict(X.head(3)).shape == (3,)


def test_model_quality_gate(clean_df):
    X_tr, X_te, y_tr, y_te = train_test_split(
        clean_df[FEATURE_COLUMNS], clean_df["diagnosis"], test_size=0.2,
        random_state=42, stratify=clean_df["diagnosis"])
    model = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    assert f1_score(y_te, model.predict(X_te)) > 0.90
