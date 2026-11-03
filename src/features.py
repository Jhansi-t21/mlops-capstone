"""Step 3 - FEATURE ENGINEERING and the model pipeline factory.

The feature engineering lives INSIDE the scikit-learn Pipeline. That guarantees the
exact same transformations run during training and inside the FastAPI service
(no "training/serving skew").
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-inspired ratio / interaction features to the 30 raw columns."""

    NEW_FEATURES = [
        "area_perimeter_ratio",
        "radius_texture_interaction",
        "compactness_concavity_interaction",
        "worst_to_mean_radius",
        "worst_to_mean_area",
        "radius_error_ratio",
    ]

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
        out = a / b.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["area_perimeter_ratio"] = self._safe_divide(X["mean_area"], X["mean_perimeter"])
        X["radius_texture_interaction"] = X["mean_radius"] * X["mean_texture"]
        X["compactness_concavity_interaction"] = X["mean_compactness"] * X["mean_concavity"]
        X["worst_to_mean_radius"] = self._safe_divide(X["worst_radius"], X["mean_radius"])
        X["worst_to_mean_area"] = self._safe_divide(X["worst_area"], X["mean_area"])
        X["radius_error_ratio"] = self._safe_divide(X["radius_error"], X["mean_radius"])
        return X


def build_pipeline(classifier) -> Pipeline:
    """feature engineering -> median imputation -> scaling -> classifier."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
