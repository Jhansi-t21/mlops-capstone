"""Shared configuration helpers."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# The 30 numeric input features of the Breast Cancer Wisconsin (Diagnostic) dataset,
# after column names were converted to snake_case (e.g. "mean radius" -> "mean_radius").
FEATURE_COLUMNS = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error",
    "compactness_error", "concavity_error", "concave_points_error", "symmetry_error",
    "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry",
    "worst_fractal_dimension",
]

LABELS = {0: "benign", 1: "malignant"}


def load_params(path: str | Path | None = None) -> dict:
    """Read params.yaml from the project root."""
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative_path: str) -> Path:
    """Turn a path from params.yaml into an absolute path inside the project."""
    return ROOT / relative_path


def clean_column_name(name: str) -> str:
    """'mean concave points' -> 'mean_concave_points'."""
    return name.strip().lower().replace(" ", "_")
