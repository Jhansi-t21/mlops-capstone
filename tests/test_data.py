"""Tests for data collection and cleaning."""
from src.config import FEATURE_COLUMNS
from src.collect import collect_raw_data
from src.preprocess import clean_data


def test_raw_data_has_enough_records(raw_df):
    assert len(raw_df) >= 500


def test_raw_data_is_dirty_when_simulated(raw_df):
    assert raw_df.duplicated().sum() >= 10
    assert raw_df.isna().sum().sum() > 0


def test_clean_data_removes_problems(raw_df):
    clean, report = clean_data(raw_df)
    assert report["duplicates_removed"] >= 10
    assert report["invalid_rows_removed"] >= 1
    assert not clean.duplicated().any()
    # NaNs are allowed (imputed later in the pipeline); real values must be positive
    assert (clean["mean_radius"].dropna() > 0).all()
    assert (clean["mean_area"].dropna() > 0).all()


def test_clean_data_schema_and_target(clean_df):
    assert list(clean_df.columns) == FEATURE_COLUMNS + ["diagnosis"]
    assert set(clean_df["diagnosis"].unique()) == {0, 1}
    assert len(clean_df) >= 500


def test_collect_is_reproducible():
    a = collect_raw_data(simulate_dirty=True, seed=7)
    b = collect_raw_data(simulate_dirty=True, seed=7)
    assert a.equals(b)
