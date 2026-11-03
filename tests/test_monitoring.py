"""Tests for Evidently-based drift detection."""
from src.monitor import run_drift_report, simulate_drift


def test_no_drift_on_same_distribution(reference_and_current, tmp_path):
    ref, cur = reference_and_current
    summary = run_drift_report(ref, cur, tmp_path)
    assert summary["dataset_drift_detected"] is False
    assert (tmp_path / "latest_drift_report.html").exists()
    assert (tmp_path / "drift_summary.json").exists()


def test_drift_detected_when_data_shifts(reference_and_current, tmp_path):
    ref, cur = reference_and_current
    summary = run_drift_report(ref, simulate_drift(cur), tmp_path)
    assert summary["dataset_drift_detected"] is True
    assert "mean_area" in summary["drifted_features"]
