"""Step 10 - DATA DRIFT DETECTION AND MONITORING with Evidently AI.

Compares REFERENCE data (what the model was trained on) with CURRENT data (what the
model sees in production) and produces
    monitoring/reports/drift_report_<timestamp>.html   (open in a browser)
    monitoring/reports/latest_drift_report.html
    monitoring/reports/drift_summary.json              (machine readable)

Three ways to choose the current data:
    python -m src.monitor --simulate none     # test set  -> expect NO drift
    python -m src.monitor --simulate drift    # artificially shifted test set -> expect DRIFT
    python -m src.monitor --current logs/predictions.csv    # real requests logged by the API

Add --fail-on-drift to return exit code 1 when drift is detected (useful for CI / alerts).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import FEATURE_COLUMNS, ROOT, load_params, resolve

# Features we artificially shift to simulate a change in the real world
# (e.g. a new scanner that measures tumours ~35 % larger and a bit noisier).
SIMULATED_SHIFT = {
    "mean_radius": 1.35, "mean_perimeter": 1.35, "mean_area": 1.8,
    "worst_radius": 1.35, "worst_perimeter": 1.35, "worst_area": 1.8,
    "mean_texture": 1.25, "worst_texture": 1.25, "mean_concavity": 1.5,
}


def simulate_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col, factor in SIMULATED_SHIFT.items():
        out[col] = out[col] * factor * rng.normal(1.0, 0.03, size=len(out))
    return out


def _parse_columns(raw_report: dict) -> tuple[dict, list[dict]]:
    """Pull the overall drift share and the per-column results out of Evidently's dict."""
    overall, columns = {}, []
    for metric in raw_report["metrics"]:
        cfg = metric["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            overall = metric["value"]  # {"count": .., "share": ..}
        elif cfg["type"].endswith("ValueDrift"):
            method = cfg.get("method", "")
            value = float(metric["value"])
            threshold = float(cfg.get("threshold", 0.05))
            # p-value based tests: drift when p < threshold. Distance based: drift when > threshold.
            drifted = value < threshold if "p_value" in method else value > threshold
            columns.append({"column": cfg["column"], "method": method,
                            "score": round(value, 6), "threshold": threshold,
                            "drifted": bool(drifted)})
    return overall, columns


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                     out_dir: Path, share_threshold: float = 0.30) -> dict:
    """Run Evidently, save HTML + JSON, and return a summary dict."""
    definition = DataDefinition(numerical_columns=FEATURE_COLUMNS)
    ref_ds = Dataset.from_pandas(reference[FEATURE_COLUMNS], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[FEATURE_COLUMNS], data_definition=definition)

    snapshot = Report([DataDriftPreset()]).run(cur_ds, ref_ds)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"drift_report_{stamp}.html"
    snapshot.save_html(str(html_path))
    snapshot.save_html(str(out_dir / "latest_drift_report.html"))

    overall, columns = _parse_columns(snapshot.dict())
    n_drifted = sum(c["drifted"] for c in columns)
    share = n_drifted / max(len(columns), 1)

    summary = {
        "timestamp": stamp,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_checked": len(columns),
        "features_drifted": int(n_drifted),
        "drift_share": round(share, 4),
        "share_threshold": share_threshold,
        "dataset_drift_detected": bool(share >= share_threshold),
        "drifted_features": sorted(c["column"] for c in columns if c["drifted"]),
        "report_html": str(html_path.relative_to(ROOT)) if html_path.is_relative_to(ROOT)
        else str(html_path),
        "columns": columns,
    }
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _log_to_mlflow(summary: dict, params: dict) -> None:
    """Keep a history of drift checks next to the training experiments (best effort)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]))
        mlflow.set_experiment("drift-monitoring")
        with mlflow.start_run(run_name=f"drift-{summary['timestamp']}"):
            mlflow.log_metric("drift_share", summary["drift_share"])
            mlflow.log_metric("features_drifted", summary["features_drifted"])
            mlflow.set_tag("dataset_drift_detected", str(summary["dataset_drift_detected"]))
            mlflow.log_artifact(str(resolve(params["paths"]["drift_dir"]) /
                                    "latest_drift_report.html"))
    except Exception as exc:  # monitoring must never crash because of tracking problems
        print(f"[monitor] (MLflow logging skipped: {exc})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidently data drift check")
    parser.add_argument("--current", help="CSV with current/production data (e.g. logs/predictions.csv)")
    parser.add_argument("--simulate", choices=["none", "drift"],
                        help="use the test set as 'current' data, optionally with artificial drift")
    parser.add_argument("--min-rows", type=int, default=30,
                        help="refuse to run when the current data has fewer rows")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args(argv)

    params = load_params()
    reference = pd.read_csv(resolve(params["data"]["train_path"]))

    if args.current:
        current = pd.read_csv(args.current)
        source = args.current
    elif args.simulate:
        current = pd.read_csv(resolve(params["data"]["test_path"]))
        if args.simulate == "drift":
            current = simulate_drift(current)
        source = f"test set (simulate={args.simulate})"
    else:
        parser.error("choose --current <csv> or --simulate none|drift")

    if len(current) < args.min_rows:
        print(f"[monitor] only {len(current)} rows in current data (need {args.min_rows}). "
              "Send more requests to the API first.")
        return 2

    summary = run_drift_report(reference, current, resolve(params["paths"]["drift_dir"]),
                               params["monitoring"]["drift_share_threshold"])
    _log_to_mlflow(summary, params)

    print(f"[monitor] current data: {source}")
    print(f"[monitor] drifted features: {summary['features_drifted']}/{summary['features_checked']} "
          f"(share={summary['drift_share']:.0%}, alert threshold={summary['share_threshold']:.0%})")
    if summary["dataset_drift_detected"]:
        print("[monitor] *** DATA DRIFT DETECTED -> consider retraining ***")
    else:
        print("[monitor] no significant drift")
    print(f"[monitor] report: {summary['report_html']}")

    return 1 if (args.fail_on_drift and summary["dataset_drift_detected"]) else 0


if __name__ == "__main__":
    sys.exit(main())
