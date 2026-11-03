"""Step 6 - FINAL MODEL EVALUATION of the exported champion on the untouched test set.

Writes metrics/metrics.json (tracked by DVC as a metrics file) plus a confusion matrix
and ROC curve, and logs everything to MLflow as a separate "final-evaluation" run.
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,  # noqa: E402
                             average_precision_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from src.config import FEATURE_COLUMNS, load_params, resolve  # noqa: E402


def evaluate_model(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),          # a.k.a. sensitivity
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "average_precision": average_precision_score(y, y_proba),
        "mcc": matthews_corrcoef(y, y_pred),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def main() -> None:
    params = load_params()
    p_cfg, d_cfg, m_cfg = params["paths"], params["data"], params["mlflow"]

    model = joblib.load(resolve(p_cfg["model"]))
    test = pd.read_csv(resolve(d_cfg["test_path"]))
    X, y = test[FEATURE_COLUMNS], test[d_cfg["target"]]

    metrics = evaluate_model(model, X, y)
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}

    resolve(p_cfg["metrics"]).write_text(json.dumps(metrics, indent=2))

    plots = resolve(p_cfg["plots_dir"])
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=["benign", "malignant"],
                                          ax=ax, colorbar=False)
    ax.set_title("Champion model - confusion matrix")
    fig.savefig(plots / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.set_title("Champion model - ROC curve")
    fig.savefig(plots / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(y, model.predict(X), target_names=["benign", "malignant"])
    (plots / "classification_report.txt").write_text(report)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", m_cfg["tracking_uri"]))
    mlflow.set_experiment(m_cfg["experiment_name"])
    with mlflow.start_run(run_name="final-evaluation-champion"):
        info = json.loads(resolve(p_cfg["model_info"]).read_text())
        mlflow.set_tag("registered_model_version", str(info["version"]))
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifacts(str(plots))

    print("[evaluate] champion model on the test set:")
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
