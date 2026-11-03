"""Steps 4-6 - MODEL TRAINING, EXPERIMENT TRACKING (MLflow), MODEL REGISTRY.

What this script does:
  1. trains 4 different models (each is a full pipeline: features -> impute -> scale -> model)
  2. runs 5-fold cross-validation + evaluates on the hold-out test set
  3. logs params, metrics, confusion matrix and the model of every candidate to MLflow
  4. picks the best candidate (params.yaml -> training.selection_metric)
  5. quality gate: registers it in the MLflow Model Registry only if metric >= threshold
  6. promotes it to the alias "champion" only if it is at least as good as the old champion
  7. exports the champion to models/model.joblib so FastAPI / Docker can serve it
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI needed (works on servers / CI)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from src.config import FEATURE_COLUMNS, load_params, resolve  # noqa: E402
from src.features import build_pipeline  # noqa: E402

ALIAS = "champion"


def get_candidates(model_params: dict) -> dict:
    """Name -> untrained classifier, configured from params.yaml."""
    return {
        "logistic_regression": LogisticRegression(**model_params["logistic_regression"]),
        "random_forest": RandomForestClassifier(**model_params["random_forest"]),
        "gradient_boosting": GradientBoostingClassifier(**model_params["gradient_boosting"]),
        "svc": SVC(**model_params["svc"]),
    }


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_confusion_matrix(model, X_test, y_test, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["benign", "malignant"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {name}")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def promote_if_better(client: MlflowClient, model_name: str, new_version: str,
                      metric: str, new_value: float) -> bool:
    """Move the 'champion' alias to the new version only if it is not worse."""
    try:
        current = client.get_model_version_by_alias(model_name, ALIAS)
        old_value = float(current.tags.get(metric, "-1"))
    except Exception:  # no champion yet
        old_value = -1.0

    if new_value >= old_value:
        client.set_registered_model_alias(model_name, ALIAS, new_version)
        return True
    return False


def main() -> None:
    params = load_params()
    t_cfg, m_cfg, d_cfg, p_cfg = (params["training"], params["mlflow"],
                                  params["data"], params["paths"])

    # ---------- data ----------
    train = pd.read_csv(resolve(d_cfg["train_path"]))
    test = pd.read_csv(resolve(d_cfg["test_path"]))
    target = d_cfg["target"]
    X_train, y_train = train[FEATURE_COLUMNS], train[target]
    X_test, y_test = test[FEATURE_COLUMNS], test[target]

    # ---------- MLflow setup ----------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", m_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(m_cfg["experiment_name"])
    client = MlflowClient()

    cv = StratifiedKFold(n_splits=t_cfg["cv_folds"], shuffle=True, random_state=42)
    selection_metric = t_cfg["selection_metric"]
    results = []

    # ---------- train + track every candidate ----------
    for name, clf in get_candidates(t_cfg["models"]).items():
        pipeline = build_pipeline(clf)
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("model_family", name)
            mlflow.log_params({f"{name}__{k}": v for k, v in t_cfg["models"][name].items()})
            mlflow.log_param("cv_folds", t_cfg["cv_folds"])
            mlflow.log_param("n_train_rows", len(X_train))

            cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)
            metrics["cv_f1_mean"] = float(cv_f1.mean())
            metrics["cv_f1_std"] = float(cv_f1.std())
            mlflow.log_metrics(metrics)

            log_confusion_matrix(pipeline, X_test, y_test, name)
            signature = infer_signature(X_test, pipeline.predict(X_test))
            # cloudpickle is used because our pipeline contains a custom class
            # (FeatureEngineer); the newer default "skops" format refuses unknown classes.
            info = mlflow.sklearn.log_model(
                pipeline, name="model", signature=signature, input_example=X_test.head(3),
                serialization_format="cloudpickle",
            )

            results.append({"model": name, "run_id": run.info.run_id,
                            "model_uri": info.model_uri, **metrics})
            print(f"[train] {name:<20} f1={metrics['f1']:.4f}  recall={metrics['recall']:.4f}  "
                  f"precision={metrics['precision']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    # ---------- compare ----------
    comparison = pd.DataFrame(results).sort_values(selection_metric, ascending=False)
    comp_path = resolve(p_cfg["comparison"])
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.drop(columns=["model_uri"]).to_csv(comp_path, index=False)

    best = comparison.iloc[0]
    print(f"\n[train] BEST MODEL: {best['model']} ({selection_metric}={best[selection_metric]:.4f})")

    # ---------- quality gate ----------
    if best[selection_metric] < t_cfg["min_metric_to_register"]:
        raise SystemExit(
            f"Quality gate FAILED: {selection_metric}={best[selection_metric]:.4f} "
            f"< {t_cfg['min_metric_to_register']}. Model not registered."
        )

    # ---------- register + promote ----------
    model_name = m_cfg["registered_model_name"]
    version = mlflow.register_model(best["model_uri"], model_name)
    for key in ("f1", "recall", "precision", "accuracy", "roc_auc"):
        client.set_model_version_tag(model_name, version.version, key, f"{best[key]:.6f}")
    client.set_model_version_tag(model_name, version.version, "algorithm", best["model"])
    client.set_model_version_tag(model_name, version.version, "run_id", best["run_id"])

    promoted = promote_if_better(client, model_name, version.version,
                                 selection_metric, float(best[selection_metric]))
    print(f"[train] registered '{model_name}' version {version.version}; "
          f"promoted to '{ALIAS}': {promoted}")

    # ---------- export the champion for serving ----------
    champion = client.get_model_version_by_alias(model_name, ALIAS)
    champion_model = mlflow.sklearn.load_model(f"models:/{model_name}@{ALIAS}")
    model_path = resolve(p_cfg["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, model_path)

    info = {
        "registered_model_name": model_name,
        "version": champion.version,
        "alias": ALIAS,
        "algorithm": champion.tags.get("algorithm"),
        "selection_metric": selection_metric,
        "f1": float(champion.tags.get("f1", "nan")),
        "recall": float(champion.tags.get("recall", "nan")),
        "feature_columns": FEATURE_COLUMNS,
    }
    resolve(p_cfg["model_info"]).write_text(json.dumps(info, indent=2))
    print(f"[train] champion = version {champion.version} ({info['algorithm']}) -> {model_path}")


if __name__ == "__main__":
    main()
