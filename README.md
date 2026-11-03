# End-to-End MLOps Capstone: Breast Cancer Diagnosis Pipeline

A complete, reproducible MLOps project: data versioning, experiment tracking, model registry,
REST API, Docker, CI/CD and drift monitoring.

> **Educational project.** The model predicts whether a tumour is malignant or benign from 30
> measurements of a cell-nucleus image (Breast Cancer Wisconsin Diagnostic dataset, 569 patients,
> bundled with scikit-learn). It is NOT a medical device.

| Item | Value |
|---|---|
| Prediction target | `diagnosis` (1 = malignant, 0 = benign) |
| Records | 569 original (581 after simulated duplicates) -> 566 after cleaning |
| Features | 30 raw + 6 engineered |
| Models compared | Logistic Regression, Random Forest, Gradient Boosting, SVC |

## Architecture

```
 collect ──► preprocess ──► train ──────────► evaluate        <- DVC pipeline (dvc.yaml)
 (raw CSV)   (clean, split)  │  4 models                         versions data, params, models
                             ▼
                          MLflow  ── tracking (params, metrics, plots)
                             │     ── Model Registry (versions + "champion" alias)
                             ▼
                      models/model.joblib ──► FastAPI (/health, /predict) ──► Docker image
                                                   │                              ▲
                                       logs/predictions.csv                       │
                                                   ▼                     GitHub Actions CI/CD
                                         Evidently drift report     (lint → dvc repro → tests → docker)
```

## Project structure

```
mlops-capstone/
├── .github/workflows/
│   ├── ci.yml                 # lint, dvc repro, tests, docker build + smoke test, push to GHCR
│   └── drift-monitor.yml      # weekly Evidently drift report
├── app/
│   ├── main.py                # FastAPI service
│   └── sample_payloads.json   # example requests (also used by tests / smoke test)
├── src/
│   ├── config.py              # paths, feature list, params loader
│   ├── collect.py             # step 1: data collection
│   ├── preprocess.py          # step 2: cleaning + train/test split
│   ├── features.py            # step 3: feature engineering + sklearn pipeline
│   ├── train.py               # steps 4-6: train, MLflow tracking, registry, champion export
│   ├── evaluate.py            # step 6: final metrics + plots
│   └── monitor.py             # step 10: Evidently drift detection
├── tests/                     # step 11: pytest suite (19 tests)
├── scripts/
│   ├── smoke_test.py          # checks a running API (local, Docker or CI)
│   ├── send_traffic.py        # generates API traffic (normal or drifted)
│   └── run_all.ps1            # one-click local run on Windows
├── docs/WINDOWS_GUIDE.md      # <- START HERE: full step-by-step guide
├── params.yaml                # all tunable settings
├── dvc.yaml                   # pipeline definition
├── Dockerfile / .dockerignore
├── requirements.txt / requirements-api.txt
└── pytest.ini / .flake8 / .gitignore
```

## Quick start (details in `docs/WINDOWS_GUIDE.md`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
git init
dvc init
dvc repro                                   # data -> train -> evaluate
pytest -v                                   # tests
uvicorn app.main:app --reload               # API at http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db   # MLflow at http://127.0.0.1:5000
python -m src.monitor --simulate drift      # Evidently report
docker build -t breast-cancer-api .         # container
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + whether the model is loaded |
| GET | `/model-info` | registered model name / version / metrics being served |
| POST | `/predict` | one record -> `prediction`, `label`, `probability_malignant` |
| POST | `/predict/batch` | up to 1000 records |

## Results (reference run)

| Model | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (champion)** | **0.976** | 0.976 | 0.976 | 0.999 |
| SVC | 0.976 | 0.952 | 1.000 | 0.999 |
| Gradient Boosting | 0.963 | 0.929 | 1.000 | 0.998 |
| Random Forest | 0.940 | 0.929 | 0.951 | 0.996 |

Test set = 114 patients. Your numbers should match because every seed is fixed.
Expected output files are in `docs/expected_output/`.

## Limitations (be honest in your report)

* The dataset is small and "easy" (all models reach > 94 % F1), so differences between models are
  within noise on a 114-row test set. The value of this project is the *pipeline*, not the model.
* Duplicates / missing values in the raw file are **simulated** (`data.simulate_dirty_data`) to
  demonstrate cleaning. Set it to `false` to use the original data.
* The "drift" scenario is synthetic (values are scaled on purpose) to prove the monitor works.
