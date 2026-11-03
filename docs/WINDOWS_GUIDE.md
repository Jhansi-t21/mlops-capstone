# Windows Step-by-Step Guide

Every command below is for **Windows PowerShell** (search "PowerShell" in the Start menu; no admin
rights needed unless stated). Lines starting with `#` are comments, do not type them.

Total time: about 60-90 minutes the first time (most of it is installing tools).

---

## PART 0 - What each tool does (in simple words)

| Tool | One-line meaning | Where it appears in this project |
|---|---|---|
| **Python / Pandas / Scikit-learn** | load data, build models | `src/*.py` |
| **DVC** | "Git for data and models": remembers which data + code + settings produced which model | `dvc.yaml`, `params.yaml` |
| **MLflow** | a lab notebook for experiments + a shelf (registry) where approved model versions live | `src/train.py` |
| **FastAPI** | turns the model into a web service others can call | `app/main.py` |
| **Docker** | packs the API and everything it needs into one box that runs the same everywhere | `Dockerfile` |
| **GitHub Actions** | a robot that re-runs lint, pipeline, tests and Docker build on every `git push` | `.github/workflows/` |
| **Evidently AI** | compares today's incoming data with training data and warns if it changed ("drift") | `src/monitor.py` |

---

## PART 1 - Install the tools

### 1.1 Python 3.12
1. Download **Python 3.12.x** from https://www.python.org/downloads/windows/
2. Run the installer. **Tick "Add python.exe to PATH"** on the first screen, then "Install Now".
3. Open a NEW PowerShell and check:
```powershell
python --version      # must print Python 3.12.x (3.13 also works; 3.11 does NOT, numpy 2.5 needs 3.12+)
pip --version
```
If `python` opens the Microsoft Store, go to *Settings -> Apps -> Advanced app settings -> App execution aliases* and switch off the two "python" aliases.

### 1.2 Git
1. Download "Git for Windows" from https://git-scm.com/download/win and install with default options.
2. Check and set your identity (use the same email as your GitHub account):
```powershell
git --version
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

### 1.3 Docker Desktop
1. Download from https://www.docker.com/products/docker-desktop/ and install (keep "Use WSL 2" ticked).
2. Restart Windows if asked, start **Docker Desktop**, wait until the bottom-left says "Engine running".
3. Check:
```powershell
docker --version
docker run hello-world      # prints "Hello from Docker!"
```
If it complains about virtualization, enable "Virtualization" in your BIOS/UEFI (search your laptop model + "enable virtualization").

### 1.4 GitHub account + VS Code (optional but recommended)
* Create a free account at https://github.com
* VS Code: https://code.visualstudio.com (install the "Python" extension).

---

## PART 2 - Create the project and the environment

### 2.1 Get the files
Unzip `mlops-capstone.zip` to a folder **without spaces in the path**, e.g. `C:\Projects\mlops-capstone`.
```powershell
cd C:\Projects\mlops-capstone
dir          # you should see README.md, dvc.yaml, params.yaml, src, app, tests ...
```

> Every file's full code is in the zip. Section "PART 11" explains what each file does and what you may edit.

### 2.2 Virtual environment (an isolated Python for this project)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Your prompt now starts with `(.venv)`. If PowerShell says *"running scripts is disabled"*, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
then activate again. **Always activate the venv in every new PowerShell window.**

### 2.3 Install the packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt       # takes 3-8 minutes
```
Verify:
```powershell
python -c "import pandas, sklearn, mlflow, evidently, fastapi; print('imports OK')"
dvc --version
mlflow --version
```
Optional (silences harmless MLflow messages) for the current window:
```powershell
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:MLFLOW_DISABLE_TELEMETRY = "1"
```

---

## PART 3 - Initialise Git and DVC (data versioning)

```powershell
git init
dvc init
git add .dvc .dvcignore
git commit -m "Initialise DVC"
```
`dvc init` creates the hidden `.dvc` folder (DVC's settings + cache).

### 3.1 Add a DVC remote (where data/model versions are stored)
A "remote" is like GitHub but for big files. For learning, use a folder on your disk:
```powershell
mkdir C:\dvc-storage
dvc remote add -d localstorage C:\dvc-storage
git add .dvc\config
git commit -m "Add local DVC remote"
```
*(In a real company this would be S3 / Azure Blob / Google Drive / DagsHub.)*

---

## PART 4 - Run the whole pipeline

```powershell
dvc repro
```
This runs four stages defined in `dvc.yaml`:

| Stage | Command it runs | What happens | Output |
|---|---|---|---|
| collect | `python -m src.collect` | loads the 569-patient dataset, simulates duplicates / missing values | `data/raw/breast_cancer_raw.csv` |
| preprocess | `python -m src.preprocess` | cleans (dups, invalid rows), splits 80/20 stratified | `data/processed/train.csv`, `test.csv`, `metrics/data_quality.json` |
| train | `python -m src.train` | feature engineering, trains 4 models, logs to MLflow, registers best, exports champion | `models/model.joblib`, `metrics/model_comparison.csv` |
| evaluate | `python -m src.evaluate` | final metrics + plots on the test set | `metrics/metrics.json`, `metrics/plots/*.png` |

Expected last lines of the training stage:
```
[train] BEST MODEL: logistic_regression (f1=0.9762)
Successfully registered model 'breast-cancer-classifier'.
[train] registered 'breast-cancer-classifier' version 1; promoted to 'champion': True
```
See the results:
```powershell
dvc metrics show        # table of metrics
dvc dag                 # picture of the pipeline stages
dvc status              # "Data and pipelines are up to date."
```
Run `dvc repro` again: every stage says **"didn't change, skipping"**. That is DVC's superpower.

### 4.1 Commit the pipeline (IMPORTANT: this step includes the `.gitignore` files DVC creates)
```powershell
git add -A
git status               # data\*.csv and models\model.joblib must NOT be listed (DVC keeps them out of Git)
git commit -m "Add MLOps pipeline, data version 1 and model v1"
dvc push                 # copies data + model to C:\dvc-storage
```

### 4.2 Prove data versioning works (great screenshot)
```powershell
# change a parameter, e.g. test size 0.2 -> 0.3
(Get-Content params.yaml) -replace 'test_size: 0.2','test_size: 0.3' | Set-Content params.yaml
dvc repro                # only preprocess, train, evaluate re-run (collect is skipped)
dvc params diff          # shows test_size 0.2 -> 0.3
dvc metrics diff         # shows how the metrics changed
git checkout params.yaml dvc.lock      # go back in time
dvc checkout             # restores the matching data and model
```

---

## PART 5 - MLflow: experiment tracking and model registry

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000 (leave this window running; open a new PowerShell for other commands).

What to click:
1. **Experiments -> breast-cancer-classification**: you see 4 runs (one per model) + 1 "final-evaluation-champion".
2. Tick all 4 model runs -> **Compare**: parallel-coordinates and scatter plots of F1/recall/precision.
3. Open one run: Parameters, Metrics, Artifacts (confusion matrix, saved model).
4. **Models -> breast-cancer-classifier**: version 1, alias **champion**, tags f1/recall/algorithm.

### 5.1 Create "version 2" and see version management
```powershell
(Get-Content params.yaml) -replace 'n_estimators: 150','n_estimators: 300' | Set-Content params.yaml
dvc repro
```
Refresh the MLflow Models page: a new **version 2** appears. The `champion` alias only moves to v2 if its F1 is
**>=** the old champion's F1 (see `promote_if_better` in `src/train.py`). Otherwise v1 stays champion.

---

## PART 6 - Tests and code quality

```powershell
flake8 src app tests scripts      # no output = clean
pytest -v                         # expect: 19 passed
```
What is tested: dataset size >= 500, cleaning rules, feature engineering, missing-value handling,
model quality gate (F1 > 0.90), every API endpoint (valid input, bad input, 503 without model, logging),
and Evidently drift (no drift on healthy data, drift on shifted data).

---

## PART 7 - FastAPI service

```powershell
uvicorn app.main:app --reload
```
* Swagger UI (interactive): http://127.0.0.1:8000/docs
* Health: http://127.0.0.1:8000/health

In a **second** PowerShell (venv activated):
```powershell
python scripts\smoke_test.py
```
Expected: `SMOKE TEST PASSED`. Or try it by hand:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model-info

$body = Get-Content app\sample_payloads.json | ConvertFrom-Json
$json = $body.malignant_example | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $json
```
Expected: `prediction 1, label malignant, probability_malignant ~0.9997`.

Try a bad request (a missing field) in `/docs` -> you get **422** (validation works).

---

## PART 8 - Docker

Make sure Docker Desktop is running and `dvc repro` has created `models\model.joblib`.
```powershell
docker build -t breast-cancer-api:latest .
docker images breast-cancer-api
docker run -d --name bc-api -p 8000:8000 breast-cancer-api:latest
docker ps                                  # STATUS should become "Up ... (healthy)" after ~15 s
docker logs bc-api
python scripts\smoke_test.py http://localhost:8000
```
(Stop `uvicorn` from Part 7 first if port 8000 is busy.)

Stop and clean up:
```powershell
docker stop bc-api
docker rm bc-api
```

---

## PART 9 - Drift detection with Evidently AI

### 9.1 Quick demo with simulated data
```powershell
python -m src.monitor --simulate none     # healthy data  -> "no significant drift"
python -m src.monitor --simulate drift    # shifted data   -> "DATA DRIFT DETECTED"
start monitoring\reports\latest_drift_report.html     # opens in the browser
```
The HTML shows, for each of the 30 features, the reference vs current distribution and the K-S test p-value.

### 9.2 Realistic demo: drift on REAL API traffic
```powershell
# terminal 1 (API running, either uvicorn or Docker)
Remove-Item logs\predictions.csv -ErrorAction SilentlyContinue
python scripts\send_traffic.py              # 100 normal requests
python -m src.monitor --current logs\predictions.csv       # -> no drift

Remove-Item logs\predictions.csv
python scripts\send_traffic.py --drift      # 100 shifted requests
python -m src.monitor --current logs\predictions.csv       # -> DRIFT DETECTED
python -m src.monitor --current logs\predictions.csv --fail-on-drift ; echo $LASTEXITCODE   # prints 1
```
If the API runs in Docker, copy its log out first:
`docker cp bc-api:/code/logs/predictions.csv logs\predictions.csv`

The exit code `1` is what CI/alerting systems use: "drift -> fail the job -> notify -> retrain".

### 9.3 Retraining after drift (the full MLOps loop)
Update your data, then `dvc repro` -> new MLflow version -> `docker build` -> redeploy.

---

## PART 10 - Push to GitHub and run CI/CD

### 10.1 Create the repository
1. https://github.com -> **New repository** -> name `mlops-capstone` -> **Public** (or Private)
2. **Do NOT** tick "Add a README / .gitignore / license" (the project already has them).
3. Copy the URL, e.g. `https://github.com/YOUR_USERNAME/mlops-capstone.git`

### 10.2 Push
```powershell
git add -A
git commit -m "Complete MLOps capstone"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mlops-capstone.git
git push -u origin main
```
A browser window opens for GitHub login on the first push (Git Credential Manager). If you are asked
for a password, create a Personal Access Token (GitHub -> Settings -> Developer settings -> Tokens (classic) -> scope `repo`)
and paste it instead of the password.

### 10.3 Watch the pipeline
GitHub repo -> **Actions** tab -> workflow **ci**. Jobs:
1. `train-and-test`: install -> flake8 -> `dvc repro` -> `dvc metrics show` -> pytest -> drift check -> upload artifacts
2. `docker`: download model -> `docker build` -> start container -> `smoke_test.py` -> push image to GHCR (main branch only)

Green check = success. Click a job to see logs (screenshot!). Download the **model-and-reports** artifact at the
bottom of the run page.

If the final push step fails with *permission denied*: repo **Settings -> Actions -> General -> Workflow
permissions -> "Read and write permissions" -> Save**, then re-run the job.

### 10.4 Run the drift workflow manually
**Actions -> drift-monitor -> Run workflow** -> choose `drift` -> download the `drift-report` artifact.

### 10.5 Pull your published image (optional)
```powershell
docker run -d -p 8000:8000 ghcr.io/YOUR_USERNAME_IN_LOWERCASE/mlops-capstone/breast-cancer-api:latest
```
(If the package is private: GitHub profile -> Packages -> the package -> Package settings -> change visibility.)

### 10.6 Make CI fail on purpose (shows the safety net works)
Open `params.yaml`, set `min_metric_to_register: 0.999`, commit and push -> the `train-and-test` job fails at
"Reproduce the DVC pipeline" with *Quality gate FAILED*. Revert the value and push again.

---

## PART 11 - File-by-file explanation and WHAT TO CHANGE

| File | What it does | What you can/should change |
|---|---|---|
| `params.yaml` | every setting in one place | `test_size`, model hyper-parameters, `selection_metric` (try `recall`), `min_metric_to_register`, `drift_share_threshold`, `simulate_dirty_data` |
| `src/config.py` | feature names, label names, helpers | `FEATURE_COLUMNS` if you change the dataset |
| `src/collect.py` | data collection | replace the data source (see 11.2) |
| `src/preprocess.py` | cleaning + split | cleaning rules for your own data |
| `src/features.py` | feature engineering + sklearn pipeline | add/remove engineered features |
| `src/train.py` | trains 4 models, MLflow, registry | add models in `get_candidates()` and `params.yaml` |
| `src/evaluate.py` | final metrics/plots | add metrics |
| `src/monitor.py` | Evidently drift | `SIMULATED_SHIFT` (which features to shift) |
| `app/main.py` | API | add endpoints; schema is generated from `FEATURE_COLUMNS` |
| `Dockerfile` | image recipe | Python version, port |
| `.github/workflows/ci.yml` | CI/CD | branch names, add deploy steps |
| `dvc.yaml` | pipeline graph | add stages |

### 11.1 Things you MUST personalise
1. `git config` name/email (Part 1.2).
2. `YOUR_USERNAME` in the GitHub URLs (Part 10).
3. Put your name, college and internship details in `README.md` (top) before submission.
4. Take your OWN screenshots (Part 12).

### 11.2 Using your own dataset instead (example: customer churn CSV)
1. Put `churn.csv` into `data\raw\` and in `src/collect.py` replace the body of `collect_raw_data()` by `pd.read_csv(...)`. Remove the dirty-data simulation (`simulate_dirty_data: false`).
2. `src/config.py`: set `FEATURE_COLUMNS` to your numeric columns. For categorical columns, extend `build_pipeline()` in `src/features.py` with a `ColumnTransformer` + `OneHotEncoder`.
3. `src/preprocess.py`: adapt `clean_data()` (target mapping, validity rules).
4. `params.yaml`: set `data.target`.
5. `src/features.py`: rewrite `FeatureEngineer.transform()` for your columns.
6. `app/sample_payloads.json`: two example records of your features (the API schema updates automatically).
7. `src/monitor.py`: update `SIMULATED_SHIFT` to your column names.
8. `tests/`: adjust the assertions (column names, row counts).
9. `dvc repro`, `pytest`, done.

---

## PART 12 - Proof-of-execution checklist (screenshots)

Press **Win + Shift + S**, drag over the window, and paste into a folder `screenshots\` (create it) named as below.

| # | File name | What the screen must show |
|---|---|---|
| 1 | `01_versions.png` | PowerShell: `python --version`, `git --version`, `docker --version`, `dvc --version`, `mlflow --version` |
| 2 | `02_project_tree.png` | VS Code Explorer (or `tree /F /A`) showing the folder structure |
| 3 | `03_dvc_repro.png` | `dvc repro` output with the 4 stages running |
| 4 | `04_dvc_dag_metrics.png` | `dvc dag` and `dvc metrics show` |
| 5 | `05_dvc_reproducibility.png` | second `dvc repro` with "didn't change, skipping", and the `dvc params diff` / `dvc metrics diff` from 4.2 |
| 6 | `06_mlflow_runs.png` | MLflow experiment page with the 4 runs |
| 7 | `07_mlflow_compare.png` | the Compare view of the 4 runs |
| 8 | `08_mlflow_registry.png` | Models -> breast-cancer-classifier with versions and `champion` alias |
| 9 | `09_metrics_plots.png` | `metrics\plots\confusion_matrix.png` and `roc_curve.png` |
| 10 | `10_pytest.png` | `pytest -v` ending with `19 passed` |
| 11 | `11_swagger.png` | http://127.0.0.1:8000/docs with the `/predict` response expanded |
| 12 | `12_health.png` | browser/PowerShell showing `/health` with `"model_loaded": true` |
| 13 | `13_docker_build_run.png` | `docker build`, `docker ps` showing `(healthy)` and the smoke test passing |
| 14 | `14_drift_no_drift.png` | `python -m src.monitor --simulate none` result |
| 15 | `15_drift_detected.png` | drift run + the Evidently HTML report in the browser |
| 16 | `16_github_repo.png` | repository page on GitHub with all folders |
| 17 | `17_actions_green.png` | Actions tab with a green `ci` run (both jobs) |
| 18 | `18_ghcr_package.png` | (optional) the Docker image under "Packages" |

Tip: also run `git log --oneline` for a screenshot that proves incremental commits.

---

## PART 13 - Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | reinstall Python with "Add to PATH"; open a NEW PowerShell |
| `.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ImportError ... infer_dtype_from_object` from evidently | you have pandas 3; run `pip install "pandas==2.3.3"` |
| `ModuleNotFoundError: No module named 'src'` | run commands from the project root, using `python -m src.xxx` |
| `dvc repro` fails: *output is already tracked by SCM* | `git rm -r --cached data models\model.joblib` then `git commit`, then `dvc repro --force` |
| MLflow UI shows no experiments | start it from the project folder with `--backend-store-uri sqlite:///mlflow.db` |
| `Port 8000 is already allocated` | `docker stop bc-api` / stop uvicorn, or use `-p 8001:8000` |
| Docker build: `models/model.joblib not found` | run `dvc repro` first |
| `docker: error during connect` | Docker Desktop is not running |
| Drift tool says "need 30 rows" | send more requests with `scripts\send_traffic.py` |
| CI fails at `dvc repro` with *Quality gate FAILED* | lower `min_metric_to_register` or fix the model |
| `git push` rejected | `git pull --rebase origin main` then push again |

Reset everything and start fresh:
```powershell
Remove-Item -Recurse -Force mlruns, mlflow.db, logs, monitoring\reports, data\raw, data\processed, models\model.joblib -ErrorAction SilentlyContinue
dvc repro --force
```
