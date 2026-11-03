"""Send realistic traffic to a RUNNING API so it writes logs/predictions.csv
(which Evidently then compares with the training data).

    python scripts/send_traffic.py                 # 100 normal requests  -> no drift expected
    python scripts/send_traffic.py --drift         # 100 shifted requests -> drift expected
    python scripts/send_traffic.py --url http://localhost:8000 --n 150
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURE_COLUMNS, resolve, load_params  # noqa: E402
from src.monitor import simulate_drift  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--drift", action="store_true", help="shift feature values before sending")
args = parser.parse_args()

params = load_params()
df = pd.read_csv(resolve(params["data"]["test_path"]))[FEATURE_COLUMNS]
df = df.fillna(df.median())                          # the API requires complete records
df = df.sample(args.n, replace=True, random_state=1)  # sample with replacement if n > rows
if args.drift:
    df = simulate_drift(df)

ok = 0
for _, row in df.iterrows():
    body = json.dumps({c: float(row[c]) for c in FEATURE_COLUMNS}).encode()
    req = urllib.request.Request(args.url.rstrip("/") + "/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        ok += resp.status == 200

print(f"sent {ok}/{len(df)} requests ({'DRIFTED' if args.drift else 'normal'} data)")
