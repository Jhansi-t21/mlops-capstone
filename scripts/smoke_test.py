"""Smoke test for a RUNNING API (local uvicorn, Docker container, or CI).

    python scripts/smoke_test.py                       # default http://127.0.0.1:8000
    python scripts/smoke_test.py http://localhost:8000

Uses only the standard library, so it works anywhere. Exit code 0 = all good.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "app" / "sample_payloads.json").read_text())


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def wait_until_healthy(seconds: int = 30) -> dict:
    deadline = time.time() + seconds
    while True:
        try:
            return call("/health")
        except Exception:
            if time.time() > deadline:
                raise SystemExit(f"API at {BASE} did not become healthy in {seconds}s")
            time.sleep(1)


health = wait_until_healthy()
print("GET /health      ->", health)
assert health["status"] == "ok" and health["model_loaded"] is True

mal = call("/predict", SAMPLES["malignant_example"])
ben = call("/predict", SAMPLES["benign_example"])
print("POST /predict    ->", mal)
print("POST /predict    ->", ben)
assert mal["prediction"] == 1 and ben["prediction"] == 0

print("\nSMOKE TEST PASSED")
