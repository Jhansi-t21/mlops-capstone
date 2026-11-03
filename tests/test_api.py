"""Tests for the FastAPI service (health, predict, validation, prediction logging)."""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_dir / "model.joblib"))
    monkeypatch.setenv("MODEL_INFO_PATH", str(trained_model_dir / "model_info.json"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.csv"))
    import app.main as main
    main = importlib.reload(main)  # re-read the environment variables
    with TestClient(main.app) as c:  # "with" triggers startup -> model loading
        c.log_path = tmp_path / "predictions.csv"
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_malignant_and_benign(client, sample_payloads):
    mal = client.post("/predict", json=sample_payloads["malignant_example"]).json()
    ben = client.post("/predict", json=sample_payloads["benign_example"]).json()
    assert mal["prediction"] == 1 and mal["label"] == "malignant"
    assert ben["prediction"] == 0 and ben["label"] == "benign"
    assert 0.0 <= mal["probability_malignant"] <= 1.0


def test_predict_rejects_missing_field(client, sample_payloads):
    payload = dict(sample_payloads["malignant_example"])
    payload.pop("mean_radius")
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_negative_and_extra_fields(client, sample_payloads):
    bad = dict(sample_payloads["malignant_example"], mean_radius=-5)
    assert client.post("/predict", json=bad).status_code == 422
    extra = dict(sample_payloads["malignant_example"], hacker_field=1)
    assert client.post("/predict", json=extra).status_code == 422


def test_batch_predict(client, sample_payloads):
    body = {"records": [sample_payloads["malignant_example"], sample_payloads["benign_example"]]}
    r = client.post("/predict/batch", json=body)
    assert r.status_code == 200
    assert [p["prediction"] for p in r.json()["predictions"]] == [1, 0]


def test_predictions_are_logged_for_monitoring(client, sample_payloads):
    client.post("/predict", json=sample_payloads["benign_example"])
    lines = client.log_path.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 row


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and r.json()["version"] == "1"


def test_predict_returns_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    import app.main as main
    main = importlib.reload(main)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        sample = main._load_example()
        assert c.post("/predict", json=sample).status_code == 503
