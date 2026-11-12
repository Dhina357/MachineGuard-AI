import os
from pathlib import Path
os.environ["DATABASE_URL"] = "sqlite:///./test_machineguard.db"
os.environ["MODEL_PATH"] = str(Path("./missing-test-model.joblib").resolve())
os.environ["RUL_MODEL_PATH"] = str(Path("./missing-rul-model.joblib").resolve())
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine
Base.metadata.drop_all(bind=engine); Base.metadata.create_all(bind=engine)
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ingest_and_predict():
    payload = {"machine_id":"TEST-01","temperature":70,"vibration":.3,"pressure":7,"rotational_speed":1500,"operating_hours":1000}
    assert client.post("/api/v1/readings", json=payload).status_code == 201
    result = client.post("/api/v1/predict", json=payload)
    assert result.status_code == 200
    assert 0 <= result.json()["risk_score"] <= 1

def test_invalid_sensor_range():
    payload = {"machine_id":"BAD","temperature":10000,"vibration":.3,"pressure":7,"rotational_speed":1500,"operating_hours":1000}
    assert client.post("/api/v1/readings", json=payload).status_code == 422

def test_rul_requires_trained_model():
    payload = {"machine_id":"ENGINE-01","cycle":100,"settings":[0.1,0.2,0.3],"sensors":[0.0]*21}
    response = client.post("/api/v1/rul/predict", json=payload)
    assert response.status_code == 503

def test_rul_rejects_wrong_sensor_count():
    payload = {"machine_id":"ENGINE-01","cycle":100,"settings":[0.1,0.2,0.3],"sensors":[0.0]*20}
    assert client.post("/api/v1/rul/predict", json=payload).status_code == 422
