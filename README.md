# MachineGuard AI

Predictive-maintenance application with a FastAPI backend, scikit-learn models, SQLite storage, and a React + TypeScript dashboard.

> **Safety and validity:** The fallback failure-risk score is illustrative. The RUL pipeline is designed for NASA C-MAPSS FD001 turbofan run-to-failure data. Neither model is validated for real industrial machinery or safety-critical maintenance decisions.

## Features
- Sensor reading ingestion and validation
- Failure-risk scoring with a trained classifier or labelled demo fallback
- NASA C-MAPSS FD001 Remaining Useful Life (RUL) regression pipeline
- RUL endpoint that refuses to fabricate a result when no trained artifact exists
- Machine summaries, prediction history, SQLite, Docker Compose, and tests

## Quick start

Python 3.11+:

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m app.scripts.generate_sample_data --output ../data/sample_sensor_data.csv
python -m app.ml.train --input ../data/sample_sensor_data.csv --output ../models/failure_model.joblib
uvicorn app.main:app --reload
```

Dashboard:

```bash
cd frontend
npm install
npm run dev
```

API docs: http://localhost:8000/docs
Dashboard: http://localhost:5173
Docker: `docker compose up --build`

## RUL model: NASA C-MAPSS FD001

1. Download the NASA C-MAPSS dataset from the official NASA Prognostics Center of Excellence repository.
2. Extract `train_FD001.txt` into a local `data/` directory. Do not commit the dataset.
3. Train the model:

```bash
cd backend
python -m app.ml.train_rul --input ../data/train_FD001.txt --output ../models/rul_model.joblib
```

The training script derives RUL as each engine's final training cycle minus the current cycle, capped at 125 cycles. It uses an engine-level holdout split to reduce leakage between training and evaluation engines and reports MAE/RMSE in cycles. Those metrics are printed by the script and stored in the model artifact; they are not pre-claimed here.

The inference request uses the current cycle, three operating settings, and 21 sensor values in the same feature order as C-MAPSS. Example:

```json
{
  "machine_id": "ENGINE-01",
  "cycle": 100,
  "settings": [0.002, 0.0003, 100.0],
  "sensors": [518.67, 642.0, 1580.0, 1400.0, 14.62, 21.61, 553.0, 2388.0, 9050.0, 1.3, 47.0, 521.0, 2388.0, 8138.0, 8.4, 0.03, 392.0, 2388.0, 100.0, 39.0, 23.4]
}
```

`POST /api/v1/rul/predict` returns the estimated remaining cycles. It returns HTTP 503 until a trained RUL model artifact is present; no heuristic RUL is invented.

## Existing failure-risk CSV schema

`temperature,vibration,pressure,rotational_speed,operating_hours,failure`

The target `failure` is binary (0/1). Metrics describe only the supplied dataset. Do not use synthetic sample metrics as evidence of real-world model performance.

## API
- `GET /health`
- `GET /api/v1/machines`
- `GET /api/v1/readings`
- `POST /api/v1/readings`
- `POST /api/v1/predict`
- `GET /api/v1/predictions`
- `POST /api/v1/rul/predict`

## Tests

```bash
cd backend
pip install -r requirements.txt
pytest
```
