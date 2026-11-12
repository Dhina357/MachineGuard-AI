# MachineGuard AI

Predictive-maintenance starter project with a FastAPI backend, scikit-learn training pipeline, SQLite storage, and React + TypeScript dashboard.

> **Important:** The demo data and fallback heuristic are synthetic/illustrative only. They are not validated for real machinery or safety-critical maintenance decisions. Train and evaluate using representative, labelled equipment data before operational use.

## Features
- Sensor reading ingestion and validation
- Failure-risk scoring with a trained Random Forest model
- Clearly labelled demo heuristic if no model artifact exists
- Machine summaries and prediction history
- SQLite persistence, Docker Compose, automated tests

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

API docs: http://localhost:8000/docs

Dashboard:

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

Docker: `docker compose up --build`

## API
- `GET /health`
- `GET /api/v1/machines`
- `GET /api/v1/readings`
- `POST /api/v1/readings`
- `POST /api/v1/predict`
- `GET /api/v1/predictions`

## Training CSV schema
`temperature,vibration,pressure,rotational_speed,operating_hours,failure`

The target `failure` is binary (0/1). Metrics describe only the supplied dataset. Do not use synthetic sample metrics as evidence of real-world model performance.

RUL estimation is intentionally not fabricated: it requires suitable run-to-failure histories or validated degradation labels. PostgreSQL, MLflow, authentication, and production monitoring are future extensions.

## Tests
```bash
cd backend
pip install -r requirements.txt
pytest
```
