from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Prediction, SensorReading
from app.ml.predictor import FailurePredictor, risk_level
from app.ml.rul import RULPredictor
from app.schemas.rul import RULPredictRequest, RULPredictResponse
from app.schemas.sensor import MachineSummary, PredictionOut, SensorReadingCreate, SensorReadingOut

router = APIRouter(prefix="/api/v1")
predictor = FailurePredictor()
rul_predictor = RULPredictor()

@router.post("/readings", response_model=SensorReadingOut, status_code=201)
def create_reading(payload: SensorReadingCreate, db: Session = Depends(get_db)):
    row = SensorReading(**payload.model_dump())
    db.add(row); db.commit(); db.refresh(row)
    return row

@router.get("/readings", response_model=list[SensorReadingOut])
def list_readings(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    return list(db.scalars(select(SensorReading).order_by(desc(SensorReading.created_at)).limit(limit)))

@router.post("/predict", response_model=PredictionOut)
def predict(payload: SensorReadingCreate, db: Session = Depends(get_db)):
    row = SensorReading(**payload.model_dump())
    db.add(row); db.flush()
    score, failed, source = predictor.predict(row)
    result = Prediction(reading_id=row.id, machine_id=row.machine_id, risk_score=round(score, 4),
        risk_level=risk_level(score), predicted_failure=failed, model_source=source)
    db.add(result); db.commit(); db.refresh(result)
    return result

@router.post("/rul/predict", response_model=RULPredictResponse)
def predict_rul(payload: RULPredictRequest):
    try:
        estimate, cap = rul_predictor.predict(payload.cycle, payload.settings, payload.sensors)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RULPredictResponse(
        machine_id=payload.machine_id,
        predicted_rul_cycles=estimate,
        model_source="nasa_cmapss_fd001_random_forest",
        target_cap_cycles=cap,
        warning="Research/demo estimate from NASA C-MAPSS FD001; validate on representative equipment before maintenance decisions."
    )

@router.get("/predictions", response_model=list[PredictionOut])
def list_predictions(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    return list(db.scalars(select(Prediction).order_by(desc(Prediction.created_at)).limit(limit)))

@router.get("/machines", response_model=list[MachineSummary])
def list_machines(db: Session = Depends(get_db)):
    readings = db.scalars(select(SensorReading).order_by(desc(SensorReading.created_at))).all()
    latest = {}
    for row in readings: latest.setdefault(row.machine_id, row)
    predictions = db.scalars(select(Prediction).order_by(desc(Prediction.created_at))).all()
    latest_prediction = {}
    for item in predictions: latest_prediction.setdefault(item.machine_id, item)
    return [MachineSummary(machine_id=k, latest_temperature=v.temperature, latest_vibration=v.vibration,
        latest_pressure=v.pressure, latest_rotational_speed=v.rotational_speed, operating_hours=v.operating_hours,
        latest_risk=latest_prediction[k].risk_score if k in latest_prediction else None,
        latest_risk_level=latest_prediction[k].risk_level if k in latest_prediction else "Unknown", updated_at=v.created_at)
        for k, v in latest.items()]
