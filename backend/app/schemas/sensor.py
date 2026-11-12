from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class SensorReadingCreate(BaseModel):
    machine_id: str = Field(min_length=1, max_length=80)
    temperature: float = Field(ge=-50, le=500)
    vibration: float = Field(ge=0, le=100)
    pressure: float = Field(ge=0, le=1000)
    rotational_speed: float = Field(ge=0, le=100000)
    operating_hours: float = Field(ge=0, le=10000000)

class SensorReadingOut(SensorReadingCreate):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PredictionOut(BaseModel):
    id: int
    reading_id: int
    machine_id: str
    risk_score: float
    risk_level: str
    predicted_failure: bool
    model_source: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class MachineSummary(BaseModel):
    machine_id: str
    latest_temperature: float
    latest_vibration: float
    latest_pressure: float
    latest_rotational_speed: float
    operating_hours: float
    latest_risk: float | None = None
    latest_risk_level: str = "Unknown"
    updated_at: datetime
