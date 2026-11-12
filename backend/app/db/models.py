from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    temperature: Mapped[float] = mapped_column(Float)
    vibration: Mapped[float] = mapped_column(Float)
    pressure: Mapped[float] = mapped_column(Float)
    rotational_speed: Mapped[float] = mapped_column(Float)
    operating_hours: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    reading_id: Mapped[int] = mapped_column(Integer, index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(20))
    predicted_failure: Mapped[bool] = mapped_column(Integer)
    model_source: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
