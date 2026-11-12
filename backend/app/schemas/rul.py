from pydantic import BaseModel, Field

class RULPredictRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=80)
    cycle: int = Field(ge=1, le=100000)
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: list[float] = Field(min_length=21, max_length=21)

class RULPredictResponse(BaseModel):
    machine_id: str
    predicted_rul_cycles: float
    model_source: str
    target_cap_cycles: int
    warning: str
