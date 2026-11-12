FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "rotational_speed", "operating_hours"]

def to_feature_row(reading) -> list[float]:
    return [float(getattr(reading, name)) for name in FEATURE_COLUMNS]
