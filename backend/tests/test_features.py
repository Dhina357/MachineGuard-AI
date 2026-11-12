from types import SimpleNamespace
from app.ml.features import to_feature_row
from app.ml.predictor import risk_level

def test_feature_order():
    row = SimpleNamespace(temperature=70, vibration=.4, pressure=7, rotational_speed=1500, operating_hours=100)
    assert to_feature_row(row) == [70.0, .4, 7.0, 1500.0, 100.0]

def test_risk_levels():
    assert risk_level(.1) == "Low"
    assert risk_level(.5) == "Medium"
    assert risk_level(.8) == "High"
