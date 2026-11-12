import numpy as np
import joblib
from pathlib import Path
from app.core.config import get_settings
from app.ml.features import to_feature_row

class FailurePredictor:
    """Uses a trained sklearn model, otherwise a labelled demo-only heuristic."""
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or get_settings().model_file
        self._model = None
        self._mtime = None
        self._load()

    @property
    def source(self):
        return "trained_model" if self._model is not None else "demo_heuristic"

    def _load(self):
        if not self.model_path.exists():
            self._model = None
            return
        try:
            model = joblib.load(self.model_path)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model must support predict_proba")
            self._model = model
            self._mtime = self.model_path.stat().st_mtime
        except (OSError, ValueError, EOFError):
            self._model = None

    def predict(self, reading):
        if self.model_path.exists() and self.model_path.stat().st_mtime != self._mtime:
            self._load()
        row = np.asarray([to_feature_row(reading)], dtype=float)
        if self._model is not None:
            score = float(self._model.predict_proba(row)[0][1])
            source = "trained_model"
        else:
            # Illustrative thresholds only; not calibrated for any actual machine.
            score = (0.4 * np.clip((reading.temperature - 65) / 55, 0, 1)
                     + 0.4 * np.clip(reading.vibration / 1.5, 0, 1)
                     + 0.2 * np.clip(abs(reading.pressure - 7) / 7, 0, 1))
            score, source = float(np.clip(score, 0, 1)), "demo_heuristic"
        return score, score >= 0.5, source

def risk_level(score: float) -> str:
    if score >= 0.7: return "High"
    if score >= 0.35: return "Medium"
    return "Low"
