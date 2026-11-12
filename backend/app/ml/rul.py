"""RUL inference for NASA C-MAPSS-style turbofan telemetry."""
from pathlib import Path
import joblib
import numpy as np
from app.core.config import get_settings

FEATURE_COLUMNS = (
    ["cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

class RULPredictor:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or get_settings().rul_model_file
        self._artifact = None
        self._mtime = None
        self._load()

    @property
    def available(self) -> bool:
        return self._artifact is not None

    def _load(self):
        if not self.model_path.exists():
            self._artifact = None
            self._mtime = None
            return
        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("Invalid RUL model artifact")
        if artifact.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("RUL model feature schema does not match this API")
        self._artifact = artifact
        self._mtime = self.model_path.stat().st_mtime

    def predict(self, cycle: int, settings: list[float], sensors: list[float]) -> tuple[float, int]:
        if self.model_path.exists() and self.model_path.stat().st_mtime != self._mtime:
            self._load()
        if self._artifact is None:
            raise FileNotFoundError("RUL model is not trained. Train it with NASA C-MAPSS FD001 data.")
        row = np.asarray([[cycle, *settings, *sensors]], dtype=float)
        prediction = float(self._artifact["model"].predict(row)[0])
        cap = int(self._artifact.get("target_cap_cycles", 125))
        return round(float(np.clip(prediction, 0, cap)), 2), cap
