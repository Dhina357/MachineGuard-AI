import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from app.ml.features import FEATURE_COLUMNS

def train(input_path: Path, output_path: Path) -> dict:
    data = pd.read_csv(input_path)
    required = FEATURE_COLUMNS + ["failure"]
    missing = [c for c in required if c not in data.columns]
    if missing: raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data[required].dropna()
    if data.empty: raise ValueError("No complete training rows")
    if not set(data.failure.unique()).issubset({0, 1, False, True}): raise ValueError("failure must be binary 0/1")
    if data.failure.nunique() < 2: raise ValueError("Training requires both target classes")
    x, y = data[FEATURE_COLUMNS], data.failure.astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
    model.fit(xtr, ytr)
    probabilities = model.predict_proba(xte)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    metrics = {"rows": len(data), "train_rows": len(xtr), "test_rows": len(xte),
      "accuracy": float(accuracy_score(yte, predicted)),
      "precision": float(precision_score(yte, predicted, zero_division=0)),
      "recall": float(recall_score(yte, predicted, zero_division=0)),
      "f1": float(f1_score(yte, predicted, zero_division=0)),
      "roc_auc": float(roc_auc_score(yte, probabilities)) if yte.nunique() == 2 else None,
      "warning": "Metrics describe only the supplied data, not production performance."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output), indent=2))
