"""Train an RUL regressor using NASA C-MAPSS FD001 run-to-failure data.

Expected input: train_FD001.txt (space-separated, no header; 26 columns).
Download the official NASA C-MAPSS dataset separately; do not commit the dataset.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from app.ml.rul import FEATURE_COLUMNS

RAW_COLUMNS = ["unit_nr", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]

def train(input_path: Path, output_path: Path, target_cap: int = 125) -> dict:
    frame = pd.read_csv(input_path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError("Expected a 26-column NASA C-MAPSS FD001 training file")
    max_cycles = frame.groupby("unit_nr")["cycle"].max()
    frame["rul"] = frame["unit_nr"].map(max_cycles) - frame["cycle"]
    frame["rul"] = frame["rul"].clip(upper=target_cap)
    X, y, groups = frame[list(FEATURE_COLUMNS)], frame["rul"], frame["unit_nr"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups))
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, max_features=0.8,
                                  n_jobs=-1, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])
    metrics = {
        "mae_cycles": round(float(mean_absolute_error(y.iloc[test_idx], predictions)), 3),
        "rmse_cycles": round(float(mean_squared_error(y.iloc[test_idx], predictions) ** 0.5), 3),
        "train_engines": int(groups.iloc[train_idx].nunique()),
        "test_engines": int(groups.iloc[test_idx].nunique()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS,
                 "target_cap_cycles": target_cap, "dataset": "NASA C-MAPSS FD001",
                 "evaluation": metrics}, output_path)
    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to train_FD001.txt")
    parser.add_argument("--output", type=Path, default=Path("../models/rul_model.joblib"))
    parser.add_argument("--target-cap", type=int, default=125)
    args = parser.parse_args()
    metrics = train(args.input, args.output, args.target_cap)
    print(f"Saved RUL model to {args.output}")
    print(f"Engine-group holdout metrics: {metrics}")

if __name__ == "__main__":
    main()
