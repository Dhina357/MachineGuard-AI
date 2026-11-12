import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic rows for software demos only, not model validation."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(68, 12, rows).clip(25, 140)
    vibration = rng.gamma(2.0, .22, rows).clip(0, 3)
    pressure = rng.normal(7, 1.4, rows).clip(1, 14)
    speed = rng.normal(1500, 180, rows).clip(500, 2600)
    hours = rng.uniform(0, 30000, rows)
    latent = .045*(temperature-65) + 1.25*(vibration-.5) + .14*np.abs(pressure-7) + .000015*(hours-10000)
    failure = rng.binomial(1, 1/(1+np.exp(-latent)))
    return pd.DataFrame({"temperature":temperature,"vibration":vibration,"pressure":pressure,
        "rotational_speed":speed,"operating_hours":hours,"failure":failure})

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("data/sample_sensor_data.csv"))
    parser.add_argument("--rows",type=int,default=2000)
    args=parser.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    generate(args.rows).to_csv(args.output,index=False)
    print(f"Wrote {args.rows} synthetic rows to {args.output}")
