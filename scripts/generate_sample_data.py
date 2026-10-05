"""Create a SYNTHETIC dataset with the same schema as the real 'Charging Record' CSV.
Only for smoke-testing the pipeline. Replace with the real dataset for final results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

rng = np.random.default_rng(42)
N = 72000
companies = ["ChargePoint", "Tesla", "EVgo", "Shell Recharge"]
locations = ["Downtown", "Airport", "Mall", "University", "Highway", "Residential"]
types = {"AC Level 2": 7, "DC Fast": 50, "DC Ultra": 120}

chargers = pd.DataFrame({
    "ChargerID": [f"CH{i:03d}" for i in range(60)],
    "ChargerCompany": rng.choice(companies, 60),
    "Location": rng.choice(locations, 60),
    "ChargerType": rng.choice(list(types), 60, p=[0.5, 0.35, 0.15]),
})
start = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 24 * 60 * 540, N), unit="m")
c = chargers.sample(N, replace=True, random_state=1).reset_index(drop=True)
hour = start.hour
peak = 1 + 0.5 * np.exp(-((hour - 18) ** 2) / 8) + 0.3 * np.exp(-((hour - 9) ** 2) / 6)
duration = np.clip(rng.gamma(2.5, 0.5, N) * peak, 0.1, 12)
power = c["ChargerType"].map(types).values
trend = 1 + (start - start.min()).days / 900
demand = np.clip(duration * power * rng.uniform(0.3, 0.8, N) * trend / 3, 0.5, None)
df = pd.DataFrame({
    "UserID": rng.integers(1, 5000, N), **c.to_dict("list"),
    "StartDatetime": start, "EndDatetime": start + pd.to_timedelta(duration, unit="h"),
    "Duration": duration.round(3), "Demand": demand.round(3),
})
df["StartDay"] = df["StartDatetime"].dt.date
df["StartTime"] = df["StartDatetime"].dt.time
df["EndDay"] = df["EndDatetime"].dt.date
df["EndTime"] = df["EndDatetime"].dt.time
C.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(C.RAW_PATH, index=False)
print(f"Wrote {len(df)} synthetic rows -> {C.RAW_PATH}")
