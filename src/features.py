"""Feature engineering and preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

TIME_NUM = ["hour", "day", "month", "dayofweek", "is_weekend", "week_of_year", "hour_sin", "hour_cos"]
HISTORY_NUM = ["charger_prev_demand", "charger_roll_mean_5", "charger_session_idx",
               "user_prior_sessions", "hours_since_last_charger_session"]
CAT_FEATURES = C.CAT_BASE + ["time_of_day"]


def _time_of_day(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 16, 20, 24]
    labels = ["Night", "Morning", "Afternoon", "Evening", "Late Evening"]
    return pd.cut(hour, bins=bins, labels=labels).astype(str)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["StartDatetime"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["time_of_day"] = _time_of_day(df["hour"])
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Usage-history features. Every value only uses sessions BEFORE the current
    one (shift(1)), so no information from the current/future row leaks in."""
    df = df.sort_values("StartDatetime").copy()
    g = df.groupby("ChargerID")[C.TARGET]
    df["charger_prev_demand"] = g.shift(1)
    df["charger_roll_mean_5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["charger_session_idx"] = df.groupby("ChargerID").cumcount()
    df["user_prior_sessions"] = df.groupby("UserID").cumcount()
    prev_start = df.groupby("ChargerID")["StartDatetime"].shift(1)
    df["hours_since_last_charger_session"] = (df["StartDatetime"] - prev_start).dt.total_seconds() / 3600
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_history_features(add_time_features(df))


def feature_lists():
    num = TIME_NUM + HISTORY_NUM + (["Duration"] if C.USE_DURATION_AS_FEATURE else [])
    return num, CAT_FEATURES


def make_preprocessor() -> ColumnTransformer:
    num, cat = feature_lists()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def chronological_split(df: pd.DataFrame, test_size: float = C.TEST_SIZE):
    """Time-ordered split: train on the past, test on the future (prevents leakage)."""
    df = df.sort_values("StartDatetime").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def get_Xy(df: pd.DataFrame):
    num, cat = feature_lists()
    return df[num + cat], df[C.TARGET]


def build_inference_row(history: pd.DataFrame, charger_id: str, start_dt, company=None, location=None, ctype=None):
    """Create one model-ready row for the dashboard from a chosen charger + datetime."""
    h = history[history["ChargerID"] == charger_id].sort_values("StartDatetime")
    last = h.iloc[-1]
    row = pd.DataFrame([{
        "ChargerID": charger_id, "UserID": "dash_user", "StartDatetime": pd.Timestamp(start_dt),
        "ChargerCompany": company or last["ChargerCompany"],
        "Location": location or last["Location"],
        "ChargerType": ctype or last["ChargerType"],
        "Duration": history["Duration"].median(),
    }])
    row = add_time_features(row)
    row["charger_prev_demand"] = last[C.TARGET]
    row["charger_roll_mean_5"] = h[C.TARGET].tail(5).mean()
    row["charger_session_idx"] = len(h)
    row["user_prior_sessions"] = history.groupby("UserID").size().median()
    row["hours_since_last_charger_session"] = (pd.Timestamp(start_dt) - last["StartDatetime"]).total_seconds() / 3600
    return row
