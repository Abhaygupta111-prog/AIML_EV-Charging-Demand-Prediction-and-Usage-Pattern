"""Data understanding and cleaning."""
import numpy as np
import pandas as pd

from . import config as C


def load_raw(path=C.RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def data_overview(df: pd.DataFrame) -> dict:
    """Quick data-understanding summary (used by notebook / dashboard)."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }


def _to_datetime(df: pd.DataFrame, dt_col: str, day_col: str, time_col: str) -> pd.Series:
    """Parse a datetime column; fall back to Day + Time columns if needed."""
    out = pd.to_datetime(df[dt_col], errors="coerce") if dt_col in df else pd.Series(pd.NaT, index=df.index)
    if out.isna().any() and day_col in df and time_col in df:
        fallback = pd.to_datetime(df[day_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        out = out.fillna(fallback)
    return out


def clean(df: pd.DataFrame, cap_quantile: float = 0.995) -> pd.DataFrame:
    """Handle duplicates, types, missing values and outliers."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    df["StartDatetime"] = _to_datetime(df, "StartDatetime", "StartDay", "StartTime")
    df["EndDatetime"] = _to_datetime(df, "EndDatetime", "EndDay", "EndTime")

    for col in ("Demand", "Duration"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rebuild Duration (hours) if missing
    if "Duration" not in df or df["Duration"].isna().all():
        df["Duration"] = (df["EndDatetime"] - df["StartDatetime"]).dt.total_seconds() / 3600

    # Drop rows that cannot be used
    df = df.dropna(subset=["StartDatetime", C.TARGET])
    df = df[df[C.TARGET] > 0]
    df = df[(df["EndDatetime"].isna()) | (df["EndDatetime"] >= df["StartDatetime"])]

    # Fill categorical gaps, numeric gaps with median
    for col in C.CAT_BASE:
        df[col] = df[col].astype(str).str.strip().replace({"nan": "Unknown"}).fillna("Unknown")
    df["Duration"] = df["Duration"].fillna(df["Duration"].median())
    for col in ("UserID", "ChargerID"):
        df[col] = df[col].astype(str)

    # Outliers: flag via IQR, cap extreme values (winsorise) for stable modelling
    q1, q3 = df[C.TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["Is_Outlier"] = ((df[C.TARGET] < q1 - 1.5 * iqr) | (df[C.TARGET] > q3 + 1.5 * iqr)).astype(int)
    cap = df[C.TARGET].quantile(cap_quantile)
    df[C.TARGET] = df[C.TARGET].clip(upper=cap)

    return df.sort_values("StartDatetime").reset_index(drop=True)


def save_processed(df: pd.DataFrame, path=C.PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed(path=C.PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["StartDatetime", "EndDatetime"])
