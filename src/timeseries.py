"""Time-series analysis and forecasting (Moving Average, Exponential Smoothing)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from . import config as C


def to_series(df: pd.DataFrame, freq: str = C.TS_FREQ, agg: str = "sum") -> pd.Series:
    """Aggregate session demand into a regular time series (gaps filled with 0)."""
    s = df.set_index("StartDatetime")[C.TARGET].resample(freq).agg(agg)
    return s.fillna(0.0)


def seasonal_period(freq: str) -> int:
    return 7 if freq.upper().startswith("D") else 24


def decompose(series: pd.Series, freq: str = C.TS_FREQ):
    return seasonal_decompose(series, model="additive", period=seasonal_period(freq))


def adf_test(series: pd.Series) -> dict:
    stat, p, _, _, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {"ADF statistic": stat, "p-value": p, "critical values": crit, "stationary": bool(p < 0.05)}


def _metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def moving_average_forecast(train: pd.Series, steps: int, window: int = 7) -> np.ndarray:
    """Flat forecast from the mean of the last `window` observations."""
    return np.repeat(train.tail(window).mean(), steps)


def rolling_moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def forecast_models(train: pd.Series, steps: int, freq: str = C.TS_FREQ) -> dict:
    m = seasonal_period(freq)
    out = {"Naive (last value)": np.repeat(train.iloc[-1], steps),
           "Moving Average (7)": moving_average_forecast(train, steps, 7),
           "Moving Average (14)": moving_average_forecast(train, steps, 14)}
    out["Simple Exp. Smoothing"] = SimpleExpSmoothing(train, initialization_method="estimated").fit().forecast(steps).values
    try:
        hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=m,
                                  initialization_method="estimated").fit()
        out["Holt-Winters (additive)"] = hw.forecast(steps).values
    except Exception as exc:
        print(f"  [ts] Holt-Winters skipped: {exc}")
    return out


def evaluate_forecasts(series: pd.Series, holdout: int = C.TS_HOLDOUT, freq: str = C.TS_FREQ):
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    preds = forecast_models(train, holdout, freq)
    table = pd.DataFrame([{"Model": k, **_metrics(test.values, v)} for k, v in preds.items()])
    return table.sort_values("RMSE").reset_index(drop=True), test, preds


def future_forecast(series: pd.Series, best_model: str, steps: int = C.TS_HORIZON, freq: str = C.TS_FREQ) -> pd.DataFrame:
    vals = forecast_models(series, steps, freq).get(best_model)
    if vals is None:
        vals = moving_average_forecast(series, steps)
    idx = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq)
    return pd.DataFrame({"Datetime": idx, "Forecast": np.clip(vals, 0, None)})


def trend_periods(series: pd.Series) -> pd.DataFrame:
    """Label monthly change as increasing / decreasing."""
    monthly = series.resample("MS").sum().to_frame("Demand")
    monthly["Change %"] = monthly["Demand"].pct_change() * 100
    monthly["Direction"] = np.where(monthly["Change %"] > 0, "Increasing", "Decreasing")
    return monthly.dropna().reset_index()
