"""Demand forecasting, evaluated the way a planner would have to live with it.

The question is not "which model has the lowest error on a held-out tail" --
one split is one draw. It is "which model would have been better, month after
month, across every origin a planner actually forecast from, at the horizon
the supply chain commits at."

So: rolling-origin (expanding-window) backtest, every method scored on the
same origins and the same horizons, with a scale-free metric.

MASE is the metric, scaled by the in-sample seasonal-naive error. MASE < 1
means the method beat a seasonal naive forecast; MASE > 1 means it did not,
however impressive its sMAPE looks.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data" / "a10.csv"

#: Monthly data, so the seasonal period is 12.
PERIOD = 12
#: Longest horizon scored. A planner committing stock 12 months out needs h=12
#: to be honest, not just h=1.
MAX_H = 12
#: Number of rolling origins. Each origin must have MAX_H actuals after it.
N_ORIGINS = 24


def load() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    s = pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
    return s.asfreq("MS")


# --- metrics ------------------------------------------------------------------

def mase_scale(train: np.ndarray, period: int = PERIOD) -> float:
    """In-sample seasonal-naive MAE: the denominator that makes MASE scale-free."""
    return float(np.mean(np.abs(train[period:] - train[:-period])))


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    return float(np.mean(np.abs(actual - pred) / denom) * 100)


# --- methods ------------------------------------------------------------------
# Each takes the history and a horizon, and returns h point forecasts.

def seasonal_naive(train: np.ndarray, h: int) -> np.ndarray:
    """Last year's same month. The baseline everything must beat to earn its keep."""
    return np.array([train[-PERIOD + (i % PERIOD)] for i in range(h)])


def holt_winters(train: np.ndarray, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # Multiplicative seasonality: this series' seasonal swing grows with level.
    m = ExponentialSmoothing(train, trend="add", seasonal="mul",
                             seasonal_periods=PERIOD,
                             initialization_method="estimated").fit()
    return np.asarray(m.forecast(h))


def sarima(train: np.ndarray, h: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Modelled in logs, so the multiplicative seasonality becomes additive.
    m = SARIMAX(np.log(train), order=(1, 1, 1),
                seasonal_order=(0, 1, 1, PERIOD),
                enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False)
    return np.exp(np.asarray(m.forecast(h)))


def _lag_matrix(y: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    start = max(lags)
    rows, targets = [], []
    for t in range(start, len(y)):
        rows.append([y[t - L] for L in lags])
        targets.append(y[t])
    return np.array(rows), np.array(targets)


def gbm_direct(train: np.ndarray, h: int) -> np.ndarray:
    """Gradient boosting on lag features, one model per horizon (direct strategy).

    Direct rather than recursive: recursive multi-step compounds its own error,
    and at h=12 that difference dominates whatever the learner is doing.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    lags = [1, 2, 3, 11, 12, 13, 24]
    preds = []
    for step in range(1, h + 1):
        y = train
        start = max(lags) + step - 1
        rows, targets = [], []
        for t in range(start, len(y)):
            rows.append([y[t - L - step + 1] for L in lags])
            targets.append(y[t])
        X, yy = np.array(rows), np.array(targets)
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=3,
            min_samples_leaf=5, random_state=0).fit(X, yy)
        feat = np.array([[y[len(y) - L] for L in lags]])
        preds.append(float(model.predict(feat)[0]))
    return np.array(preds)


METHODS = {
    "seasonal_naive": seasonal_naive,
    "holt_winters": holt_winters,
    "sarima": sarima,
    "gbm_direct": gbm_direct,
}


# --- the backtest -------------------------------------------------------------

@dataclass(frozen=True)
class Score:
    method: str
    horizon: int
    mase: float
    smape: float
    n_origins: int


def backtest(series: pd.Series | None = None,
             n_origins: int = N_ORIGINS,
             max_h: int = MAX_H) -> pd.DataFrame:
    """Expanding-window rolling-origin evaluation over every method."""
    y = (series if series is not None else load()).to_numpy(dtype=float)
    T = len(y)
    # Each origin needs max_h actuals after it.
    origins = list(range(T - max_h - n_origins + 1, T - max_h + 1))

    records = []
    for method, fn in METHODS.items():
        errs = {h: [] for h in range(1, max_h + 1)}
        scaled = {h: [] for h in range(1, max_h + 1)}
        for o in origins:
            train = y[:o]
            actual = y[o:o + max_h]
            pred = fn(train, max_h)
            scale = mase_scale(train)
            for h in range(1, max_h + 1):
                errs[h].append((actual[h - 1], pred[h - 1]))
                scaled[h].append(abs(actual[h - 1] - pred[h - 1]) / scale)
        for h in range(1, max_h + 1):
            a = np.array([x[0] for x in errs[h]])
            p = np.array([x[1] for x in errs[h]])
            records.append(Score(method, h, round(float(np.mean(scaled[h])), 4),
                                 round(smape(a, p), 2), len(origins)).__dict__)
    return pd.DataFrame(records)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MASE per method across all horizons, best first."""
    out = (df.groupby("method")
             .agg(mase=("mase", "mean"), smape=("smape", "mean"))
             .round(4).sort_values("mase"))
    return out.reset_index()


def by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """MASE per method at each horizon -- where the ranking changes, if it does."""
    return df.pivot(index="horizon", columns="method", values="mase").round(4)


def winner_at(df: pd.DataFrame, horizon: int) -> str:
    sub = df[df["horizon"] == horizon].sort_values("mase")
    return str(sub.iloc[0]["method"])


def beats_baseline(df: pd.DataFrame) -> dict[str, bool]:
    """Which methods actually earned their complexity, averaged over horizons."""
    s = summary(df).set_index("method")["mase"]
    base = s["seasonal_naive"]
    return {m: bool(v < base) for m, v in s.items() if m != "seasonal_naive"}
