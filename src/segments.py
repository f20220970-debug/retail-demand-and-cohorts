"""Micro-segmentation of retail outlets, with the check that usually isn't run.

Segmentation is easy to do and easy to fool yourself with: cluster any retail
estate on unscaled figures and you rediscover size, dressed up as insight. The
segments look plausible, the deck gets approved, and every recommendation is
really "big stores are different from small stores."

So this does two things. It segments on *intensity* features -- per square
metre, per hour -- which are size-neutral by construction. And it then tests
whether the segments are a size proxy anyway, and reports the answer either
way rather than hoping nobody checks.

Data: 400 men's fashion stores, Netherlands, 1990 (R `Ecdat::Clothing`), via
pydataset. Real outlets, not simulated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "clothing.csv"

#: Candidate segment counts. Two is not a segmentation; past six, an area
#: manager cannot hold the playbook in their head.
K_RANGE = (2, 3, 4, 5, 6)
RANDOM_STATE = 0

#: Size-neutral by construction: every feature is a rate, not a total.
FEATURES = ["sales_per_sqm", "margin", "hours_per_sqm", "investment_per_sqm"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    out = pd.DataFrame({
        "sales_per_sqm": df["sales"],
        "margin": df["margin"],
        "hours_per_sqm": df["hoursw"] / df["ssize"],
        "investment_per_sqm": (df["inv1"] + df["inv2"]) / df["ssize"],
        "ssize": df["ssize"],
        "annual_sales": df["tsales"],
    })
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _scaled(df: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(df[FEATURES].to_numpy(dtype=float))


@dataclass(frozen=True)
class Choice:
    k: int
    silhouette: float


def choose_k(df: pd.DataFrame | None = None) -> list[Choice]:
    """Silhouette across candidate k. Reported in full, not just the argmax."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    df = load() if df is None else df
    X = _scaled(df)
    out = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=RANDOM_STATE).fit_predict(X)
        out.append(Choice(k, round(float(silhouette_score(X, labels)), 4)))
    return out


def best_k(df: pd.DataFrame | None = None) -> int:
    return max(choose_k(df), key=lambda c: c.silhouette).k


def segment(df: pd.DataFrame | None = None, k: int | None = None) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    df = load() if df is None else df
    k = best_k(df) if k is None else k
    X = _scaled(df)
    df = df.copy()
    df["segment"] = KMeans(n_clusters=k, n_init=10,
                           random_state=RANDOM_STATE).fit_predict(X)
    return df


def profile(df: pd.DataFrame | None = None, k: int | None = None) -> pd.DataFrame:
    """What each segment looks like. Medians, because retail estates have tails."""
    d = segment(df, k)
    out = (d.groupby("segment")
             .agg(stores=("ssize", "size"),
                  sales_per_sqm=("sales_per_sqm", "median"),
                  margin=("margin", "median"),
                  hours_per_sqm=("hours_per_sqm", "median"),
                  investment_per_sqm=("investment_per_sqm", "median"),
                  median_store_sqm=("ssize", "median"))
             .round(2))
    out["share_of_estate"] = (out["stores"] / out["stores"].sum()).round(4)
    return out.reset_index()


def is_a_size_proxy(df: pd.DataFrame | None = None, k: int | None = None,
                    threshold: float = 0.5) -> dict:
    """The check that decides whether this segmentation says anything new.

    If store size alone predicts the segment, the segmentation is a size
    ranking wearing a costume. Measured as the share of variance in log size
    explained by segment membership (one-way eta-squared).
    """
    d = segment(df, k)
    logsize = np.log(d["ssize"].to_numpy(dtype=float))
    grand = logsize.mean()
    ss_total = float(((logsize - grand) ** 2).sum())
    ss_between = 0.0
    for _, grp in d.groupby("segment"):
        v = np.log(grp["ssize"].to_numpy(dtype=float))
        ss_between += len(v) * (v.mean() - grand) ** 2
    eta_sq = ss_between / ss_total
    return {
        "eta_squared_on_log_size": round(float(eta_sq), 4),
        "threshold": threshold,
        "is_size_proxy": bool(eta_sq > threshold),
    }


def stability(df: pd.DataFrame | None = None, k: int | None = None,
              n_resamples: int = 40) -> dict:
    """Would these segments survive next quarter's data?

    A cohort you re-cut every month is only actionable if the cut is stable.
    Each resample refits k-means on a bootstrap of the estate, relabels every
    store, and scores agreement with the base solution by adjusted Rand index
    (0 = chance, 1 = identical).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    df = load() if df is None else df
    k = best_k(df) if k is None else k
    scaler = StandardScaler().fit(df[FEATURES].to_numpy(dtype=float))
    X = scaler.transform(df[FEATURES].to_numpy(dtype=float))
    base = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)

    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(n_resamples):
        idx = rng.choice(len(X), size=len(X), replace=True)
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=RANDOM_STATE).fit(X[idx])
        scores.append(adjusted_rand_score(base, km.predict(X)))
    return {
        "k": k,
        "mean_adjusted_rand": round(float(np.mean(scores)), 4),
        "min_adjusted_rand": round(float(np.min(scores)), 4),
        "n_resamples": n_resamples,
    }


def spread(df: pd.DataFrame | None = None, k: int | None = None) -> dict:
    """How far apart the best and worst segments are on productivity.

    This is the number that says whether acting on the segmentation is worth
    anyone's time.
    """
    p = profile(df, k)
    hi = float(p["sales_per_sqm"].max())
    lo = float(p["sales_per_sqm"].min())
    return {
        "best_segment_sales_per_sqm": round(hi, 2),
        "worst_segment_sales_per_sqm": round(lo, 2),
        "ratio": round(hi / lo, 2),
    }
