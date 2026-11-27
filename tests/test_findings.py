"""Tests that fail if the findings stop holding.

Each asserts a claim the write-up makes. A change to the backtest, the
features or the clustering that would make the report wrong breaks the build
instead of shipping quietly.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import demand as d  # noqa: E402
import segments as s  # noqa: E402


@pytest.fixture(scope="module")
def bt():
    return d.backtest()


@pytest.fixture(scope="module")
def summary(bt):
    return d.summary(bt).set_index("method")["mase"]


# --- forecasting: the data is real and the evaluation is honest ---------------

def test_the_series_is_the_full_published_history():
    y = d.load()
    assert len(y) == 204
    assert str(y.index[0].date()) == "1991-07-01"
    assert str(y.index[-1].date()) == "2008-06-01"


def test_every_method_is_scored_on_the_same_origins_and_horizons(bt):
    assert set(bt["method"]) == set(d.METHODS)
    assert sorted(bt["horizon"].unique()) == list(range(1, 13))
    assert set(bt["n_origins"]) == {24}
    # 4 methods x 12 horizons
    assert len(bt) == 48


# --- the finding that matters to a demand-planning team -----------------------

def test_the_machine_learning_model_loses_to_the_naive_baseline(summary):
    """The headline.

    A gradient-boosted lag model, given 204 monthly observations, is worse
    than repeating last year's same month. More capacity than data is not an
    improvement, and this is the number that says so.
    """
    assert summary["gbm_direct"] > summary["seasonal_naive"]
    assert summary["gbm_direct"] == pytest.approx(3.27, abs=0.15)
    assert summary["seasonal_naive"] == pytest.approx(2.80, abs=0.10)


def test_the_classical_methods_do_earn_their_complexity(summary):
    beats = d.beats_baseline(d.backtest())
    assert beats["sarima"] is True
    assert beats["holt_winters"] is True
    assert beats["gbm_direct"] is False


def test_sarima_is_the_best_method_overall(summary):
    assert summary.idxmin() == "sarima"
    assert summary["sarima"] == pytest.approx(1.54, abs=0.12)


def test_sarima_roughly_halves_the_baseline_error(summary):
    reduction = 1 - summary["sarima"] / summary["seasonal_naive"]
    assert reduction > 0.40


def test_accuracy_decays_with_horizon_for_every_serious_method(bt):
    """A one-month-ahead number is not a twelve-month-ahead promise."""
    table = d.by_horizon(bt)
    for method in ("sarima", "holt_winters", "seasonal_naive"):
        assert table[method].iloc[-1] > table[method].iloc[0]


def test_the_winner_does_not_change_with_horizon(bt):
    """Worth asserting: if it ever does, the recommendation has to change."""
    assert {d.winner_at(bt, h) for h in (1, 6, 12)} == {"sarima"}


# --- segmentation: real stores, size-neutral features -------------------------

def test_the_estate_is_the_full_published_sample():
    df = s.load()
    assert len(df) == 400
    assert list(s.FEATURES) == ["sales_per_sqm", "margin",
                                "hours_per_sqm", "investment_per_sqm"]


def test_silhouette_prefers_two_segments_not_many():
    """The ask was micro-segmentation. The evidence supports two macro-cohorts.

    Silhouette drops sharply from k=2 to k=3 and never recovers, so cutting
    finer is a presentation choice, not a structural one.
    """
    scores = {c.k: c.silhouette for c in s.choose_k()}
    assert s.best_k() == 2
    assert scores[2] > scores[3]
    assert scores[2] == pytest.approx(0.4378, abs=0.01)
    assert max(scores[k] for k in (3, 4, 5, 6)) < scores[2]


def test_the_segmentation_is_not_a_size_ranking_in_disguise():
    """The check that decides whether any of this says something new."""
    v = s.is_a_size_proxy()
    assert v["is_size_proxy"] is False
    assert v["eta_squared_on_log_size"] < 0.2


def test_the_productive_segment_is_the_smaller_stores():
    """Counter to the intuition that bigger stores are better stores."""
    p = s.profile().set_index("segment")
    best = p["sales_per_sqm"].idxmax()
    worst = p["sales_per_sqm"].idxmin()
    assert p.loc[best, "median_store_sqm"] < p.loc[worst, "median_store_sqm"]


def test_the_segments_differ_on_productivity_not_on_pricing():
    """Margins are nearly identical, so this is an operations story, not a price one."""
    p = s.profile()
    assert abs(p["margin"].max() - p["margin"].min()) <= 1.0
    assert s.spread()["ratio"] > 2.0


def test_segments_are_stable_enough_to_act_on():
    v = s.stability()
    assert v["mean_adjusted_rand"] > 0.7


def test_cutting_finer_costs_stability_and_buys_an_unusable_segment():
    """k=5 produces a segment too small to run a programme against."""
    assert s.stability(k=5)["mean_adjusted_rand"] < s.stability(k=2)["mean_adjusted_rand"]
    smallest = s.profile(k=5)["share_of_estate"].min()
    assert smallest < 0.02
