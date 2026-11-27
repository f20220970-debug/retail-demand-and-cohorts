# Retail demand and cohorts

Built for a **Victoria's Secret PS-II (Bengaluru)** application, against the two project briefs on the notice: *time-series forecasting for demand planning* and *dynamic micro-cohorting and micro-segmentation of customers*.

🌐 **[Live results](https://f20220970-debug.github.io/retail-demand-and-cohorts/)** · [Forecasting](src/demand.py) · [Segmentation](src/segments.py) · [15 tests](tests/test_findings.py)

---

## Two findings

**1. The machine-learning model loses to repeating last year.**

| Method | MASE | sMAPE | Error vs baseline |
| --- | ---: | ---: | ---: |
| SARIMA (1,1,1)(0,1,1)₁₂ | **1.536** | 8.0% | **−45%** |
| Holt-Winters (mult. seasonal) | 1.612 | 8.4% | −42% |
| Seasonal naive *(baseline)* | 2.802 | 15.3% | — |
| Gradient boosting, direct | **3.272** | 17.3% | **+17%** |

Expanding-window rolling origin: **24 origins × 12 horizons**, every method scored on identical origins, MASE scaled by the in-sample seasonal-naive error.

The brief asks about implementing state-of-the-art deep learning. On 204 monthly observations a gradient-boosted lag model is **17% worse than repeating last year's same month**, while two classical methods cut error by over 40%. More capacity than the data can identify is not an improvement, and the way to know which case you are in is a backtest, not a preference. The ranking also holds at every horizon from 1 to 12 months, which is worth checking rather than assuming — stock is committed at the long horizon, and a method that won at h=1 and lost at h=12 would change the recommendation.

**2. The estate splits in two, and not by size.**

| Segment | Stores | Share | Sales/m² | Margin | Hours/m² | Median m² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 74 | 18.5% | **11,194** | 39.5% | 1.60 | 80 |
| 1 | 326 | 81.5% | 4,959 | 39.0% | 0.79 | 140 |

Clustered on **rates rather than totals**, so the features are size-neutral by construction. The productive segment is **2.3× ahead on sales per m²** and has the *smaller* footprint. Margins sit within a point of each other, so this is an operations and staffing story rather than a pricing one.

## The two checks that decide whether any of it is usable

**Is it a size ranking in disguise?** Store size explains **14%** of the variance in segment membership (one-way η² on log size), under the 50% bar. So the segmentation carries information a size ranking does not. This is the check most segmentation work skips, and skipping it is how an estate gets "insights" that only say big stores differ from small ones.

**Does micro-segmentation survive the evidence?** Not here. Silhouette peaks at **k=2 (0.438)**, drops to 0.286 at k=3 and never recovers. Push to k=5 and the smallest segment holds **1.8% of the estate** — seven stores, too few to run a programme against — while stability falls from **0.79 to 0.65** adjusted Rand over 40 bootstrap refits. Cutting finer is a presentation choice, not a structural one. A cohort you re-cut every month has to be stable enough to act on, and that is measurable rather than assumed.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q        # 15 tests, ~70s
python src/build_docs.py   # rebuilds the live page from the models
```

The tests assert the findings, not the plumbing. If boosting starts beating the baseline, if the segments become a size proxy, or if the operating numbers move, the build breaks rather than the write-up quietly becoming wrong.

## Scope, stated rather than buried

Two real public datasets, each matched to one question: **204 months of monthly demand (1991–2008)** for the forecasting, and **400 Dutch menswear outlets (1990, `Ecdat::Clothing` via pydataset)** for the segmentation. They stand in for a retail planning problem; they are not evidence about any particular retailer, and the segmentation is of outlets rather than customers.

The forecast comparison is a single series, so it shows that model capacity has to match the data available — not that boosting always loses. With thousands of SKU-store series and exogenous drivers, the answer can go the other way, and the same backtest is how you would find out.

Two of the 48 model fits raised convergence warnings. They were kept rather than dropped, because silently discarding the hard origins is how a backtest flatters itself.

---

Built by Pavitra Seth.
