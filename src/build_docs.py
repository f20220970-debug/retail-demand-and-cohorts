"""Rebuild docs/index.html from the models.  python src/build_docs.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import demand as d  # noqa: E402
import segments as s  # noqa: E402

OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/f20220970-debug/retail-demand-and-cohorts"

LABEL = {
    "sarima": "SARIMA (1,1,1)(0,1,1)<sub>12</sub>",
    "holt_winters": "Holt-Winters (mult. seasonal)",
    "seasonal_naive": "Seasonal naive <em>(baseline)</em>",
    "gbm_direct": "Gradient boosting, direct",
}


def main() -> int:
    bt = d.backtest()
    summ = d.summary(bt).set_index("method")
    horizon = d.by_horizon(bt)
    prof = s.profile()
    proxy = s.is_a_size_proxy()
    stab2, stab5 = s.stability(), s.stability(k=5)
    sil = {c.k: c.silhouette for c in s.choose_k()}
    sp = s.spread()

    base = summ.loc["seasonal_naive", "mase"]
    best = summ.loc["sarima", "mase"]
    gbm = summ.loc["gbm_direct", "mase"]

    rows = ""
    for m in summ.index:
        cls = " class=hi" if m == "gbm_direct" else ""
        vs = (summ.loc[m, "mase"] / base - 1) * 100
        verdict = (f"{vs:+.0f}%" if m != "seasonal_naive" else "&mdash;")
        rows += (f"<tr{cls}><td>{LABEL[m]}</td>"
                 f"<td class='n'>{summ.loc[m, 'mase']:.3f}</td>"
                 f"<td class='n'>{summ.loc[m, 'smape']:.1f}%</td>"
                 f"<td class='n'>{verdict}</td></tr>")

    hrows = "".join(
        f"<tr><td class='n'>{h}</td>"
        + "".join(f"<td class='n'>{horizon.loc[h, m]:.2f}</td>"
                  for m in ("sarima", "holt_winters", "seasonal_naive", "gbm_direct"))
        + "</tr>"
        for h in (1, 3, 6, 9, 12))

    srows = "".join(
        f"<tr><td>Segment {int(r.segment)}</td>"
        f"<td class='n'>{int(r.stores)}</td>"
        f"<td class='n'>{r.share_of_estate * 100:.1f}%</td>"
        f"<td class='n'>{r.sales_per_sqm:,.0f}</td>"
        f"<td class='n'>{r.margin:.1f}%</td>"
        f"<td class='n'>{r.hours_per_sqm:.2f}</td>"
        f"<td class='n'>{r.median_store_sqm:,.0f}</td></tr>"
        for r in prof.itertuples())

    krows = "".join(
        f"<tr{' class=hi' if k == 2 else ''}><td class='n'>{k}</td>"
        f"<td class='n'>{v:.4f}</td></tr>" for k, v in sil.items())

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retail Demand &amp; Cohorts</title>
<style>
:root{{--bg:#fbfaf9;--panel:#fff;--ink:#17171a;--muted:#6a6a72;--line:#e7e4e0;
  --accent:#8a1f45;--warn:#b4411f}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#131318;--panel:#1c1c23;--ink:#eceaf0;--muted:#9b99a3;--line:#2b2b35;
  --accent:#e88fb0;--warn:#e87f52}}}}
:root[data-theme="dark"]{{--bg:#131318;--panel:#1c1c23;--ink:#eceaf0;--muted:#9b99a3;
  --line:#2b2b35;--accent:#e88fb0;--warn:#e87f52}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif}}
.wrap{{max-width:880px;margin:0 auto;padding:48px 16px 80px}}
h1{{font-size:clamp(1.7rem,4.5vw,2.4rem);line-height:1.15;margin:0 0 8px;letter-spacing:-.025em}}
h2{{font-size:1.15rem;margin:46px 0 6px;letter-spacing:-.01em}}
.sub{{color:var(--muted);margin:0 0 6px}}
.lede{{color:var(--muted);max-width:64ch;margin:0 0 26px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;margin:12px 0;overflow-x:auto}}
.kpis{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:24px 0}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.kpi .n{{font-size:1.55rem;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.kpi .l{{color:var(--muted);font-size:.82rem;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
td.n,th.n{{text-align:right}}
th{{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.05em}}
tbody tr:last-child td{{border-bottom:none}}
tr.hi td{{color:var(--warn);font-weight:600}}
.note{{font-size:.88rem;color:var(--muted);max-width:66ch}}
a{{color:var(--accent)}}
</style></head><body><div class="wrap">

<h1>Retail demand and cohorts</h1>
<p class="sub">Rolling-origin forecast evaluation &middot; size-neutral store segmentation</p>
<p class="lede">Two questions a retail analytics team has to answer with evidence rather
than preference: does a more modern forecasting model actually plan stock better, and does a
segmentation say anything that store size does not already say?</p>

<div class="kpis">
  <div class="kpi"><div class="n">{best:.2f}</div><div class="l">MASE, best model (SARIMA)</div></div>
  <div class="kpi"><div class="n">{(1 - best / base) * 100:.0f}%</div><div class="l">error cut against the naive baseline</div></div>
  <div class="kpi"><div class="n">{gbm:.2f}</div><div class="l">MASE, gradient boosting &mdash; <em>worse than naive</em></div></div>
  <div class="kpi"><div class="n">{proxy['eta_squared_on_log_size']:.2f}</div><div class="l">of segment variance explained by store size</div></div>
</div>

<h2>The machine-learning model loses to repeating last year</h2>
<div class="panel">
<table><thead><tr><th>Method</th><th class='n'>MASE</th><th class='n'>sMAPE</th>
<th class='n'>Error vs baseline</th></tr></thead><tbody>{rows}</tbody></table>
</div>
<p class="note">Expanding-window rolling origin: <strong>24 origins &times; 12 horizons</strong>,
every method scored on identical origins. MASE is scaled by the in-sample seasonal-naive error,
so it is comparable across series and the <em>relative</em> ordering is what carries meaning.
<strong>Gradient boosting is {(gbm / base - 1) * 100:.0f}% worse than the naive baseline</strong>
on 204 monthly observations &mdash; more model capacity than the data can identify. SARIMA and
Holt-Winters both earn their complexity; the modern method does not.</p>

<h2>And the ranking holds at every horizon</h2>
<div class="panel">
<table><thead><tr><th class='n'>h (months)</th><th class='n'>SARIMA</th>
<th class='n'>Holt-Winters</th><th class='n'>Seasonal naive</th>
<th class='n'>Gradient boosting</th></tr></thead><tbody>{hrows}</tbody></table>
</div>
<p class="note">Worth checking rather than assuming: a method that wins at one month ahead and
loses at twelve would change the recommendation, because stock is committed at the long horizon.
Here it does not change &mdash; but every method degrades with horizon, so a one-month-ahead
accuracy figure is not a twelve-month-ahead promise.</p>

<h2>The estate splits in two, and not by size</h2>
<div class="panel">
<table><thead><tr><th>Segment</th><th class='n'>Stores</th><th class='n'>Share</th>
<th class='n'>Sales/m&sup2;</th><th class='n'>Margin</th><th class='n'>Hours/m&sup2;</th>
<th class='n'>Median m&sup2;</th></tr></thead><tbody>{srows}</tbody></table>
</div>
<p class="note">400 real outlets, clustered on <strong>rates rather than totals</strong>
(sales per m&sup2;, margin, hours per m&sup2;, investment per m&sup2;), which makes the
features size-neutral by construction. The productive segment is
<strong>{sp['ratio']:.1f}&times;</strong> ahead on sales per m&sup2; &mdash; and is the
<em>smaller</em>-footprint group. Margins are within a point of each other across segments, so
this is an operations and staffing story, not a pricing one.</p>

<h2>Two checks that decide whether any of it is usable</h2>
<div class="panel">
<table><thead><tr><th class='n'>k</th><th class='n'>Silhouette</th></tr></thead>
<tbody>{krows}</tbody></table>
</div>
<p class="note"><strong>The ask is usually micro-segments; the evidence here supports two.</strong>
Silhouette drops sharply from k=2 to k=3 and never recovers, so cutting finer is a presentation
choice rather than a structural one. At k=5 the smallest segment holds
<strong>{s.profile(k=5)['share_of_estate'].min() * 100:.1f}% of the estate</strong> &mdash;
seven stores, too few to run a programme against &mdash; and stability falls from
<strong>{stab2['mean_adjusted_rand']:.2f}</strong> to
<strong>{stab5['mean_adjusted_rand']:.2f}</strong> adjusted Rand across 40 bootstrap refits.</p>
<p class="note"><strong>Is it just size in a costume?</strong> Store size explains
<strong>{proxy['eta_squared_on_log_size'] * 100:.0f}%</strong> of the variance in segment
membership. Under the 50% bar, so the segmentation carries information a size ranking does not.
This is the check that most segmentation decks skip.</p>

<h2>Scope, stated rather than buried</h2>
<p class="note">Two real public datasets, each matched to one question: 204 months of monthly
demand (1991&ndash;2008) for the forecasting, and 400 Dutch menswear outlets (1990,
<code>Ecdat::Clothing</code>) for the segmentation. They are stand-ins for a retail planning
problem, not evidence about any particular retailer. The forecast comparison is one series, so
it shows that capacity must match data, not that boosting always loses. Two of the 48 model fits
raised convergence warnings and were kept rather than dropped, since silently discarding the
hard origins is how a backtest flatters itself.</p>

<p class="note" style="margin-top:40px">
Built by Pavitra Seth. &middot;
<a href="{REPO}/blob/main/src/demand.py">Forecasting</a> &middot;
<a href="{REPO}/blob/main/src/segments.py">Segmentation</a> &middot;
<a href="{REPO}/blob/main/tests/test_findings.py">15 tests</a> &middot;
<a href="{REPO}">Repository</a>
</p>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
