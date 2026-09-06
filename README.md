# Short-horizon directional forecasting for equities

A local research application that estimates the probability an equity closes higher over
the next 1 and 5 trading sessions, detects the prevailing market regime, and explains
each prediction. It is built as a methodology exercise: the interesting question is not
how high accuracy can be pushed, but whether an honest evaluation can distinguish a real
signal from noise at all.

**This is research and educational software. It is not financial advice, and nothing here
should be used to trade.**

---

## The problem

Let `C_t` be the split- and dividend-adjusted close of session `t`, and `F_t` everything
observable at or before that close. For a horizon `h` sessions the model estimates

```
p_t(h) = P( y_t(h) = 1 | F_t )
```

Deliberately **not** a price forecast. Predicting the level of a price series that is
close to a random walk produces impressive-looking charts and no information; a
calibrated probability over direction is a question that can actually be scored.

### Targets

| Name | Definition | Why it exists |
|---|---|---|
| `direction` | `1[ C_(t+h)/C_t - 1 > 0 ]` | The stated problem |
| `excess_direction` | `1[ r_t(h) - r_benchmark,t(h) > 0 ]` | Removes the equity drift that the raw target gives away for free |
| `direction_open` | Return measured from the next open rather than the signal close | Execution-realistic robustness check |
| `ternary` | `sign(r)` outside a volatility-scaled band, `0` inside | Separates real moves from noise without a hand-picked threshold |

The second target matters more than it looks. Measured across the 42-ticker universe from
2006 to 2026:

| Target | Base rate |
|---|---|
| `direction_1d` | 0.5180 |
| `excess_direction_1d` | 0.4953 |
| `direction_5d` | **0.5493** |
| `excess_direction_5d` | 0.5040 |

A model that always predicts "up" scores 54.9% on the raw 5-day target. Reporting that as
accuracy would be meaningless. Both targets are carried through every experiment so the
drift component is always visible.

### The ternary band

A fixed neutral band such as ±0.5% is not defensible: it is a large move in a calm market
and noise in a stressed one. Instead

```
tau_t = kappa * sigma_t * sqrt(h)
```

where `sigma_t` is an EWMA volatility (RiskMetrics, lambda 0.94) estimated from returns up
to and including `t`, so the band is known at prediction time. `kappa` is **solved inside
each training window** so the neutral class takes a target share of the training labels,
then applied unchanged to the test window.

Fitting on AAPL 2006-2015 and applying to 2016-2026 gives `kappa = 0.366` at `h=1`, below
the 0.431 that a normal distribution implies — the expected direction for fat-tailed
returns. The volatility-scaled band also holds its neutral share out of sample
(0.333 to 0.356) far better than an unconditional-tertile ablation (0.333 to 0.420).

---

## Data

`yfinance` daily bars from 2005 to the latest close, cached locally as parquet and
refreshed incrementally. Nothing is hard-coded: the provider sits behind a protocol.

```python
class MarketDataProvider(Protocol):
    def get_ohlcv(self, ticker: str, start: date, end: date | None) -> pd.DataFrame: ...
    def get_metadata(self, ticker: str) -> AssetMetadata: ...
```

Four implementations ship: `yfinance` (default), `csv` (offline fixtures and tests),
`alphavantage` (optional, key read only from `ALPHAVANTAGE_API_KEY`), and `stooq`. Stooq
now fronts its CSV endpoint with a JavaScript proof-of-work challenge; the provider
detects that and raises rather than attempting to defeat it.

**Corporate actions.** Yahoo delivers OHLC adjusted for splits but not dividends, so the
whole bar is scaled by `adj_close / close`. Range estimators such as ATR and Parkinson
volatility need high, low and close on one consistent scale; adjusting only the close
silently corrupts them. Volume is left alone because it is already split adjusted.

**Validation** runs on every download: schema and dtypes, chronological unique index,
non-positive prices, inverted bars, sessions missing against the NYSE calendar, rows dated
on exchange holidays, implausible single-day moves, zero-volume runs, and staleness against
the last closed session. Index symbols are exempted from the volume and extreme-move rules.

The checks earn their place. On the shipped universe they flagged a `^VIX` row dated
2026-05-25, a Memorial Day holiday when the exchange was closed. They also flagged MS
+87.0% on 2008-10-13 and VIX +115.6% on 2018-02-05, both of which are genuine.

---

## Features

58 features in `v1`, each declared in a registry that carries its definition, formula,
economic rationale, lookback and leakage notes. `docs/FEATURES.md` is generated from that
registry, so the documentation cannot drift from the code.

| Group | Count | Examples |
|---|---|---|
| returns | 10 | multi-horizon log returns, five daily lags |
| momentum | 11 | price relative to four moving averages, RSI, MACD/price, 12-1 skip-month momentum, 52-week position |
| volatility | 9 | realised volatility at 5/20/60, volatility term structure, ATR/price, Parkinson, downside share, standardised move |
| volume | 4 | relative volume, volume z-score, traded value ratio, return-volume correlation |
| trend | 4 | Kaufman efficiency ratio, distance from 52-week high and low, up-day ratio |
| market | 13 | benchmark returns and trend, rolling beta and correlation, residual return, VIX level/z-score/change, sector return and relative strength |
| calendar | 7 | weekday dummies, cyclic month encoding, days to month end |

Two rules are enforced mechanically rather than by convention:

**Every feature is scale free.** Not `sma_50` but `close / sma_50 - 1`; not `volume` but
`log(volume / average_volume)`; not `atr` but `atr / close`. A pooled model trained across
tickers whose prices span two orders of magnitude and across twenty years of price levels
cannot use raw levels. The audit multiplies the whole price series by a constant and
requires every feature to be unchanged.

**Nothing reads forward.** No `bfill` anywhere; market context is aligned forward only,
with a bounded carry and a staleness counter.

---

## Leakage auditing

Reading `shift` calls is not evidence. Three independent audits run over the real pipeline,
and each one is also required to **fail** on a deliberately planted bug — a test that can
only ever pass proves nothing.

| Audit | Method | Planted bug it catches |
|---|---|---|
| perturbation | Overwrite every row after `t` with a different price path, rebuild, require row `t` unchanged | next-day return as a feature; centre-weighted rolling window; leak through the market context; a backward fill |
| truncation | Rebuild from history up to `t` only, require row `t` to match the full-series value | normalisation fitted on the whole sample |
| scale | Rescale prices and volumes, require every feature unchanged | a raw moving-average level; raw volume |
| target causality | Drive the horizon window up and down; require the label to respond, and to ignore anything past `t+h` | a label reaching to `t+15` while declared as `h=5`; a label that never looks forward |

All 58 features pass all three audits on every ticker tested.

```
$ make audit
AAPL   pass   25 checks  58 features
MSFT   pass   25 checks  58 features
...
6 tickers audited, 0 failing reports
```

---

## Universe

42 tickers across technology, financials, healthcare, consumer, energy, industrials and
utilities, split into three disjoint groups fixed before any modelling:

- **dev** (20) — all training and iteration
- **val** (11) — sanity checks between phases
- **test** (11) — scored once, at the end

Groups are stratified by sector rather than sampled at random, so "unseen asset" is not
confounded with "unseen sector". Crossed with the temporal split this gives a 2x2, and the
unseen-asset by unseen-time cell is the honest generalisation number.

Splits are always taken on **dates**, never on row position: same-day returns are
cross-sectionally correlated, so putting one date on both sides of a split would leak one
ticker's outcome into another ticker's training set.

---

## Running it

Requires Python 3.11+ and, on macOS, the OpenMP runtime that xgboost and lightgbm link
against.

```bash
brew install libomp          # macOS only
make setup                   # creates .venv and installs the package
make data                    # downloads the universe into data/cache (a few minutes)
make panel                   # builds the pooled feature and target panel
make audit                   # runs the leakage, truncation and scale audits
make check                   # ruff, mypy and the test suite
```

The test suite runs entirely offline against synthetic fixtures.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data pipeline, features, targets, leakage auditing | complete |
| 2 | Baselines, models, walk-forward evaluation, calibration | not started |
| 3 | Regime detection, SHAP, error analysis, backtesting | not started |
| 4 | FastAPI backend | not started |
| 5 | Dashboard | not started |
| 6 | Documentation and final report | not started |

---

## Known limitations

Stated now rather than discovered later.

- **The expected effect size is very small.** Daily equity direction from technical
  features is close to unpredictable. A defensible out-of-sample result is ROC-AUC in the
  0.50 to 0.54 range. Anything much above that will be treated as a suspected leak.
- **Overlapping labels.** At `h=5` consecutive labels share four of five days, so the
  effective sample is roughly `N/5` and naive confidence intervals are badly overstated.
  Phase 2 handles this with purging, embargoing and block bootstrap.
- **Survivorship bias.** The universe contains only companies still listed today.
  Delisted names are absent, which biases any historical result optimistic.
- **Adjusted prices are restated.** The adjusted close for 2015 that Yahoo serves today is
  not what an observer saw in 2015; it embeds every subsequent split and dividend. This is
  a genuine look-ahead that free data cannot avoid.
- **Sector membership is current, not historical.** A mild look-ahead for firms since
  reclassified.
- **Technical features only.** No fundamentals, no news, no order-flow data, so the
  information ceiling is low by construction.
- **Prediction accuracy is not profitability.** Costs, slippage and market impact are
  addressed explicitly in Phase 3 and are not an afterthought.
