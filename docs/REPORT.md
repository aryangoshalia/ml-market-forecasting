# Methodology report

What was built, why each decision was made that way, and what went wrong along the road.

Numbers here describe the runs recorded on 2026-09-07 against a panel ending 2026-09-04.
Claims are phrased so they stay true if a rerun moves the third decimal.

---

## 1. Problem formulation

Let `C_t` be the split and dividend adjusted close of session `t`, and `F_t` everything
observable at or before that close. For a horizon `h` of one or five sessions the model
estimates

```
p_t(h) = P( y_t(h) = 1 | F_t )
```

This is probabilistic classification, not point forecasting. Predicting the *level* of a
series that is close to a random walk produces a chart that looks impressive and carries no
information, because the model learns that tomorrow's price is roughly today's. Direction
can at least be scored honestly.

### Targets

| Name | Definition |
|---|---|
| `direction` | `1[ C_(t+h)/C_t - 1 > 0 ]` |
| `excess_direction` | `1[ r_t(h) - r_benchmark,t(h) > 0 ]` |
| `direction_open` | return measured from the next open rather than the signal close |
| `ternary` | `sign(r)` outside a volatility scaled band, `0` inside |

The second target matters more than it appears. Equities drift upward, so the raw direction
target is partly a free win.

| Target | Base rate |
|---|---|
| `direction_1d` | 0.5180 |
| `excess_direction_1d` | 0.4953 |
| `direction_5d` | 0.5493 |
| `excess_direction_5d` | 0.5040 |

A model that always predicts up scores 54.9 percent on the raw five-day target. Reporting
that as accuracy would be meaningless, so both targets run through every experiment.

### The neutral band

A fixed threshold such as plus or minus half a percent is not defensible. Half a percent is
a large move in 2017 and noise in March 2020. The band is scaled instead:

```
tau_t = kappa * sigma_t * sqrt(h)
```

`sigma_t` is an exponentially weighted volatility estimated from returns up to and including
`t`, so the band is known at prediction time. `kappa` is solved inside each training window
so the neutral class takes a target share of training labels, then applied unchanged.

Fitting on 2006 to 2015 and applying to 2016 onward gives `kappa = 0.366` at one day. A
normal distribution implies 0.431. The gap is what fat tails predict. The volatility scaled
band also holds its neutral share out of sample, drifting from 0.333 to 0.356, while an
unconditional tertile ablation drifts to 0.420.

### Execution convention

Close to close returns assume you can trade at the close you just predicted from. That
assumption is stated rather than hidden, and `direction_open` measures the return from the
next open as a robustness check.

---

## 2. Data

Daily bars from Yahoo Finance through `yfinance`, 2005 to the latest close, cached as
parquet with incremental refresh. Everything downstream depends on a protocol rather than a
vendor:

```python
class MarketDataProvider(Protocol):
    def get_ohlcv(self, ticker: str, start: date, end: date | None) -> pd.DataFrame: ...
    def get_metadata(self, ticker: str) -> AssetMetadata: ...
```

Four implementations exist. Yahoo is the default. A CSV provider backs the offline test
suite. Alpha Vantage is optional and reads its key from an environment variable only. Stooq
was written as a second live vendor for cross-source comparison, but it now fronts its CSV
endpoint with a JavaScript proof-of-work challenge, so the provider detects that and raises
rather than attempting to defeat it.

### Corporate actions

Yahoo delivers OHLC adjusted for splits but not dividends. Scaling only the close would
leave high, low and close on different bases and quietly corrupt every range estimator, so
the whole bar is scaled by `adj_close / close`. Volume is left alone, since it is already
split adjusted and dividends do not change share counts.

Verified against NVDA's ten for one split in June 2024: no discontinuity survives, and the
largest remaining single-day moves are real events, including minus 30.7 percent on
2008-07-03 and plus 24.4 percent on 2023-05-25.

### Validation

Every download is checked for schema and dtype, a sorted index without duplicates,
non-positive prices, inverted bars, sessions missing against the exchange calendar, rows
dated on days the exchange was closed, implausible single-day moves, zero-volume runs, and
staleness against the last closed session. Index symbols are exempted from the volume and
extreme-move rules, because an index has no share volume and the VIX legitimately moves 100
percent in a day.

The checks earn their place. They flagged a VIX row dated 2026-05-25, which was Memorial
Day, and later a live VIX quote published on Labor Day 2026 while the exchange was closed.
They also flagged Morgan Stanley rising 87 percent on 2008-10-13 and the VIX rising 115.6
percent on 2018-02-05, both genuine.

---

## 3. Features

58 features in version `v1`, each declared in a registry carrying its formula, rationale,
lookback and leakage notes. `docs/FEATURES.md` is generated from that registry, so the
documentation cannot drift from the code.

| Group | Count |
|---|---|
| returns | 10 |
| momentum | 11 |
| volatility | 9 |
| volume | 4 |
| trend | 4 |
| market context | 13 |
| calendar | 7 |

Two rules are enforced by tests rather than by convention.

**Everything is scale free.** Not `sma_50` but `close / sma_50 - 1`. Not `volume` but
`log(volume / average_volume)`. Not `atr` but `atr / close`. A model pooled across tickers
whose prices differ by two orders of magnitude cannot use raw levels. The audit multiplies
the entire price series by a constant and requires every feature to be unchanged.

**Nothing reads forward.** No backward fill anywhere. Market context is aligned forward
only, with a bounded carry and a staleness counter.

Features come from the literature rather than from combination. Skip-month momentum omits
the most recent month because short-term reversal works against the momentum effect.
Parkinson volatility uses the high-low range, a more efficient estimator than close-to-close
at the same window. Kaufman's efficiency ratio separates a clean trend from a choppy market
that travelled the same net distance.

---

## 4. Leakage auditing

Reading `shift` calls is not evidence. Three audits run over the real pipeline, and each is
also required to fail on a deliberately planted bug.

| Audit | Method |
|---|---|
| perturbation | overwrite every row after `t` with a different price path, rebuild, require row `t` unchanged |
| truncation | rebuild from history up to `t` only, require row `t` to match the full-series value |
| scale | rescale prices and volumes, require every feature unchanged |

Targets are audited in the opposite direction. A label at `t` must respond when the window
it is defined over is driven up versus down, and must not change when anything past `t+h` is
altered.

Nine planted faults, each a mistake an ordinary pipeline could contain:

| Planted fault | Caught by |
|---|---|
| next-day return used as a feature | perturbation |
| centre-weighted rolling window | perturbation |
| leak through the market context series | perturbation |
| backward fill | perturbation |
| normalisation fitted on the whole sample | truncation |
| raw moving-average level | scale |
| raw share volume | scale |
| label reaching to `t+15` while declared as `h=5` | target causality |
| label that never looks forward | target causality |

All 58 features pass all three audits on every ticker tested.

---

## 5. Validation design

Anchored walk-forward, 29 folds, test coverage from 2012-01-05 to 2026-07-20. Fold geometry
in session positions:

```
[ fit ........ ] [gap] [ inner ] [gap] [ test ]
```

The gap is `h` purge sessions plus a five-session embargo. The purge is not optional. The
label on the last fitting row is defined by prices `h` sessions later, which fall inside the
next window, so without it every fold trains on part of its own evaluation period. This is
visible in the geometry: at `h=5` the fitting window ends four sessions earlier than at
`h=1`.

Everything learned lives inside the training window. Scaling, imputation, calibration, the
decision threshold, the ternary `kappa`, hyperparameters and the regime model are all fitted
on training data and applied unchanged.

Splits are taken on **dates**, not row positions. Returns are correlated across tickers on
the same session, so splitting by row would put one ticker's outcome into another ticker's
training set.

### Statistical testing

Two dependencies make naive testing wrong. Labels at horizon five overlap, so consecutive
observations are not independent. Returns are cross-sectionally correlated, so rows within a
session are not independent either.

The fold is therefore the unit of observation. Model comparisons use a paired Wilcoxon
signed-rank test across folds with Benjamini-Hochberg correction. Confidence intervals come
from a block bootstrap resampling contiguous runs of folds. The pooled permutation null
permutes whole sessions inside each fold, preserving both dependencies.

Measured effective breadth is worth recording. Among the twenty development tickers, mean
pairwise daily return correlation is 0.40 and the first principal component explains 43.5
percent of cross-sectional variance. Under an equicorrelation model that caps effective
independent names per day at 2.5, so moving from twenty tickers to five hundred would buy
about four percent on the standard error. Ticker count is not the binding constraint for
absolute direction. The market factor is.

---

## 6. Results

| Target | Horizon | Best learner | ROC-AUC | Persistence | Base rate |
|---|---|---|---|---|---|
| `excess_direction` | 1 | random forest | 0.5080 | 0.5018 | 0.4962 |
| `excess_direction` | 5 | random forest | 0.5056 | 0.5016 | 0.5070 |
| `direction` | 1 | random forest | 0.5119 | 0.5129 | 0.5197 |
| `direction` | 5 | xgboost | 0.5084 | 0.5024 | 0.5539 |

### The negatives

On absolute direction at one day the only statistically significant result belongs to the
persistence baseline, at p = 0.0084 after correction. Five machine learning models with 58
features do not beat "assume the last move repeats". At five days nothing is significant
anywhere.

In all 32 model-and-target cells, accuracy sits below the best constant predictor.

The learners are also worse calibrated than a constant, with Brier scores of 0.252 to 0.256
against 0.250, even after isotonic calibration.

### The one positive result

Predicting whether a stock beats the benchmark over one session is the single formulation
where the learners are significantly better than the baselines.

| Model | AUC | Folds won | p adjusted |
|---|---|---|---|
| lightgbm | 0.5079 | 25/29 | 0.0001 |
| random forest | 0.5080 | 22/29 | 0.0015 |
| xgboost | 0.5067 | 22/29 | 0.0026 |
| elastic net | 0.5068 | 21/29 | 0.0026 |
| logistic | 0.5052 | 19/29 | 0.0221 |
| persistence | 0.5018 | 9/29 | 0.49 |

It survives Bonferroni correction across all 24 tests, a permutation null preserving fold
base rates and same-day structure, a block bootstrap interval excluding 0.5, and replication
across four independent training configurations. The persistence baseline never sees it in
any run, so it is not first-order autocorrelation.

The effect remains about 0.8 points of AUC. Accuracy is still below the base rate.

### Unseen assets

Twenty two tickers were held out before any modelling began and scored exactly once, after
every decision was frozen. Training used the twenty development names; scoring used stocks
the model had never seen, over the same test windows.

| Model | Seen | Unseen | Change | Folds won | p adjusted |
|---|---|---|---|---|---|
| elastic net | 0.5068 | 0.5067 | -0.0002 | 20/29 | 0.0475 |
| random forest | 0.5080 | 0.5054 | -0.0025 | 20/29 | 0.0475 |
| xgboost | 0.5067 | 0.5053 | -0.0015 | 20/29 | 0.0920 |
| logistic | 0.5052 | 0.5046 | -0.0006 | 19/29 | 0.0920 |
| lightgbm | 0.5079 | 0.5039 | -0.0040 | 18/29 | 0.0920 |

Base rates are comparable, 0.4962 on the development group against 0.4946 on the held-out
group, so this is a fair comparison rather than an easier problem.

The edge survives. Two models stay significant after correction and the mean degradation is
about 0.002 of AUC. Whatever the model found is a property of the cross-section rather than
of the twenty tickers it was fitted on.

At five sessions nothing is significant on unseen assets, which matches the development
result. Absolute direction at one day shows random forest at 0.5099 with p = 0.0375, its
only significant appearance anywhere.

This evaluation is the reason the asset split was fixed before any modelling and touched
once. Running it earlier, or rerunning it after a change, would have destroyed the only
clean estimate of generalisation in the project.

---

## 7. Why hyperparameter tuning failed

About eighty configurations were searched across five models, on validation windows lying
entirely before the first test fold so no scored session influenced the choice.

Two of the five models could not beat chance with any setting. Random forest's best of
nineteen configurations scored 0.4962 on the tuning windows. LightGBM's best of twenty
scored 0.4940.

More telling: for three of five models the **fold-to-fold standard deviation exceeded the
entire spread between the best and worst configuration.** LightGBM's configurations spanned
0.0142 while its fold noise was plus or minus 0.0306. When that holds, the argmax is not the
best setting. It is the luckiest one.

Applied to the test folds, tuning improved 7 of 20 cells and made 13 worse, for a mean
effect of minus 0.0004. Random forest was worse in all four formulations, the clearest sign
of selection on noise. Every individual delta is smaller than the fold-level standard error,
so the honest claim is that tuning produced no detectable improvement rather than that it
actively hurt.

---

## 8. Robustness

**Start date.** Comparing a 2005 start against a 2010 start on the 52,920 test observations
they share, more history helped in 16 of 24 cells, largest effect 0.0057. Linear models
benefit most consistently, which is the bias-variance story behaving as expected. Every
effect sits inside the noise and no conclusion changes.

**Window scheme.** Rolling windows of fixed width against expanding windows show no
meaningful difference. Only `direction` at five days favours rolling, where all six models
improve, but a paired test returns p = 0.44.

---

## 9. Regime detection

Fitted on market-level features rather than per stock, so one regime series conditions every
ticker and stays comparable across them.

### Choosing the method

Selection was not by likelihood alone. A mixture will always fit better in sample while
producing states that flicker daily.

| Method | Median state duration | Switches per year | Usable |
|---|---|---|---|
| k-means | 3 to 4 sessions | 16 to 31 | no |
| Gaussian mixture | 2 to 6 sessions | 4 to 17 | no |
| HMM, 4 or 5 states | degenerate states | | no |
| HMM, 3 states | 21.5 sessions | 6.0 | yes |

Three states won on BIC and held-out likelihood among candidates with plausible durations.
k-means defines no likelihood, so its BIC column is left empty rather than filled with a
number that invites a false comparison.

### Filtered against smoothed

Smoothed state probabilities condition on the whole sample, including sessions after `t`.
Using them to describe the regime at `t` is look-ahead, and it is a common mistake. Only the
filtered posterior is used here.

The difference was measured rather than assumed. Filtered probabilities are bit-identical
whether computed on a prefix or the full series. Smoothed probabilities change when later
data arrives, and the two disagree on 4.2 percent of sessions.

### The fitted states

Labels are generated from each state's own statistics after fitting. No cluster was called
bullish or bearish in advance.

| State | Volatility | VIX z | Breadth | Annualised return |
|---|---|---|---|---|
| low volatility, positive trend | 9.3% | -0.62 | 76% | +24.5% |
| high volatility, negative trend | 15.8% | +0.79 | 51% | -6.2% |
| high volatility, sideways | 27.5% | -0.26 | 54% | +10.3% |

### Conditional performance

The benchmark-relative edge at one day is two to three times stronger once dispersion rises.
All five learners independently show lift over the majority baseline of 0.001 to 0.004 in
the calm uptrend regime, against 0.006 to 0.007 in the mid-volatility regime. Persistence
has negative lift in the calm regime.

That is economically sensible. When everything rallies together, relative stock selection is
hard.

---

## 10. Explainability

SHAP on the tree models, reported in log-odds so the base value plus the contributions
reconstructs the prediction exactly.

### The mechanism

Rolling beta ranks 10th, 4th, 8th, 4th and 1st across five folds for the benchmark-relative
target. For absolute direction the same feature ranks 32nd, 31st, 57th, 54th and 41st out of
58.

That is what the arithmetic of relative returns implies. Expected excess return is
`(beta - 1)` times expected market return, so beta combined with market state should predict
relative direction while contributing little to absolute direction. Market-context features
take 54.5 percent of attribution for the raw target against 26.3 percent for the excess
target, the same story from the other side.

### Importance is not well identified

Three independent measures barely agree. Spearman correlation across all 58 features is
+0.004 between SHAP and permutation importance, +0.149 between SHAP and logistic
coefficients, and -0.169 between permutation importance and coefficients. None is
significant.

They agree on the top handful, five of ten overlapping between SHAP and permutation. Below
that the ranking is noise. At this signal level a SHAP bar chart should not be presented as
truth.

---

## 11. Error analysis

**Confidence carries a little information.** Accuracy rises from 0.4974 in the lowest
confidence decile to 0.5275 in the highest, though the profile is not monotone.

**Performance varies more across tickers than across anything else.** AUC ranges from 0.4956
on Johnson and Johnson to 0.5358 on Alphabet. Large-cap technology names do best. A single
aggregate figure hides that spread.

**Conditional patterns line up with the regime finding.** AUC rises with the stock's own
volatility, from 0.5062 to 0.5179 across quintiles, and with beta, from 0.5086 to 0.5146.

**Trend reversals are not a weak point.** Accuracy near a change in trend sign differs from
elsewhere by 0.0004, which is nothing. A momentum-driven model might have been expected to
struggle at turning points. It does not.

---

## 12. Friction

The question is narrow. Does a directional edge of this size survive realistic costs?

Trading the benchmark-relative signal market neutral:

| Round-trip cost | Annual return | Sharpe | Max drawdown |
|---|---|---|---|
| 0 bps | +11.1% | 1.06 | -13.3% |
| 5 bps | +2.2% | 0.20 | -25.7% |
| 10 bps | -6.8% | -0.65 | -66.5% |
| 20 bps | -24.8% | -2.33 | -97.5% |

**Break-even is 6.2 basis points.** Turnover is 0.71 per day, and that churn is what kills
it. The figure is optimistic, because the short benchmark leg is not charged.

The long-only variant returns 26.2 percent gross at a Sharpe of 1.39, which looks strong
until compared against the right benchmark. Equal-weighted buy and hold over the same
sessions returns 20.6 percent at a Sharpe of 1.23 with no turnover at all. At five basis
points the strategy returns 17.3 percent and loses to doing nothing.

Everything in this section flatters the result. Fills are assumed at the close, market impact
and capacity are ignored, borrow on the short leg is free, and the universe contains only
firms that still exist.

---

## 13. Six defects the instrumentation found

Each was found by the project's own checks rather than by inspection, and each has a test
that would catch it again.

**1. A crash on unsorted input.** The session-gap checker assumed a sorted index and threw on
a reversed one. Found by a validation test.

**2. Look-ahead in the hyperparameter search.** The first version searched over the earliest
few walk-forward folds. Fold one's inner window sits after fold zero's test window, so
settings chosen there would have been applied backwards in time. The search now builds its
own nested validation windows confined entirely to the pre-test period, and a test asserts
the cutoff precedes every test window.

**3. Isotonic calibration returning degenerate probabilities.** Between 0.47 and 1.19 percent
of calibrated predictions were exactly 0 or 1, while none of the raw probabilities were. A
pure isotonic bin maps to the endpoints, which claims certainty no finite sample supports and
makes log loss unbounded. Fixed with a Laplace bound of `1/(n+2)`. Ranking metrics are
unaffected because the clip is monotone, which a test asserts.

**4. A permutation null fooled by base-rate drift.** The majority baseline was scoring a
pooled AUC of 0.507 when a constant predictor must score exactly 0.5. It predicts each fold's
training base rate, and those correlate +0.458 with test outcomes, so pooling folds turned
drift into apparent discrimination. The corrected null permutes whole sessions inside each
fold. Two negative-control tests assert the naive null is fooled and the corrected one is
not. This changed a conclusion: `excess_direction` at five days went from p = 0.007 to
p = 0.137 and is no longer significant.

**5. F1 threshold selection collapsing.** Between 56 and 62 percent of folds chose the lowest
threshold on the grid, producing classifiers that predicted positive 98 to 99 percent of the
time. F1 rewards recall, so against a near-random ranker its maximum is "always predict
positive". The default is now balanced accuracy, which chooses 0.50 and predicts positive 55
percent of the time. A warning fires whenever a chosen threshold lands on the grid edge.

**6. A phantom bar on a market holiday.** Yahoo published a live VIX quote on Labor Day 2026
while the exchange was closed. The validator warned but the loader did not act, so the row
would have shifted the reported latest session and every trailing window. Rows dated outside
exchange sessions are now dropped.

---

## 14. Limitations

- **Survivorship.** The universe contains only companies still listed today. Every historical
  result is biased optimistic and free data offers no fix.
- **Restated prices.** Adjusted closes are rewritten whenever a dividend or split occurs. The
  2015 history available today is not what an observer saw in 2015. A genuine look-ahead that
  cannot be removed.
- **Overlapping labels.** At five sessions the effective sample is roughly a fifth of the row
  count.
- **Multiple testing.** Eight models across four formulations and several configurations is
  many comparisons. Nominal and adjusted p-values are both reported.
- **Point-in-time metadata.** Sector membership comes from current classification data.
- **Information ceiling.** Technical features only. No fundamentals, no news, no order flow.
- **Accuracy is not profitability.** The friction section answers that, and not favourably.

---

## 15. Future work

**Cross-sectional ranking.** Asking which five of twenty stocks will outperform the other
fifteen, rather than whether one stock rises, is how quantitative equity is actually done.
The market factor cancels and effective breadth then grows with the number of names instead
of capping at 2.5. It was deliberately not added here: the market-relative target already
captures most of that insight, and adding formulations until one scores well is exactly the
behaviour this project is built to avoid.

**Longer horizons.** Twenty-day direction is more predictable than one-day, at the cost of
fewer independent observations.

**Regime-conditional models.** Fitting a separate model per regime, given that the edge
concentrates in higher-dispersion regimes.

**Data beyond free OHLCV.** Fundamentals, news sentiment and options-implied volatility are
the real ceiling, and they cost money.
