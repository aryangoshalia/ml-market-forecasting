# Feature reference (`v1`)

58 features across 7 groups. Longest lookback is 253 sessions, which sets the warm-up period dropped from the start of every ticker.

Generated from the feature registry by `scripts/feature_docs.py`; do not edit by hand.

## returns (10)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `ret_1d` | 2 | 1-session log return<br>`log(C_t / C_(t-1))` | Short horizons carry weak reversal, longer horizons carry weak continuation; together they describe the recent return path |
| `ret_3d` | 4 | 3-session log return<br>`log(C_t / C_(t-3))` | Short horizons carry weak reversal, longer horizons carry weak continuation; together they describe the recent return path |
| `ret_5d` | 6 | 5-session log return<br>`log(C_t / C_(t-5))` | Short horizons carry weak reversal, longer horizons carry weak continuation; together they describe the recent return path |
| `ret_10d` | 11 | 10-session log return<br>`log(C_t / C_(t-10))` | Short horizons carry weak reversal, longer horizons carry weak continuation; together they describe the recent return path |
| `ret_20d` | 21 | 20-session log return<br>`log(C_t / C_(t-20))` | Short horizons carry weak reversal, longer horizons carry weak continuation; together they describe the recent return path |
| `ret_1d_lag1` | 3 | Daily log return lagged 1 sessions<br>`log(C_(t-1) / C_(t-1-1))` | Lets the model learn sign patterns in the recent return sequence rather than only its aggregate |
| `ret_1d_lag2` | 4 | Daily log return lagged 2 sessions<br>`log(C_(t-2) / C_(t-2-1))` | Lets the model learn sign patterns in the recent return sequence rather than only its aggregate |
| `ret_1d_lag3` | 5 | Daily log return lagged 3 sessions<br>`log(C_(t-3) / C_(t-3-1))` | Lets the model learn sign patterns in the recent return sequence rather than only its aggregate |
| `ret_1d_lag4` | 6 | Daily log return lagged 4 sessions<br>`log(C_(t-4) / C_(t-4-1))` | Lets the model learn sign patterns in the recent return sequence rather than only its aggregate |
| `ret_1d_lag5` | 7 | Daily log return lagged 5 sessions<br>`log(C_(t-5) / C_(t-5-1))` | Lets the model learn sign patterns in the recent return sequence rather than only its aggregate |

## momentum (11)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `px_to_sma_10` | 10 | Price relative to its 10-session simple moving average<br>`C_t / SMA_10(C)_t - 1` | Distance from a trailing average is the scale-free form of trend position; the raw average level is not comparable across tickers |
| `px_to_sma_20` | 20 | Price relative to its 20-session simple moving average<br>`C_t / SMA_20(C)_t - 1` | Distance from a trailing average is the scale-free form of trend position; the raw average level is not comparable across tickers |
| `px_to_sma_50` | 50 | Price relative to its 50-session simple moving average<br>`C_t / SMA_50(C)_t - 1` | Distance from a trailing average is the scale-free form of trend position; the raw average level is not comparable across tickers |
| `px_to_sma_200` | 200 | Price relative to its 200-session simple moving average<br>`C_t / SMA_200(C)_t - 1` | Distance from a trailing average is the scale-free form of trend position; the raw average level is not comparable across tickers |
| `sma_ratio_20_50` | 50 | 20-session average relative to the 50-session average<br>`SMA_20(C)_t / SMA_50(C)_t - 1` | Continuous form of the moving-average crossover: sign gives the trend direction, magnitude gives its separation |
| `sma_ratio_50_200` | 200 | 50-session average relative to the 200-session average<br>`SMA_50(C)_t / SMA_200(C)_t - 1` | Continuous form of the moving-average crossover: sign gives the trend direction, magnitude gives its separation |
| `rsi_14` | 42 | Wilder relative strength index over 14 sessions, scaled to [0, 1]<br>`100 - 100 / (1 + avg_gain / avg_loss), divided by 100` | Bounded measure of how one-sided recent moves have been; extremes are associated with short-term mean reversion |
| `macd_norm` | 78 | MACD line divided by price<br>`(EMA_12(C) - EMA_26(C)) / C_t` | Difference of two exponential averages responds faster than a simple crossover; dividing by price makes it comparable across tickers |
| `macd_hist_norm` | 87 | MACD histogram divided by price<br>`(MACD_t - signal_t) / C_t` | Rate of change of the MACD line, an early indication of a turning trend |
| `mom_12_1` | 253 | Return from 252 to 21 sessions ago, skipping the most recent month<br>`log(C_(t-21) / C_(t-252))` | The classic cross-sectional momentum construction; the recent month is skipped because it carries short-term reversal that offsets the momentum effect |
| `price_position_252` | 252 | Where price sits within its 252-session range<br>`(C_t - min_252(C)) / (max_252(C) - min_252(C))` | Proximity to a 52-week extreme is a documented anchor for investor behaviour and is naturally bounded |

## volatility (9)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `vol_5` | 6 | Annualised realised volatility over 5 sessions<br>`std_5(log returns) * sqrt(252)` | Volatility is strongly autocorrelated and conditions the size of any future move, so it calibrates how much weight a directional signal deserves |
| `vol_20` | 21 | Annualised realised volatility over 20 sessions<br>`std_20(log returns) * sqrt(252)` | Volatility is strongly autocorrelated and conditions the size of any future move, so it calibrates how much weight a directional signal deserves |
| `vol_60` | 61 | Annualised realised volatility over 60 sessions<br>`std_60(log returns) * sqrt(252)` | Volatility is strongly autocorrelated and conditions the size of any future move, so it calibrates how much weight a directional signal deserves |
| `vol_ratio_5_20` | 21 | 5-session volatility relative to 20-session volatility<br>`vol_5 / vol_20` | The volatility term structure: values above one mean volatility is expanding, which historically coincides with weaker forward returns |
| `vol_ratio_20_60` | 61 | 20-session volatility relative to 60-session volatility<br>`vol_20 / vol_60` | The volatility term structure: values above one mean volatility is expanding, which historically coincides with weaker forward returns |
| `atr_14_norm` | 42 | Average true range over 14 sessions, divided by price<br>`Wilder ATR / C_t` | Uses the full bar including overnight gaps, so it captures risk that a close-to-close estimate misses |
| `parkinson_vol_20` | 20 | Parkinson high-low volatility estimator over 20 sessions<br>`sqrt(mean_20(log(H/L)^2) / (4 log 2)) * sqrt(252)` | Roughly five times more efficient than the close-to-close estimator at the same window length, so it reacts to changes in risk sooner |
| `downside_vol_ratio_20` | 21 | Share of 20-session volatility contributed by down moves<br>`downside_vol_20 / vol_20` | Separates volatility driven by selling from volatility driven by rallies, which a symmetric estimator cannot distinguish |
| `standardised_move` | 21 | Today's return in units of its own 20-session volatility<br>`r_t / std_20(r)` | A one percent move means something different in calm and stressed markets; this expresses the latest move on a comparable scale |

## volume (4)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `rel_volume_20` | 20 | Log of volume against its 20-session average<br>`log(V_t / SMA_20(V)_t)` | Participation relative to the ticker's own norm; share counts differ by orders of magnitude across tickers so the raw level is unusable |
| `volume_z_20` | 20 | Volume z-score over 20 sessions<br>`(V_t - mean_20(V)) / std_20(V)` | Scales the surprise in volume by how variable volume normally is, which a plain ratio ignores |
| `dollar_volume_ratio_20` | 20 | Log traded value against its 20-session average<br>`log(C_t V_t / SMA_20(C V)_t)` | Traded value tracks the capital committed to a move better than share count, which drifts with the price level |
| `price_volume_corr_20` | 21 | Correlation of returns and volume changes over 20 sessions<br>`corr_20(r_t, log(V_t / V_(t-1)))` | Distinguishes advances on rising volume from advances on fading volume, a conviction signal neither series carries alone |

## trend (4)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `efficiency_ratio_20` | 21 | Net move divided by total path length over 20 sessions<br>`|C_t - C_(t-20)| / sum(|C_i - C_(i-1)|)` | Bounded in [0, 1] and separates a clean trend from a choppy market that travelled the same net distance; conditions whether momentum should be trusted |
| `dist_high_252` | 252 | Drawdown from the 252-session high<br>`C_t / max_252(C)_t - 1` | Never positive; how deep the current drawdown is separates a pullback from a broken trend |
| `dist_low_252` | 252 | Advance above the 252-session low<br>`C_t / min_252(C)_t - 1` | Never negative; measures how far a recovery has already run |
| `up_day_ratio_20` | 21 | Fraction of the last 20 sessions that closed higher<br>`mean_20(1[r_t > 0])` | Counts direction without weighting by size, so a single large move cannot dominate the way it does in a mean return |

## market (13)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `bench_ret_1d` | 2 | 1-session benchmark (SPY) log return<br>`log(B_t / B_(t-1))` | Most of a single stock's daily variance is the market factor, so the market's own recent path is directly relevant |
| `bench_ret_5d` | 6 | 5-session benchmark (SPY) log return<br>`log(B_t / B_(t-5))` | Most of a single stock's daily variance is the market factor, so the market's own recent path is directly relevant |
| `bench_ret_20d` | 21 | 20-session benchmark (SPY) log return<br>`log(B_t / B_(t-20))` | Most of a single stock's daily variance is the market factor, so the market's own recent path is directly relevant |
| `bench_vol_20` | 21 | Annualised benchmark volatility over 20 sessions<br>`std_20(benchmark log returns) * sqrt(252)` | Systematic risk level, which conditions how much idiosyncratic signal survives |
| `bench_px_to_sma_50` | 50 | Benchmark relative to its 50-session moving average<br>`B_t / SMA_50(B)_t - 1` | Broad-market trend state, a common conditioning variable for stock-level signals |
| `beta_60` | 61 | Rolling market beta over 60 sessions<br>`cov_60(r, r_m) / var_60(r_m)` | How much of the stock's movement is market driven; high-beta names respond differently to the same market signal than defensive ones |
| `corr_bench_60` | 61 | Correlation with the benchmark over 60 sessions<br>`corr_60(r, r_m)` | Unlike beta this is scale free, and it separates a stock trading on its own news from one moving with the index |
| `resid_ret_5d` | 61 | 5-session return net of the beta-scaled market return<br>`r_t(5) - beta_t * r_m,t(5)` | Isolates stock-specific movement; residual momentum is a cleaner signal than total momentum because it removes the shared market factor |
| `vix_level` | 1 | Volatility index level expressed as a decimal<br>`VIX_t / 100` | The market's priced expectation of future volatility, which is forward looking in a way realised volatility is not |
| `vix_z_60` | 60 | Volatility index z-score over 60 sessions<br>`(VIX_t - mean_60) / std_60` | The level alone drifts across decades; the z-score asks whether fear is elevated relative to the recent regime |
| `vix_chg_5d` | 6 | 5-session log change in the volatility index<br>`log(VIX_t / VIX_(t-5))` | Direction of repricing in risk, which often leads price weakness |
| `sector_ret_5d` | 6 | 5-session return of the sector ETF proxy<br>`log(S_t / S_(t-5))` | Sector rotation explains a large share of the return left after the market factor |
| `rel_sector_ret_5d` | 6 | 5-session return relative to the sector ETF<br>`r_t(5) - r_sector,t(5)` | Relative strength within a sector separates a stock leading its peers from one simply carried by them |

Leakage notes:

- Beta is estimated on the trailing window ending at t, so no future returns enter the residual.
- Sector membership is taken from current classification data, a mild look-ahead for firms that have since been reclassified.

## calendar (7)

| Feature | Lookback | Definition | Why it might inform the forecast |
|---|---|---|---|
| `dow_tue` | 1 | Indicator for tue<br>`1[weekday(t) == 1]` | Day-of-week effects are small but documented; Monday is the omitted baseline so the dummies are not collinear with an intercept |
| `dow_wed` | 1 | Indicator for wed<br>`1[weekday(t) == 2]` | Day-of-week effects are small but documented; Monday is the omitted baseline so the dummies are not collinear with an intercept |
| `dow_thu` | 1 | Indicator for thu<br>`1[weekday(t) == 3]` | Day-of-week effects are small but documented; Monday is the omitted baseline so the dummies are not collinear with an intercept |
| `dow_fri` | 1 | Indicator for fri<br>`1[weekday(t) == 4]` | Day-of-week effects are small but documented; Monday is the omitted baseline so the dummies are not collinear with an intercept |
| `month_sin` | 1 | Sine component of the month-of-year cycle<br>`sin(2 pi month / 12)` | Encodes seasonality as a cycle so December and January are adjacent, which an integer month code gets wrong |
| `month_cos` | 1 | Cosine component of the month-of-year cycle<br>`cos(2 pi month / 12)` | Second component needed to make the cyclic encoding one-to-one |
| `days_to_month_end` | 1 | Calendar days until month end, scaled to roughly [0, 1]<br>`(month_end(t) - t) / 31` | The turn of the month coincides with recurring flows from pension and index rebalancing, a documented seasonal pattern |

Leakage notes:

- Derived from the timestamp of row t itself.
- Uses only the calendar position of row t; the month-end date is known in advance and is not an observation.
