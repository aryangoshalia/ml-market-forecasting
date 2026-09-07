# Credits

Everything this project is built on, and where the design came from.

## Data

| Source | Used for | Notes |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com) via [`yfinance`](https://github.com/ranaroussi/yfinance) | daily OHLCV for equities, the benchmark, sector ETFs and the volatility index | Unofficial API. Data is for research and education; redistribution is not part of this project, which is why no price data is committed. |
| [Stooq](https://stooq.com) | secondary provider, implemented for failover and cross-vendor comparison | Currently fronted by a browser-verification challenge, which this project does not attempt to bypass. |
| [Alpha Vantage](https://www.alphavantage.co) | optional second live vendor | Disabled unless `ALPHAVANTAGE_API_KEY` is set. |

## Design

The interface is a restrained financial-research layout: neutral grounds, hairline rules
instead of cards, one accent colour, and red and green reserved strictly for sign.

Influences are the density conventions of professional terminals such as Bloomberg and
FactSet, and the editorial data typography of the *Financial Times* and *The Economist*
statistics pages. No design template, theme or component library was used; the interface
primitives in `frontend/components/primitives.tsx` are written for this project.

### Typefaces

| Face | Designer | Licence |
|---|---|---|
| [Inter](https://rsms.me/inter/) | Rasmus Andersson | SIL Open Font License 1.1 |
| [JetBrains Mono](https://www.jetbrains.com/lp/mono/) | JetBrains | SIL Open Font License 1.1 |

Both are served through `next/font`, which self-hosts them at build time. Every numeric
column uses `font-variant-numeric: tabular-nums` so figures align.

## Charting

| Library | Licence | Notes |
|---|---|---|
| [Lightweight Charts](https://www.tradingview.com/lightweight-charts/) by TradingView | Apache-2.0 | Candlesticks and indicator overlays. The on-chart TradingView attribution logo is disabled in favour of this page, and the library is credited here instead. |
| [Recharts](https://recharts.org) | MIT | Confidence, rolling-accuracy and distribution charts. |

## Frontend

| Package | Version | Licence |
|---|---|---|
| next | 15.5.4 | MIT |
| react, react-dom | 19.1.1 | MIT |
| tailwindcss | 3.4.17 | MIT |
| typescript | 5.9.2 | Apache-2.0 |
| swr | 2.3.6 | MIT |
| zod | 3.25.76 | MIT |
| postcss, autoprefixer, eslint | | MIT |

## Modelling and data pipeline

| Package | Version | Licence |
|---|---|---|
| numpy | 1.26.4 | BSD-3-Clause |
| pandas | 2.3.3 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-3-Clause |
| statsmodels | 0.15.0 | BSD-3-Clause |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| xgboost | 3.2.0 | Apache-2.0 |
| lightgbm | 4.7.0 | MIT |
| hmmlearn | 0.3.3 | BSD-3-Clause |
| shap | 0.49.1 | MIT |
| pyarrow | 25.0.1 | Apache-2.0 |
| pandas-market-calendars | 5.4.0 | MIT |
| matplotlib | 3.11.1 | matplotlib licence (BSD-compatible) |
| fastapi | 0.141.1 | MIT |
| uvicorn | 0.52.4 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| PyYAML | 6.0.3 | MIT |
| tabulate | 0.10.0 | MIT |

## Methodology

The evaluation design follows established practice rather than inventing it.

- Purging and embargoing between training and test windows, so a label whose horizon
  crosses the boundary cannot leak, follows Marcos López de Prado, *Advances in Financial
  Machine Learning* (Wiley, 2018).
- Skip-month momentum (`mom_12_1`) is the construction from Jegadeesh and Titman
  (1993), which omits the most recent month because short-term reversal works against the
  momentum effect.
- Parkinson volatility is the high-low range estimator from Parkinson (1980).
- Kaufman's efficiency ratio for trend quality comes from Perry Kaufman's work on
  adaptive moving averages.
- RiskMetrics EWMA volatility with a decay of 0.94 is the J.P. Morgan RiskMetrics
  convention, used here to scale the neutral band of the three-class target.
- Hidden Markov regimes with filtered rather than smoothed inference follow standard
  state-space practice; the distinction matters because smoothed probabilities condition
  on the whole sample and would be look-ahead.
- The observation that equal weighting is hard to beat out of sample is from DeMiguel,
  Garlappi and Uppal (2009), and motivates reporting buy-and-hold beside every backtest.
