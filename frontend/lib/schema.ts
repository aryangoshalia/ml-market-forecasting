import { z } from "zod";

/** Every response is validated at runtime. A shape change in the API should surface
 *  as a clear error here rather than as an undefined deep inside a chart. */

export const coverage = z.object({
  first_session: z.string(),
  last_session: z.string(),
  rows: z.number(),
  folds: z.number(),
  models: z.array(z.string()),
  run_id: z.string(),
  source: z.string(),
});

export const health = z.object({
  status: z.string(),
  shipped_artifacts: z.boolean(),
  universe_size: z.number(),
  feature_version: z.string(),
});

export const universe = z.object({
  benchmark: z.string(),
  volatility_index: z.string(),
  tickers: z.array(z.object({ ticker: z.string(), group: z.string(), sector: z.string() })),
  groups: z.record(z.number()),
});

export const snapshot = z.object({
  ticker: z.string(),
  name: z.string().nullable(),
  exchange: z.string().nullable(),
  last_session: z.string(),
  sessions_stale: z.number(),
  close: z.number(),
  change: z.number(),
  change_pct: z.number(),
  volume: z.number(),
  in_training_universe: z.boolean(),
  sector_etf: z.string(),
  sector_is_proxy: z.boolean(),
});

export const forecast = z.object({
  target: z.string(),
  horizon: z.number(),
  probability: z.number(),
  direction: z.string(),
  threshold: z.number(),
  baseline_rate: z.number(),
  model: z.string(),
  model_version: z.string(),
});

export const contribution = z.object({
  feature: z.string(),
  direction: z.string(),
  contribution: z.number(),
  value: z.number(),
});

export const explanation = z.object({
  base_value: z.number(),
  logodds: z.number(),
  raw_probability: z.number(),
  probability: z.number(),
  contributions: z.array(contribution),
  space: z.string(),
  caveat: z.string(),
});

export const regimeNow = z.object({
  state: z.number(),
  name: z.string(),
  probability: z.number(),
  states: z.array(z.object({ state: z.number(), name: z.string(), probability: z.number() })),
  inference: z.string(),
});

export const analysis = z.object({
  snapshot,
  forecasts: z.array(forecast),
  regime: regimeNow.nullable(),
  explanation: explanation.nullable(),
  disclaimer: z.string(),
});

export const priceHistory = z.object({
  ticker: z.string(),
  points: z.array(
    z.object({
      date: z.string(),
      open: z.number(),
      high: z.number(),
      low: z.number(),
      close: z.number(),
      volume: z.number(),
    }),
  ),
  indicators: z.record(z.array(z.number().nullable())),
});

export const predictionHistory = z.object({
  ticker: z.string(),
  target: z.string(),
  horizon: z.number(),
  coverage,
  rows: z.array(
    z.object({
      date: z.string(),
      model: z.string(),
      probability: z.number(),
      predicted: z.number(),
      actual: z.number(),
      correct: z.boolean(),
      fold: z.number(),
      regime: z.number().nullable(),
    }),
  ),
  accuracy: z.number(),
  n: z.number(),
});

export const performance = z.object({
  target: z.string(),
  horizon: z.number(),
  experiment: z.string(),
  coverage,
  scores: z.array(
    z.object({
      model: z.string(),
      is_baseline: z.boolean(),
      folds: z.number(),
      accuracy: z.number(),
      accuracy_over_base_rate: z.number(),
      roc_auc: z.number(),
      pr_auc: z.number(),
      brier: z.number(),
      ece: z.number(),
    }),
  ),
  note: z.string(),
});

export const regimeTimeline = z.object({
  episodes: z.array(
    z.object({
      state: z.number(),
      name: z.string(),
      start: z.string(),
      end: z.string(),
      sessions: z.number(),
    }),
  ),
  names: z.record(z.string()),
  n_states: z.number(),
  model: z.string(),
  inference: z.string(),
});

export const modelCard = z.object({
  card: z.record(z.unknown()),
  universe: z.record(z.unknown()),
  limitations: z.array(z.string()),
});

export type Coverage = z.infer<typeof coverage>;
export type Analysis = z.infer<typeof analysis>;
export type Snapshot = z.infer<typeof snapshot>;
export type Forecast = z.infer<typeof forecast>;
export type Explanation = z.infer<typeof explanation>;
export type RegimeNow = z.infer<typeof regimeNow>;
export type PriceHistory = z.infer<typeof priceHistory>;
export type PredictionHistory = z.infer<typeof predictionHistory>;
export type Performance = z.infer<typeof performance>;
export type RegimeTimeline = z.infer<typeof regimeTimeline>;
export type ModelCard = z.infer<typeof modelCard>;
export type Universe = z.infer<typeof universe>;
export type Health = z.infer<typeof health>;
