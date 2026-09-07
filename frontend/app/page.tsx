"use client";

import { useState } from "react";
import { ErrorPanel } from "@/components/ErrorPanel";
import { ExplainPanel } from "@/components/ExplainPanel";
import { ForecastPanel } from "@/components/ForecastPanel";
import { PerformancePanel } from "@/components/PerformancePanel";
import { PredictionsPanel } from "@/components/PredictionsPanel";
import { PriceChart } from "@/components/PriceChart";
import { RegimePanel } from "@/components/RegimePanel";
import { TickerBar } from "@/components/TickerBar";
import { Metric, MetricRow, Section } from "@/components/primitives";
import { compactVolume, money, pct, shortDate, signClass, signedPct } from "@/lib/format";
import {
  useAnalysis,
  useHistory,
  useModelCard,
  usePerformance,
  usePredictions,
  useRegimes,
  useUniverse,
} from "@/lib/hooks";

const SECTIONS = [
  ["overview", "Overview"],
  ["forecast", "Forecast"],
  ["price", "Price & signals"],
  ["regime", "Regime"],
  ["performance", "Model performance"],
  ["explain", "Explainability"],
  ["predictions", "Predictions"],
  ["errors", "Errors"],
] as const;

export default function Page() {
  const [ticker, setTicker] = useState("AAPL");
  const [target, setTarget] = useState("excess_direction");
  const [horizon, setHorizon] = useState(1);
  const [model, setModel] = useState("lightgbm");
  const [sessions, setSessions] = useState(504);

  const analysis = useAnalysis(ticker);
  const history = useHistory(ticker, sessions);
  const predictions = usePredictions(ticker, target, horizon, model);
  const performance = usePerformance(target, horizon, "main");
  const regimes = useRegimes();
  const card = useModelCard(target, horizon);
  const universe = useUniverse();

  const snapshot = analysis.data?.snapshot;

  return (
    <div className="min-h-screen">
      <TickerBar
        ticker={ticker}
        snapshot={snapshot}
        universe={universe.data}
        loading={analysis.isLoading}
        error={analysis.error ? String((analysis.error as Error).message) : undefined}
        onSubmit={setTicker}
      />

      <nav className="border-b border-line bg-ground">
        <div className="mx-auto flex max-w-[1400px] gap-5 overflow-x-auto px-6 py-2">
          {SECTIONS.map(([id, label]) => (
            <a
              key={id}
              href={`#${id}`}
              className="whitespace-nowrap text-2xs uppercase tracking-[0.07em] text-faint hover:text-accent"
            >
              {label}
            </a>
          ))}
        </div>
      </nav>

      <main className="mx-auto max-w-[1400px] px-6 pb-20">
        <Section
          id="overview"
          title="Overview"
          subtitle="Latest available market data, fetched on request. The forecast below is produced live from a model trained on data through the panel's last session."
          aside={card.data ? `model ${String(card.data.card.version ?? "")}` : undefined}
        >
          {snapshot ? (
            <MetricRow>
              <Metric label="Close" value={money(snapshot.close)} hint={snapshot.exchange ?? undefined} />
              <Metric
                label="Session change"
                value={signedPct(snapshot.change_pct)}
                tone={snapshot.change_pct > 0 ? "up" : snapshot.change_pct < 0 ? "down" : "muted"}
              />
              <Metric label="Volume" value={compactVolume(snapshot.volume)} />
              <Metric
                label="Data through"
                value={shortDate(snapshot.last_session)}
                hint={snapshot.sessions_stale > 0 ? `${snapshot.sessions_stale} sessions behind` : "current"}
              />
              <Metric
                label="Regime"
                value={analysis.data?.regime ? `s${analysis.data.regime.state}` : "-"}
                hint={analysis.data?.regime?.name}
                mono={false}
              />
              <Metric
                label="Sector proxy"
                value={snapshot.sector_etf}
                hint={snapshot.sector_is_proxy ? "sector unknown, benchmark used" : undefined}
              />
            </MetricRow>
          ) : (
            <div className="py-6 text-2xs text-faint">
              {analysis.error
                ? `could not analyze ${ticker}: ${(analysis.error as Error).message}`
                : "loading…"}
            </div>
          )}
        </Section>

        {analysis.data ? (
          <ForecastPanel forecasts={analysis.data.forecasts} regime={analysis.data.regime} />
        ) : null}

        <PriceChart history={history.data} sessions={sessions} onSessions={setSessions} />

        <RegimePanel data={regimes.data} />

        <PerformancePanel
          data={performance.data}
          target={target}
          horizon={horizon}
          onTarget={setTarget}
          onHorizon={setHorizon}
        />

        <ExplainPanel explanation={analysis.data?.explanation} />

        <PredictionsPanel data={predictions.data} model={model} onModel={setModel} />

        <ErrorPanel data={predictions.data} regimes={regimes.data} />

        <Section
          id="about"
          title="Limitations"
          subtitle="Stated here rather than left for the reader to discover."
        >
          <ul className="max-w-3xl space-y-1.5 text-[12px] text-muted">
            {(card.data?.limitations ?? []).map((item) => (
              <li key={item} className="border-l-2 border-line pl-3">
                {item}
              </li>
            ))}
          </ul>
          {performance.data ? (
            <p className="mt-4 text-2xs text-faint tabular">
              Best out-of-sample AUC on this formulation:{" "}
              {pct(
                Math.max(...performance.data.scores.filter((s) => !s.is_baseline).map((s) => s.roc_auc)),
                2,
              )}
              . Chance is 50%.
            </p>
          ) : null}
        </Section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto max-w-[1400px] px-6 py-5 text-2xs text-faint">
          {analysis.data?.disclaimer ??
            "Model output for research and education. Not financial advice, and not a recommendation to trade."}
        </div>
      </footer>
    </div>
  );
}
