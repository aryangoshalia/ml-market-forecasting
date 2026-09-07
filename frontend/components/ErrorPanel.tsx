"use client";

import { useMemo } from "react";
import type { PredictionHistory, RegimeTimeline } from "@/lib/schema";
import { pct, shortDate } from "@/lib/format";
import { Empty, Note, Section, Table, Td, Th } from "./primitives";

/** Aggregate accuracy hides where a model fails. This splits the same predictions by
 *  regime and by confidence, and lists the errors made with the most conviction. */
export function ErrorPanel({
  data,
  regimes,
}: {
  data?: PredictionHistory;
  regimes?: RegimeTimeline;
}) {
  const byRegime = useMemo(() => {
    if (!data) return [];
    const buckets = new Map<number, { n: number; correct: number; up: number }>();
    for (const row of data.rows) {
      if (row.regime == null) continue;
      const current = buckets.get(row.regime) ?? { n: 0, correct: 0, up: 0 };
      current.n += 1;
      current.correct += row.correct ? 1 : 0;
      current.up += row.actual;
      buckets.set(row.regime, current);
    }
    return [...buckets.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([state, v]) => ({
        state,
        name: regimes?.names[String(state)] ?? `state ${state}`,
        n: v.n,
        accuracy: v.correct / v.n,
        baseRate: v.up / v.n,
      }));
  }, [data, regimes]);

  const confident = useMemo(() => {
    if (!data) return [];
    const sorted = [...data.rows].sort(
      (a, b) => Math.abs(b.probability - 0.5) - Math.abs(a.probability - 0.5),
    );
    const cutoff = Math.max(1, Math.floor(sorted.length * 0.1));
    return sorted.slice(0, cutoff).filter((row) => !row.correct).slice(0, 12);
  }, [data]);

  const confidentStats = useMemo(() => {
    if (!data) return null;
    const sorted = [...data.rows].sort(
      (a, b) => Math.abs(b.probability - 0.5) - Math.abs(a.probability - 0.5),
    );
    const cutoff = Math.max(1, Math.floor(sorted.length * 0.1));
    const top = sorted.slice(0, cutoff);
    const bottom = sorted.slice(-cutoff);
    return {
      top: top.filter((r) => r.correct).length / top.length,
      bottom: bottom.filter((r) => r.correct).length / bottom.length,
      n: cutoff,
    };
  }, [data]);

  if (!data) {
    return (
      <Section id="errors" title="Error analysis">
        <Empty>
          No prediction history for this ticker, so there are no errors to break down. Held-out
          tickers are scored once, at the end of the study.
        </Empty>
      </Section>
    );
  }

  return (
    <Section
      id="errors"
      title="Error analysis"
      subtitle="Where the model is wrong, and whether it is wrong when it is most confident."
    >
      <div className="grid gap-8 lg:grid-cols-2">
        <div>
          <div className="mb-2 text-2xs uppercase tracking-[0.06em] text-faint">By market regime</div>
          <Table>
            <thead>
              <tr>
                <Th align="left">Regime</Th>
                <Th>Predictions</Th>
                <Th>Accuracy</Th>
                <Th title="share of sessions that actually went up">Base rate</Th>
                <Th>Edge</Th>
              </tr>
            </thead>
            <tbody>
              {byRegime.map((row) => {
                const floor = Math.max(row.baseRate, 1 - row.baseRate);
                const edge = row.accuracy - floor;
                return (
                  <tr key={row.state}>
                    <Td align="left" className="text-muted">
                      <span className="mr-2 font-mono text-ink">s{row.state}</span>
                      {row.name}
                    </Td>
                    <Td>{row.n.toLocaleString()}</Td>
                    <Td>{pct(row.accuracy, 2)}</Td>
                    <Td className="text-muted">{pct(floor, 2)}</Td>
                    <Td className={edge >= 0 ? "text-up" : "text-down"}>
                      {(edge * 100).toFixed(2)}pp
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        </div>

        <div>
          <div className="mb-2 text-2xs uppercase tracking-[0.06em] text-faint">
            Highest-conviction errors
          </div>
          {confidentStats ? (
            <p className="mb-3 text-2xs text-muted tabular">
              Top confidence decile: {pct(confidentStats.top, 2)} accurate. Bottom decile:{" "}
              {pct(confidentStats.bottom, 2)}. A model that knows what it does not know shows the
              first number clearly above the second.
            </p>
          ) : null}
          <Table>
            <thead>
              <tr>
                <Th align="left">Date</Th>
                <Th>Probability</Th>
                <Th align="left">Said</Th>
                <Th align="left">Happened</Th>
              </tr>
            </thead>
            <tbody>
              {confident.map((row) => (
                <tr key={row.date}>
                  <Td align="left" className="font-mono">{shortDate(row.date)}</Td>
                  <Td className="text-down">{pct(row.probability, 1)}</Td>
                  <Td align="left">{row.predicted === 1 ? "up" : "down"}</Td>
                  <Td align="left">{row.actual === 1 ? "up" : "down"}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      </div>

      <Note>
        Per-ticker AUC across the development universe ranges from roughly 0.496 to 0.536, so a
        single aggregate figure conceals a wide spread. The full breakdown by volatility, beta and
        ticker is in docs/ANALYSIS.md.
      </Note>
    </Section>
  );
}
