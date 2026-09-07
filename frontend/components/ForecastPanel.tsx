"use client";

import type { Forecast, RegimeNow } from "@/lib/schema";
import { num, pct, targetLabel } from "@/lib/format";
import { Note, Section, Table, Td, Th } from "./primitives";

/** The edge over the baseline is the number that means something, so it gets its own
 *  column rather than being left for the reader to compute. */
export function ForecastPanel({
  forecasts,
  regime,
}: {
  forecasts: Forecast[];
  regime: RegimeNow | null;
}) {
  return (
    <Section
      id="forecast"
      title="Forecast"
      subtitle="Probability that the return over the horizon is positive. Each row shows what always predicting the majority class would score, because a probability near the base rate carries no information."
      aside={forecasts[0] ? `model ${forecasts[0].model}` : undefined}
    >
      <Table>
        <thead>
          <tr>
            <Th align="left">Target</Th>
            <Th align="left">Horizon</Th>
            <Th>Probability</Th>
            <Th>Baseline</Th>
            <Th title="probability minus the majority-class rate">Edge</Th>
            <Th align="left">Reading</Th>
          </tr>
        </thead>
        <tbody>
          {forecasts.map((row) => {
            const edge = row.probability - row.baseline_rate;
            const meaningful = Math.abs(edge) >= 0.01;
            return (
              <tr key={`${row.target}-${row.horizon}`}>
                <Td align="left">
                  <span className="text-ink">{targetLabel(row.target)}</span>
                  <span className="ml-2 font-mono text-2xs text-faint">{row.target}</span>
                </Td>
                <Td align="left">{row.horizon}d</Td>
                <Td className="text-[13px]">{pct(row.probability, 1)}</Td>
                <Td className="text-muted">{pct(row.baseline_rate, 1)}</Td>
                <Td className={meaningful ? (edge > 0 ? "text-up" : "text-down") : "text-faint"}>
                  {edge >= 0 ? "+" : ""}
                  {(edge * 100).toFixed(1)}pp
                </Td>
                <Td align="left" className="text-muted">
                  {meaningful
                    ? edge > 0
                      ? "above the base rate"
                      : "below the base rate"
                    : "indistinguishable from the base rate"}
                </Td>
              </tr>
            );
          })}
        </tbody>
      </Table>

      {regime ? (
        <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line pt-4">
          <div className="text-2xs uppercase tracking-[0.06em] text-faint">Current regime</div>
          <div className="text-[13px] text-ink">{regime.name}</div>
          <div className="flex items-center gap-3">
            {regime.states.map((state) => (
              <div key={state.state} className="flex items-center gap-1.5">
                <span className="font-mono text-2xs text-faint">s{state.state}</span>
                <span className="inline-block h-1.5 w-14 bg-line/60">
                  <span
                    className={`block h-full ${state.state === regime.state ? "bg-accent" : "bg-faint/50"}`}
                    style={{ width: `${state.probability * 100}%` }}
                  />
                </span>
                <span className="font-mono text-2xs tabular text-muted">{num(state.probability, 2)}</span>
              </div>
            ))}
          </div>
          <span className="font-mono text-2xs text-faint">{regime.inference} inference</span>
        </div>
      ) : null}

      <Note>
        These are model outputs, not advice. Measured out-of-sample edges on this problem are
        roughly one percentage point of AUC, and the friction analysis shows costs remove them.
      </Note>
    </Section>
  );
}
