"use client";

import type { Performance } from "@/lib/schema";
import { num, pct } from "@/lib/format";
import { CoverageLabel } from "./CoverageLabel";
import { Bar, Empty, Note, Section, Table, Td, Th } from "./primitives";

export function PerformancePanel({
  data,
  target,
  horizon,
  onTarget,
  onHorizon,
}: {
  data?: Performance;
  target: string;
  horizon: number;
  onTarget: (value: string) => void;
  onHorizon: (value: number) => void;
}) {
  const best = data ? Math.max(...data.scores.map((s) => Math.abs(s.roc_auc - 0.5))) : 0;

  return (
    <Section
      id="performance"
      title="Model performance"
      subtitle="Averages across walk-forward folds. Baselines are listed alongside the learners because an accuracy figure means nothing without the floor it has to clear."
      aside={data ? <CoverageLabel coverage={data.coverage} /> : undefined}
    >
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-2xs">
        <span className="text-faint">target</span>
        {["excess_direction", "direction"].map((value) => (
          <button
            key={value}
            onClick={() => onTarget(value)}
            className={value === target ? "text-accent" : "text-faint hover:text-muted"}
          >
            {value === "excess_direction" ? "vs benchmark" : "absolute"}
          </button>
        ))}
        <span className="ml-4 text-faint">horizon</span>
        {[1, 5].map((value) => (
          <button
            key={value}
            onClick={() => onHorizon(value)}
            className={value === horizon ? "text-accent" : "text-faint hover:text-muted"}
          >
            {value}d
          </button>
        ))}
      </div>

      {data ? (
        <>
          <Table>
            <thead>
              <tr>
                <Th align="left">Model</Th>
                <Th title="area under the ROC curve, averaged over folds">ROC-AUC</Th>
                <Th>Distance from chance</Th>
                <Th>PR-AUC</Th>
                <Th>Accuracy</Th>
                <Th title="accuracy minus the best constant predictor on that fold">vs best constant</Th>
                <Th title="mean squared error of the probability">Brier</Th>
                <Th title="expected calibration error">ECE</Th>
              </tr>
            </thead>
            <tbody>
              {data.scores.map((row) => (
                <tr key={row.model} className={row.is_baseline ? "text-muted" : ""}>
                  <Td align="left">
                    <span className={row.is_baseline ? "text-muted" : "text-ink"}>{row.model}</span>
                    {row.is_baseline ? (
                      <span className="ml-2 text-2xs uppercase tracking-[0.06em] text-faint">baseline</span>
                    ) : null}
                  </Td>
                  <Td className={row.is_baseline ? "" : "text-ink"}>{num(row.roc_auc)}</Td>
                  <Td>
                    <Bar value={row.roc_auc - 0.5} max={best} tone={row.roc_auc >= 0.5 ? "accent" : "down"} />
                  </Td>
                  <Td>{num(row.pr_auc)}</Td>
                  <Td>{pct(row.accuracy, 2)}</Td>
                  <Td className={row.accuracy_over_base_rate >= 0 ? "text-up" : "text-down"}>
                    {(row.accuracy_over_base_rate * 100).toFixed(2)}pp
                  </Td>
                  <Td>{num(row.brier)}</Td>
                  <Td>{num(row.ece)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          <Note>{data.note}</Note>
        </>
      ) : (
        <Empty>loading performance</Empty>
      )}
    </Section>
  );
}
