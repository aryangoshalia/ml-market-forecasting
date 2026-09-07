"use client";

import type { Explanation } from "@/lib/schema";
import { num, pct } from "@/lib/format";
import { Empty, Note, Section } from "./primitives";

/** A waterfall in log-odds. The base value plus the contributions reconstructs the raw
 *  probability exactly; the reported forecast then applies calibration on top. Both are
 *  shown because a reader who checks the arithmetic should find it consistent. */
export function ExplainPanel({ explanation }: { explanation?: Explanation | null }) {
  if (!explanation) {
    return (
      <Section id="explain" title="Why this prediction">
        <Empty>no explanation available</Empty>
      </Section>
    );
  }

  const largest = Math.max(...explanation.contributions.map((c) => Math.abs(c.contribution)), 1e-9);
  const positives = explanation.contributions.filter((c) => c.contribution > 0);
  const negatives = explanation.contributions.filter((c) => c.contribution < 0);

  return (
    <Section
      id="explain"
      title="Why this prediction"
      subtitle="Feature contributions from SHAP, in log-odds."
      aside={`${explanation.space} space`}
    >
      <div className="mb-5 flex flex-wrap items-baseline gap-x-3 gap-y-1 font-mono text-[12px] tabular text-muted">
        <span>base {explanation.base_value >= 0 ? "+" : ""}{num(explanation.base_value, 4)}</span>
        <span className="text-faint">+</span>
        <span>
          contributions {explanation.logodds - explanation.base_value >= 0 ? "+" : ""}
          {num(explanation.logodds - explanation.base_value, 4)}
        </span>
        <span className="text-faint">=</span>
        <span className="text-ink">
          {explanation.logodds >= 0 ? "+" : ""}
          {num(explanation.logodds, 4)} log-odds
        </span>
        <span className="text-faint">→</span>
        <span>raw {pct(explanation.raw_probability, 1)}</span>
        <span className="text-faint">→ calibrated</span>
        <span className="text-ink">{pct(explanation.probability, 1)}</span>
      </div>

      <div className="grid gap-x-10 gap-y-6 md:grid-cols-2">
        {[
          { label: "Pushing the probability up", rows: positives, tone: "up" as const },
          { label: "Pushing it down", rows: negatives, tone: "down" as const },
        ].map((group) => (
          <div key={group.label}>
            <div className="mb-2 text-2xs uppercase tracking-[0.06em] text-faint">{group.label}</div>
            {group.rows.length === 0 ? (
              <div className="text-2xs text-faint">none</div>
            ) : (
              <ul className="space-y-1.5">
                {group.rows.map((row) => (
                  <li key={row.feature} className="grid grid-cols-[1fr_80px_64px] items-center gap-2">
                    <span className="truncate font-mono text-[12px] text-ink" title={row.feature}>
                      {row.feature}
                    </span>
                    <span className="h-2 bg-line/50">
                      <span
                        className={`block h-full ${group.tone === "up" ? "bg-up" : "bg-down"}`}
                        style={{ width: `${(Math.abs(row.contribution) / largest) * 100}%` }}
                      />
                    </span>
                    <span className={`text-right font-mono text-2xs tabular ${group.tone === "up" ? "text-up" : "text-down"}`}>
                      {row.contribution >= 0 ? "+" : ""}
                      {num(row.contribution, 4)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>

      <Note>{explanation.caveat}</Note>
    </Section>
  );
}
