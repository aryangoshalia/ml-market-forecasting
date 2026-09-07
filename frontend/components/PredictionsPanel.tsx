"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PredictionHistory } from "@/lib/schema";
import { pct, shortDate } from "@/lib/format";
import { CoverageLabel } from "./CoverageLabel";
import { Empty, Note, Section, Table, Td, Th } from "./primitives";

const MODELS = ["lightgbm", "xgboost", "random_forest", "logistic", "persistence", "majority"];
const AXIS = { fontSize: 10, fill: "var(--faint)" };

export function PredictionsPanel({
  data,
  model,
  onModel,
}: {
  data?: PredictionHistory;
  model: string;
  onModel: (value: string) => void;
}) {
  const [onlyWrong, setOnlyWrong] = useState(false);

  const confidence = useMemo(() => {
    if (!data) return [];
    const buckets = new Map<number, { n: number; correct: number }>();
    for (const row of data.rows) {
      const distance = Math.abs(row.probability - 0.5);
      const bucket = Math.min(9, Math.floor(distance * 40));
      const current = buckets.get(bucket) ?? { n: 0, correct: 0 };
      current.n += 1;
      current.correct += row.correct ? 1 : 0;
      buckets.set(bucket, current);
    }
    return [...buckets.entries()]
      .sort((a, b) => a[0] - b[0])
      .filter(([, v]) => v.n >= 20)
      .map(([bucket, v]) => ({
        confidence: (bucket / 40 + 0.0125).toFixed(3),
        accuracy: v.correct / v.n,
        n: v.n,
      }));
  }, [data]);

  const rolling = useMemo(() => {
    if (!data) return [];
    const window = 60;
    const out: { date: string; accuracy: number }[] = [];
    for (let i = window; i < data.rows.length; i += 5) {
      const slice = data.rows.slice(i - window, i);
      const correct = slice.filter((r) => r.correct).length;
      out.push({ date: data.rows[i]!.date, accuracy: correct / slice.length });
    }
    return out;
  }, [data]);

  if (!data) {
    return (
      <Section
        id="predictions"
        title="Historical predictions"
        subtitle="Out-of-sample predictions from the walk-forward study."
      >
        <Empty>
          This ticker is held out. The study ran on the twenty development tickers; the
          validation and test groups are scored once at the end so their result stays an
          honest estimate of performance on assets the model never saw.
        </Empty>
      </Section>
    );
  }

  const shown = (onlyWrong ? data.rows.filter((r) => !r.correct) : data.rows).slice(-120).reverse();

  return (
    <Section
      id="predictions"
      title="Historical predictions"
      subtitle="Out-of-sample predictions from the walk-forward study. Every row was produced by a model that had not seen that session."
      aside={<CoverageLabel coverage={data.coverage} />}
    >
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-2xs">
        <span className="text-faint">model</span>
        {MODELS.map((value) => (
          <button
            key={value}
            onClick={() => onModel(value)}
            className={value === model ? "text-accent" : "text-faint hover:text-muted"}
          >
            {value}
          </button>
        ))}
        <span className="ml-auto text-muted tabular">
          {data.n.toLocaleString()} predictions · accuracy {pct(data.accuracy, 2)}
        </span>
      </div>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <div>
          <div className="mb-1 text-2xs uppercase tracking-[0.06em] text-faint">
            Accuracy by confidence
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={confidence} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="confidence" tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--line)" }} />
              <YAxis domain={[0.4, 0.6]} tick={AXIS} tickLine={false} axisLine={false} width={38} />
              <ReferenceLine y={0.5} stroke="var(--faint)" strokeDasharray="2 2" />
              <Tooltip
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", fontSize: 11 }}
                formatter={(v: number) => v.toFixed(4)}
              />
              <Bar dataKey="accuracy" fill="var(--accent)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="mb-1 text-2xs uppercase tracking-[0.06em] text-faint">
            Rolling 60-session accuracy
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={rolling} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="date" tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--line)" }} minTickGap={60} />
              <YAxis domain={[0.3, 0.7]} tick={AXIS} tickLine={false} axisLine={false} width={38} />
              <ReferenceLine y={0.5} stroke="var(--faint)" strokeDasharray="2 2" />
              <Tooltip
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", fontSize: 11 }}
                formatter={(v: number) => v.toFixed(4)}
              />
              <Line type="monotone" dataKey="accuracy" stroke="var(--accent)" dot={false} strokeWidth={1.25} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mb-2 flex items-center gap-4 text-2xs">
        <button
          onClick={() => setOnlyWrong(!onlyWrong)}
          className={onlyWrong ? "text-accent" : "text-faint hover:text-muted"}
        >
          {onlyWrong ? "showing errors only" : "show errors only"}
        </button>
        <span className="text-faint">most recent 120 of {shown.length.toLocaleString()}</span>
      </div>

      <Table>
        <thead>
          <tr>
            <Th align="left">Date</Th>
            <Th>Probability</Th>
            <Th align="left">Predicted</Th>
            <Th align="left">Actual</Th>
            <Th align="left">Result</Th>
            <Th>Regime</Th>
            <Th>Fold</Th>
          </tr>
        </thead>
        <tbody>
          {shown.map((row) => (
            <tr key={row.date}>
              <Td align="left" className="font-mono">{shortDate(row.date)}</Td>
              <Td>{pct(row.probability, 1)}</Td>
              <Td align="left">{row.predicted === 1 ? "up" : "down"}</Td>
              <Td align="left">{row.actual === 1 ? "up" : "down"}</Td>
              <Td align="left" className={row.correct ? "text-up" : "text-down"}>
                {row.correct ? "correct" : "wrong"}
              </Td>
              <Td>{row.regime ?? "-"}</Td>
              <Td className="text-faint">{row.fold}</Td>
            </tr>
          ))}
        </tbody>
      </Table>

      <Note>
        Accuracy sits close to the base rate throughout. The confidence panel is the one to read:
        if accuracy does not rise with confidence, the probabilities carry no usable ordering.
      </Note>
    </Section>
  );
}
