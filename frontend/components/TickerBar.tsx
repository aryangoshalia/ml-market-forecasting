"use client";

import { useEffect, useMemo, useState } from "react";
import type { Snapshot, Universe } from "@/lib/schema";
import { compactVolume, money, shortDate, signClass, signedPct } from "@/lib/format";
import { Chip } from "./primitives";

const GROUP_LABELS: Record<string, string> = {
  dev: "Development — model trained on these",
  val: "Validation — held out",
  test: "Test — held out, scored once",
};
const GROUP_ORDER = ["dev", "val", "test"];

const prettySector = (sector: string) => sector.replace(/_/g, " ");

export function TickerBar({
  ticker,
  snapshot,
  universe,
  loading,
  error,
  onSubmit,
}: {
  ticker: string;
  snapshot?: Snapshot;
  universe?: Universe;
  loading: boolean;
  error?: string;
  onSubmit: (value: string) => void;
}) {
  const [draft, setDraft] = useState(ticker);

  useEffect(() => setDraft(ticker), [ticker]);

  /** Grouped by asset split rather than sector, because whether the model trained on a
   *  ticker is the thing a reader needs to know before trusting its numbers. */
  const grouped = useMemo(() => {
    if (!universe) return [];
    return GROUP_ORDER.map((group) => ({
      group,
      label: GROUP_LABELS[group] ?? group,
      tickers: universe.tickers
        .filter((entry) => entry.group === group)
        .sort((a, b) => a.ticker.localeCompare(b.ticker)),
    })).filter((block) => block.tickers.length > 0);
  }, [universe]);

  const known = universe?.tickers.some((entry) => entry.ticker === ticker) ?? false;

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-ground/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
        <div className="flex items-center gap-2">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              const next = draft.trim().toUpperCase();
              if (next) onSubmit(next);
            }}
            className="flex items-center gap-2"
          >
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value.toUpperCase())}
              aria-label="Ticker symbol"
              spellCheck={false}
              maxLength={16}
              placeholder="TICKER"
              className="w-[104px] border border-line bg-surface px-2 py-1 font-mono text-[13px] uppercase tracking-wide text-ink outline-none placeholder:text-faint focus:border-accent"
            />
            <button
              type="submit"
              className="border border-line px-2.5 py-1 text-2xs uppercase tracking-[0.08em] text-muted hover:border-accent hover:text-accent"
            >
              Analyze
            </button>
          </form>

          <select
            aria-label="Choose from the study universe"
            value={known ? ticker : ""}
            onChange={(event) => {
              if (event.target.value) onSubmit(event.target.value);
            }}
            className="max-w-[230px] border border-line bg-surface px-2 py-1 text-[12px] text-muted outline-none focus:border-accent"
          >
            <option value="">
              {universe ? `study universe (${universe.tickers.length})…` : "loading universe…"}
            </option>
            {grouped.map((block) => (
              <optgroup key={block.group} label={`${block.label} (${block.tickers.length})`}>
                {block.tickers.map((entry) => (
                  <option key={entry.ticker} value={entry.ticker}>
                    {entry.ticker} · {prettySector(entry.sector)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        {snapshot ? (
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-[15px] font-semibold">{snapshot.ticker}</span>
              <span className="max-w-[200px] truncate text-2xs text-muted">{snapshot.name ?? ""}</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-[15px] tabular">{money(snapshot.close)}</span>
              <span className={`font-mono text-[12px] tabular ${signClass(snapshot.change_pct)}`}>
                {signedPct(snapshot.change_pct)}
              </span>
            </div>
            <div className="text-2xs text-muted">
              <span className="text-faint">data through </span>
              <span className="font-mono tabular">{shortDate(snapshot.last_session)}</span>
              {snapshot.sessions_stale > 0 ? (
                <span className="ml-2 text-warn">{snapshot.sessions_stale} sessions behind</span>
              ) : null}
            </div>
            <div className="text-2xs text-faint">
              vol <span className="font-mono tabular">{compactVolume(snapshot.volume)}</span>
            </div>
            <div className="flex items-center gap-2">
              {snapshot.in_training_universe ? (
                <Chip>in study universe</Chip>
              ) : (
                <Chip tone="warn">outside study universe</Chip>
              )}
              {snapshot.sector_is_proxy ? <Chip tone="warn">sector proxied</Chip> : null}
            </div>
          </div>
        ) : (
          <div className="text-2xs text-faint">
            {loading ? "loading…" : error ? <span className="text-down">{error}</span> : "enter a ticker"}
          </div>
        )}

        <div className="ml-auto hidden text-2xs text-faint xl:block">
          any listed symbol works, not only the {universe?.tickers.length ?? 42} studied
        </div>
      </div>
    </header>
  );
}
