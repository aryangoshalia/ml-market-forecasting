"use client";

import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  createChart,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { PriceHistory } from "@/lib/schema";
import { Empty, Section } from "./primitives";

const OVERLAYS: Record<string, { label: string; colour: string }> = {
  px_to_sma_20: { label: "price vs 20d average", colour: "#1c5fd6" },
  px_to_sma_50: { label: "price vs 50d average", colour: "#8a6410" },
  rsi_14: { label: "RSI(14)", colour: "#6b6b66" },
  vol_20: { label: "20d volatility", colour: "#a8231d" },
  beta_60: { label: "60d beta", colour: "#17683c" },
};

const toTime = (iso: string) => (Date.parse(`${iso}T00:00:00Z`) / 1000) as UTCTimestamp;

export function PriceChart({ history, sessions, onSessions }: {
  history?: PriceHistory;
  sessions: number;
  onSessions: (value: number) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const overlayRef = useRef<ISeriesApi<"Line"> | null>(null);
  const [overlay, setOverlay] = useState<string>("px_to_sma_50");

  useEffect(() => {
    if (!container.current || !history) return;

    const styles = getComputedStyle(document.documentElement);
    const ink = styles.getPropertyValue("--ink").trim() || "#1a1a19";
    const line = styles.getPropertyValue("--line").trim() || "#e2e2df";
    const faint = styles.getPropertyValue("--faint").trim() || "#9a9a94";

    const chart = createChart(container.current, {
      autoSize: true,
      layout: {
        background: { color: "transparent" },
        textColor: faint,
        fontFamily: "var(--font-mono)",
        // The library places a TradingView logo on the chart by default. It is credited
        // in the footer instead, so the chart carries no third-party branding.
        attributionLogo: false,
      },
      grid: { vertLines: { color: line }, horzLines: { color: line } },
      rightPriceScale: { borderColor: line },
      timeScale: { borderColor: line },
      crosshair: { mode: 0 },
    });
    chartRef.current = chart;

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#17683c",
      downColor: "#a8231d",
      borderUpColor: "#17683c",
      borderDownColor: "#a8231d",
      wickUpColor: "#17683c",
      wickDownColor: "#a8231d",
    });
    candles.setData(
      history.points.map((point) => ({
        time: toTime(point.date),
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
      })),
    );

    const overlaySeries = chart.addSeries(LineSeries, {
      color: OVERLAYS[overlay]?.colour ?? ink,
      lineWidth: 1,
      priceScaleId: "overlay",
      priceLineVisible: false,
      lastValueVisible: false,
    });
    chart.priceScale("overlay").applyOptions({ scaleMargins: { top: 0.75, bottom: 0 } });

    const values = history.indicators[overlay];
    if (values) {
      overlaySeries.setData(
        history.points
          .map((point, index) => ({ time: toTime(point.date), value: values[index] }))
          .filter((row): row is { time: UTCTimestamp; value: number } => row.value != null),
      );
    }
    overlayRef.current = overlaySeries;
    chart.timeScale().fitContent();

    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [history, overlay]);

  return (
    <Section
      id="price"
      title="Price and signals"
      subtitle="Adjusted for splits and dividends across the whole bar, so the range-based indicators stay on one consistent scale."
      aside={
        <div className="flex items-center gap-3">
          {[126, 252, 504, 1260].map((value) => (
            <button
              key={value}
              onClick={() => onSessions(value)}
              className={value === sessions ? "text-accent" : "text-faint hover:text-muted"}
            >
              {value}d
            </button>
          ))}
        </div>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        {Object.entries(OVERLAYS).map(([key, meta]) => (
          <button
            key={key}
            onClick={() => setOverlay(key)}
            className={`text-2xs ${key === overlay ? "text-accent" : "text-faint hover:text-muted"}`}
          >
            {meta.label}
          </button>
        ))}
      </div>
      {history ? (
        <div ref={container} className="h-[380px] w-full border border-line" />
      ) : (
        <Empty>loading price history</Empty>
      )}
    </Section>
  );
}
