/** Formatting helpers. Numbers are rendered once, here, so columns stay consistent. */

export const pct = (value: number, digits = 2) => `${(value * 100).toFixed(digits)}%`;

export const signedPct = (value: number, digits = 2) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;

export const num = (value: number, digits = 4) => value.toFixed(digits);

export const money = (value: number) =>
  value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });

export const compactVolume = (value: number) => {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
  return value.toFixed(0);
};

export const shortDate = (iso: string) =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });

export const targetLabel = (target: string) =>
  target === "excess_direction" ? "vs benchmark" : "absolute";

export const horizonLabel = (horizon: number) => `${horizon}d`;

/** Sign drives colour. Nothing else in the interface uses red or green. */
export const signClass = (value: number) =>
  value > 0 ? "text-up" : value < 0 ? "text-down" : "text-muted";
