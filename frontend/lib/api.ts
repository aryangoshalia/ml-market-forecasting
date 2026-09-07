import type { z } from "zod";
import * as s from "./schema";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, shape: z.ZodType<T>): Promise<T> {
  const response = await fetch(path, { headers: { accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // a non-JSON error body is still worth reporting by status alone
    }
    throw new ApiError(detail, response.status);
  }
  return shape.parse(await response.json());
}

const query = (params: Record<string, string | number | undefined>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value));
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
};

export const api = {
  health: () => request("/api/health", s.health),
  universe: () => request("/api/universe", s.universe),
  analyze: (ticker: string, explain = true) =>
    request(`/api/analyze/${encodeURIComponent(ticker)}${query({ explain: String(explain) })}`, s.analysis),
  history: (ticker: string, sessions = 504) =>
    request(`/api/history/${encodeURIComponent(ticker)}${query({ sessions })}`, s.priceHistory),
  predictions: (ticker: string, target: string, horizon: number, model: string, limit = 500) =>
    request(
      `/api/predictions/${encodeURIComponent(ticker)}${query({ target, horizon, model, limit })}`,
      s.predictionHistory,
    ),
  performance: (target: string, horizon: number, experiment = "main") =>
    request(`/api/performance${query({ target, horizon, experiment })}`, s.performance),
  regimes: () => request("/api/regimes", s.regimeTimeline),
  modelCard: (target: string, horizon: number) =>
    request(`/api/model-card${query({ target, horizon })}`, s.modelCard),
};

export const fetcher = <T>(key: [string, ...unknown[]] | string): Promise<T> => {
  throw new Error(`use a typed api.* call instead of a bare fetcher: ${String(key)}`);
};
