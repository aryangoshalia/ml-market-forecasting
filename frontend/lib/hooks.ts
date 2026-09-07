"use client";

import useSWR from "swr";
import { api } from "./api";

const options = { revalidateOnFocus: false, shouldRetryOnError: false } as const;

export const useAnalysis = (ticker: string | null) =>
  useSWR(ticker ? ["analyze", ticker] : null, () => api.analyze(ticker as string), options);

export const useHistory = (ticker: string | null, sessions: number) =>
  useSWR(ticker ? ["history", ticker, sessions] : null, () => api.history(ticker as string, sessions), options);

export const usePredictions = (
  ticker: string | null,
  target: string,
  horizon: number,
  model: string,
) =>
  useSWR(
    ticker ? ["predictions", ticker, target, horizon, model] : null,
    () => api.predictions(ticker as string, target, horizon, model, 5000),
    options,
  );

export const usePerformance = (target: string, horizon: number, experiment: string) =>
  useSWR(["performance", target, horizon, experiment], () => api.performance(target, horizon, experiment), options);

export const useRegimes = () => useSWR(["regimes"], () => api.regimes(), options);

export const useUniverse = () => useSWR(["universe"], () => api.universe(), options);

export const useModelCard = (target: string, horizon: number) =>
  useSWR(["model-card", target, horizon], () => api.modelCard(target, horizon), options);
