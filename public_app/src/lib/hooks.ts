import { useQuery } from "@tanstack/react-query";
import {
  api,
  type BackendInfo,
  type DashboardSummary,
  type RunListItem,
  type TelemetrySample,
} from "@/lib/api";

/**
 * Centralized React Query keys + hooks. Keep these here instead of inlining
 * `useQuery` at call sites — single source of truth for cache keys and
 * stale-time policy means invalidating across components is one edit.
 */

export const queryKeys = {
  backend: ["backend-info"] as const,
  telemetry: ["telemetry"] as const,
  dashboard: ["dashboard"] as const,
  runs: (params?: { limit?: number; modality?: string }) =>
    ["runs", params] as const,
  runDetail: (runId: string) => ["runs", runId] as const,
};

/**
 * Backend identity + capabilities. Stable across a session so we cache for
 * 5 min — the backend doesn't switch under us at runtime.
 */
export function useBackendInfo() {
  return useQuery<BackendInfo>({
    queryKey: queryKeys.backend,
    queryFn: api.backendInfo,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });
}

/**
 * Dashboard summary for the overview route. Polled every 15s while the tab
 * is focused — covers the "is anything still running?" question without
 * hammering the backend.
 */
export function useDashboard() {
  return useQuery<DashboardSummary>({
    queryKey: queryKeys.dashboard,
    queryFn: api.dashboard,
    refetchInterval: 15_000,
    refetchIntervalInBackground: false,
  });
}

/**
 * Hardware telemetry — backs the strip across the top of every page.
 * Polled aggressively (3s) because watching values change *is the
 * point* of the strip; it's the visual heartbeat of the application.
 *
 * `placeholderData: previous` means the strip shows the previous frame
 * while a new request is in flight, instead of flickering to skeletons.
 */
export function useTelemetry() {
  return useQuery<TelemetrySample>({
    queryKey: queryKeys.telemetry,
    queryFn: api.telemetry,
    refetchInterval: 3_000,
    refetchIntervalInBackground: false,
    placeholderData: (prev) => prev,
    staleTime: 0, // every poll is fresh data
  });
}

/**
 * Run list — recent first. Same polling cadence as dashboard so the two
 * views stay coherent on the same screen.
 */
export function useRuns(params?: { limit?: number; modality?: string }) {
  return useQuery<{ items: RunListItem[] }>({
    queryKey: queryKeys.runs(params),
    queryFn: () => api.listRuns(params),
    refetchInterval: 15_000,
    refetchIntervalInBackground: false,
  });
}
