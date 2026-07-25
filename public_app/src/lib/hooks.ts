import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type BackendInfo,
  type ActivitySnapshot,
  type ArtifactOperation,
  type ArtifactQualification,
  type DashboardSummary,
  type DatasetImportSession,
  type DatasetSourceInspection,
  type HuggingFaceModelAccess,
  type HuggingFaceStatus,
  type ModelCatalogResponse,
  type ModelArtifactOccurrence,
  type PaginatedResponse,
  type RunListItem,
  type ServeLogs,
  type ServeStartPayload,
  type ServeStatus,
  type SuggestedModel,
  type TelemetrySample,
  type TrainingDataset,
  type TrainingScenarioDescriptor,
  type TrainingPreflight,
  type TrainingVerifier,
  type StorageInventory,
  type VersionInfo,
  type WorkspaceInfo,
} from "@/lib/api";

/**
 * Centralized React Query keys + hooks. Keep these here instead of inlining
 * `useQuery` at call sites — single source of truth for cache keys and
 * stale-time policy means invalidating across components is one edit.
 */

export const queryKeys = {
  version: ["version"] as const,
  backend: ["backend-info"] as const,
  workspace: ["workspace-info"] as const,
  telemetry: ["telemetry"] as const,
  dashboard: ["dashboard"] as const,
  runs: (params?: { limit?: number; modality?: string }) =>
    ["runs", params] as const,
  runSearch: (params?: Record<string, unknown>) =>
    ["runs", "search", params] as const,
  runDetail: (runId: string) => ["runs", runId] as const,
  trainingDatasets: ["training", "datasets"] as const,
  trainingVerifiers: ["training", "verifiers"] as const,
  trainingModels: (params?: { mode?: string; modality?: string }) =>
    ["training", "models", params] as const,
  modelCatalog: (params?: Record<string, string | undefined>) =>
    ["models", params] as const,
  serve: ["serve"] as const,
  serveLogs: (tail: number) => ["serve", "logs", tail] as const,
  huggingFace: ["huggingface"] as const,
  huggingFaceModel: (modelId: string) => ["huggingface", "model", modelId] as const,
  activity: ["activity"] as const,
  artifacts: (params?: Record<string, unknown>) => ["artifacts", params] as const,
  artifactOperations: ["artifact-operations"] as const,
  qualifications: (artifactId?: string) => ["qualifications", artifactId] as const,
  storage: ["storage"] as const,
  interfaceCapabilities: ["interface-capabilities"] as const,
  trainingScenarios: (params?: { includeUnavailable?: boolean; modality?: string }) =>
    ["training-scenarios", params] as const,
  trainingScenario: (scenarioId: string) => ["training-scenarios", scenarioId] as const,
  datasetImport: (importId: string) => ["dataset-imports", importId] as const,
  datasetInspection: (inspectionId: string) => ["dataset-inspections", inspectionId] as const,
};

export function useVersionInfo() {
  return useQuery<VersionInfo>({
    queryKey: queryKeys.version,
    queryFn: api.versionInfo,
    staleTime: 60 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });
}

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

export function useWorkspaceInfo() {
  return useQuery<WorkspaceInfo>({
    queryKey: queryKeys.workspace,
    queryFn: api.workspaceInfo,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });
}

export function useHuggingFaceStatus() {
  return useQuery<HuggingFaceStatus>({
    queryKey: queryKeys.huggingFace,
    queryFn: api.huggingFaceStatus,
    staleTime: 60_000,
  });
}

export function useHuggingFaceSaveToken() {
  const queryClient = useQueryClient();
  return useMutation<HuggingFaceStatus, Error, string>({
    mutationFn: (token) => api.huggingFaceSaveToken(token),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.huggingFace });
    },
  });
}

export function useHuggingFaceClearToken() {
  const queryClient = useQueryClient();
  return useMutation<HuggingFaceStatus, Error, void>({
    mutationFn: () => api.huggingFaceClearToken(),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.huggingFace });
    },
  });
}

export function useHuggingFaceCheckModel() {
  return useMutation<HuggingFaceModelAccess, Error, string>({
    mutationFn: (modelId) => api.huggingFaceCheckModel(modelId),
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

/* -------------------------------------------------------------------------
 * Training configurator data sources.
 *
 * Datasets, verifiers, and suggested models are static for the lifetime
 * of a server, so we cache aggressively (10 min) and never refetch on
 * window focus. The configurator can re-render on every keystroke
 * without hitting the backend.
 * ----------------------------------------------------------------------- */

const TRAINING_STATIC_OPTS = {
  staleTime: 10 * 60 * 1000,
  gcTime: 60 * 60 * 1000,
  refetchOnWindowFocus: false as const,
};

export function useTrainingDatasets() {
  return useQuery<{ items: TrainingDataset[] }>({
    queryKey: queryKeys.trainingDatasets,
    queryFn: api.trainingDatasets,
    ...TRAINING_STATIC_OPTS,
  });
}

export function useTrainingVerifiers() {
  return useQuery<{ items: TrainingVerifier[] }>({
    queryKey: queryKeys.trainingVerifiers,
    queryFn: api.trainingVerifiers,
    ...TRAINING_STATIC_OPTS,
  });
}

export function useTrainingModels(params: { mode?: string; modality?: string } = {}) {
  return useQuery<{ items: SuggestedModel[] }>({
    queryKey: queryKeys.trainingModels(params),
    queryFn: () => api.trainingModels(params),
    ...TRAINING_STATIC_OPTS,
  });
}

export function useInterfaceCapabilities() {
  return useQuery({
    queryKey: queryKeys.interfaceCapabilities,
    queryFn: api.interfaceCapabilities,
    staleTime: 10 * 60 * 1000,
  });
}

export function useTrainingScenarios(params: { includeUnavailable?: boolean; modality?: string } = {}) {
  return useQuery<{ items: TrainingScenarioDescriptor[]; total?: number }>({
    queryKey: queryKeys.trainingScenarios(params),
    queryFn: () => api.trainingScenarios(params),
    staleTime: 10 * 60 * 1000,
  });
}

export function useTrainingScenario(scenarioId: string) {
  return useQuery<TrainingScenarioDescriptor>({
    queryKey: queryKeys.trainingScenario(scenarioId),
    queryFn: () => api.trainingScenario(scenarioId),
    enabled: Boolean(scenarioId),
    staleTime: 10 * 60 * 1000,
  });
}

export function useDatasetImport(importId: string) {
  return useQuery<DatasetImportSession>({
    queryKey: queryKeys.datasetImport(importId),
    queryFn: () => api.datasetImport(importId),
    enabled: Boolean(importId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && ["ready", "completed", "failed", "cancelled"].includes(status) ? false : 1_000;
    },
  });
}

export function useDatasetInspection(inspectionId: string) {
  return useQuery<DatasetSourceInspection>({
    queryKey: queryKeys.datasetInspection(inspectionId),
    queryFn: () => api.datasetSourceInspection(inspectionId),
    enabled: Boolean(inspectionId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && ["completed", "failed", "cancelled"].includes(status) ? false : 1_000;
    },
  });
}

export function useModelCatalog(params: Record<string, string | undefined> = {}) {
  return useQuery<ModelCatalogResponse>({
    queryKey: queryKeys.modelCatalog(params),
    queryFn: () => api.modelCatalog(params),
    ...TRAINING_STATIC_OPTS,
  });
}

export function useActivity(limit = 100) {
  return useQuery<ActivitySnapshot>({
    queryKey: queryKeys.activity,
    queryFn: () => api.activity(limit),
    refetchInterval: 3_000,
    refetchIntervalInBackground: false,
    placeholderData: (previous) => previous,
    retry: false,
  });
}

export function useModelArtifacts(params: {
  runId?: string;
  groupId?: string;
  kind?: string;
  query?: string;
  limit?: number;
  offset?: number;
} = {}) {
  return useQuery<PaginatedResponse<ModelArtifactOccurrence>>({
    queryKey: queryKeys.artifacts(params),
    queryFn: () => api.listModelArtifacts(params),
    placeholderData: (previous) => previous,
    retry: false,
  });
}

export function useArtifactOperations() {
  return useQuery<PaginatedResponse<ArtifactOperation>>({
    queryKey: queryKeys.artifactOperations,
    queryFn: () => api.listArtifactOperations({ limit: 100 }),
    refetchInterval: 5_000,
    retry: false,
  });
}

export function useArtifactQualifications(artifactId?: string) {
  return useQuery<PaginatedResponse<ArtifactQualification>>({
    queryKey: queryKeys.qualifications(artifactId),
    queryFn: () => api.listQualifications({ artifactId, limit: 100 }),
    enabled: Boolean(artifactId),
    refetchInterval: 5_000,
    retry: false,
  });
}

export function useStorageInventory() {
  return useQuery<StorageInventory>({
    queryKey: queryKeys.storage,
    queryFn: api.storageInventory,
    refetchInterval: 30_000,
    retry: false,
  });
}

/**
 * Preflight + launch are mutations (POST). Preflight is called on a
 * debounce while the user edits the form so the right-side panel can
 * show live validation; launch is the explicit user-triggered action.
 */
export function useTrainingPreflight() {
  return useMutation<TrainingPreflight, Error, Record<string, unknown>>({
    mutationFn: (payload: Record<string, unknown>) => api.trainingPreflight(payload),
  });
}

export function useTrainingLaunch() {
  return useMutation({
    mutationFn: (payload: Record<string, unknown>) => api.trainingLaunch(payload),
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

/**
 * DB-backed run search (Track F-G). Powers the filter UI on /runs.
 * Uses a longer poll cadence than `useRuns` because it's the queryable
 * surface — frequent re-fetch is wasteful when the user is filtering.
 */
export function useRunSearch(params?: import("@/lib/api").RunSearchParams) {
  return useQuery<import("@/lib/api").RunSearchResponse>({
    queryKey: queryKeys.runSearch(params as Record<string, unknown>),
    queryFn: () => api.searchRuns(params),
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
    placeholderData: (prev) => prev,
  });
}

export function useServeStatus() {
  return useQuery<ServeStatus>({
    queryKey: queryKeys.serve,
    queryFn: api.serveStatus,
    refetchInterval: 3_000,
    refetchIntervalInBackground: false,
    placeholderData: (prev) => prev,
  });
}

export function useServeStart() {
  const queryClient = useQueryClient();
  return useMutation<ServeStatus, Error, ServeStartPayload>({
    mutationFn: (payload) => api.serveStart(payload),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.serve });
      queryClient.invalidateQueries({ queryKey: queryKeys.serveLogs(80) });
    },
  });
}

export function useServeStop() {
  const queryClient = useQueryClient();
  return useMutation<ServeStatus, Error, void>({
    mutationFn: () => api.serveStop(),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.serve });
      queryClient.invalidateQueries({ queryKey: queryKeys.serveLogs(80) });
    },
  });
}

export function useServeLogs(tail = 200, enabled = false) {
  return useQuery<ServeLogs>({
    queryKey: queryKeys.serveLogs(tail),
    queryFn: () => api.serveLogs(tail),
    enabled,
    refetchInterval: enabled ? 3_000 : false,
  });
}
