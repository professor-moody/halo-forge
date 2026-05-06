/**
 * Typed FastAPI client for the halo-forge public API.
 *
 * Mounted at `/api/public/*` by `halo_forge.public_api.app.create_app()`.
 * In dev, Vite proxies `/api/*` to `http://127.0.0.1:8000` (see
 * vite.config.ts). In prod, the FastAPI host is expected to serve the
 * built frontend under the same origin, so all calls are same-origin.
 *
 * This module is the single boundary between the React app and the
 * backend — every fetch should go through it so retry/auth/error
 * normalization can be added in one place.
 */

const API_BASE = "/api/public";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public payload?: unknown,
  ) {
    super(`${status} ${detail}`);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let payload: unknown = undefined;
    let detail = res.statusText;
    try {
      payload = await res.json();
      if (payload && typeof payload === "object" && "detail" in payload) {
        detail = String((payload as { detail: unknown }).detail ?? detail);
      }
    } catch {
      // body wasn't JSON; keep statusText as the detail
    }
    throw new ApiError(res.status, detail, payload);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/* -------------------------------------------------------------------------
 * Endpoint typings.
 *
 * These mirror the FastAPI side. Keep them narrow — only the fields the
 * frontend actually reads — so a backend extension doesn't ripple into
 * frontend type churn.
 * ----------------------------------------------------------------------- */

export type TelemetrySample = {
  timestamp: number;
  backend: string;
  device_name: string | null;
  gpu_util_percent: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  power_watts: number | null;
  temp_celsius: number | null;
  cpu_util_percent: number | null;
  sys_mem_used_gb: number | null;
  sys_mem_total_gb: number | null;
  throughput_tokens_per_sec: number | null;
  active_run_id: string | null;
  note: string | null;
};

export type BackendInfo = {
  name: string;
  device: string;
  capabilities: {
    name: string;
    supports_bf16: boolean;
    supports_fp16: boolean;
    preferred_dtype_str: string;
    supports_4bit: boolean;
    supports_8bit: boolean;
    supports_flash_attn: boolean;
    preferred_attn_impl: string;
    supports_training: boolean;
    supports_peft: boolean;
  };
  training_defaults: Record<string, unknown>;
  inference_defaults: Record<string, unknown>;
};

export type DashboardSummary = {
  readiness_tier?: string;
  generated_at?: string | null;
  cycles?: Array<Record<string, unknown>>;
  // The backend evolves; keep this open. Frontend should narrow at call site.
  [key: string]: unknown;
};

export type RunListItem = {
  run_id: string;
  modality: string;
  model_name: string;
  status?: string;
  created_at?: string | null;
  cycles_executed?: number;
  weights_updated?: boolean;
  final_train_loss?: number | null;
  effectiveness?: { verdict?: string };
  [key: string]: unknown;
};

export type RunDetail = RunListItem & {
  cycles?: Array<Record<string, unknown>>;
  yield_diagnostics?: Record<string, unknown>;
  recovery?: Record<string, unknown>;
};

export type TrainingDataset = {
  key: string;
  huggingface_id: string;
  description: string;
  domain: "code" | "vlm" | "audio" | "reasoning" | "agentic" | string;
  size_hint: string;
  default_split: string;
};

export type TrainingVerifier = {
  key: string;
  label: string;
  toolchain: string;
  modality: string;
  platforms: string[];
};

export type SuggestedModel = {
  id: string;
  for_backend: string;
};

export type TrainingPreset = {
  key: string;
  mode: string;
  label: string;
  description: string;
  when_to_use?: string;
  expected_runtime?: string;
  yield_safety?: string;
  required_fields: string[];
  optional_fields: string[];
  values: Record<string, unknown>;
};

/* -------------------------------------------------------------------------
 * Endpoints
 * ----------------------------------------------------------------------- */

export const api = {
  health: () => request<{ ok: boolean }>("/health"),
  backendInfo: () => request<BackendInfo>("/backend"),
  telemetry: () => request<TelemetrySample>("/telemetry"),
  dashboard: () => request<DashboardSummary>("/dashboard"),
  trainingPresets: () => request<{ items: TrainingPreset[] }>("/train/presets"),
  trainingDatasets: () => request<{ items: TrainingDataset[] }>("/train/datasets"),
  trainingVerifiers: () => request<{ items: TrainingVerifier[] }>("/train/verifiers"),
  trainingModels: () => request<{ items: SuggestedModel[] }>("/train/models"),
  trainingPreflight: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>("/train/preflight", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  trainingLaunch: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>("/train/launch", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listRuns: (params?: { limit?: number; modality?: string }) => {
    const search = new URLSearchParams();
    if (params?.limit) search.set("limit", String(params.limit));
    if (params?.modality) search.set("modality", params.modality);
    const qs = search.toString();
    return request<{ items: RunListItem[] }>(`/runs${qs ? `?${qs}` : ""}`);
  },
  runDetail: (runId: string) => request<RunDetail>(`/runs/${encodeURIComponent(runId)}`),
};
