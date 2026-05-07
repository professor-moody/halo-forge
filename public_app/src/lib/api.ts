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

/**
 * Track P1 — bearer-token storage. Halo-forge auth is automatic when
 * the API is bound to non-loopback. The token lives in localStorage so
 * a single dashboard tab persists it across reloads; no cookie path
 * because the API is same-origin in prod and CORS-allowlisted in dev.
 */
const TOKEN_STORAGE_KEY = "halo-forge:api-token";

export function getApiToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setApiToken(token: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (token) {
      window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // localStorage may throw in private mode; auth header just won't fire.
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getApiToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
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

/**
 * Filter / sort / paginate parameters for `/runs/search` (Track F-G).
 */
export type RunSearchParams = {
  modality?: string[];
  status?: string[];
  model?: string;
  since?: string;
  until?: string;
  hasEval?: boolean;
  weightsUpdated?: boolean;
  sortBy?: "timestamp" | "cycles_executed" | "final_train_loss" | "model_name" | string;
  sortDir?: "asc" | "desc";
  limit?: number;
  offset?: number;
};

export type RunSearchResponse = {
  items: RunListItem[];
  total: number;
  filters: Record<string, unknown>;
  facets: {
    modalities: string[];
    models: string[];
  };
};

/**
 * Per-task slice of an eval summary (F-K).
 */
export type EvalTaskCell = {
  primary_metric?: string | null;
  value?: number | null;
  n_samples?: number | null;
  error?: string | null;
};

/**
 * Per-run header in the cohort grid (F-K).
 */
export type EvalCohortRun = {
  run_id: string;
  available: boolean;
  reason?: string | null;
  model_name?: string | null;
  duration_seconds?: number | null;
  backend?: string | null;
};

export type EvalCohortResponse = {
  runs: EvalCohortRun[];
  tasks: string[];
  cells: Record<string, Record<string, EvalTaskCell>>;
  best_per_task_higher_is_better: Record<string, string | null>;
};

export type RunEvalResponse = {
  available: boolean;
  reason?: string | null;
  model_name?: string | null;
  tasks: Array<EvalTaskCell & { task: string }>;
  n_tasks_completed?: number | null;
  duration_seconds?: number | null;
  backend?: string | null;
  summary_path?: string;
};

/**
 * Per-cycle metric row exposed by /api/public/runs/{id}.details.cycle_metrics.
 * Mirrors `_project_cycles_for_charts` on the backend; every numeric field
 * is null-tolerant so older trainers / partial summaries chart cleanly.
 */
export type CycleMetric = {
  cycle: number;
  train_loss: number | null;
  initial_train_loss: number | null;
  eval_loss: number | null;
  avg_reward: number | null;
  avg_kept_reward: number | null;
  success_rate: number | null;
  samples_seen: number | null;
  samples_kept: number | null;
  train_steps_executed: number | null;
  cycle_duration_seconds: number | null;
  learning_rate: number | null;
};

/**
 * The detail endpoint returns a `TrainingRunDetailView` that flattens
 * SFT/RAFT-specific fields into UI-shaped sections: `metrics_summary`
 * carries the headline numbers; `user_summary` carries the
 * confidence/verdict; `details` carries the raw extras (cycles, seed,
 * etc.). The list view has a different (flatter) shape — keep these
 * separate from RunListItem.
 */
export type RunMetricsSummary = {
  progress_percent?: number | null;
  keep_rate?: number | null;
  update_steps?: number | null;
  final_train_loss?: number | null;
  eval_metric_name?: string | null;
  eval_metric_value?: number | null;
  eval_delta?: number | null;
};

export type RunUserSummary = {
  headline?: string;
  why_it_matters?: string;
  next_step?: string;
  /** "success" | "warning" | "danger" | "neutral" — the tone of the verdict. */
  confidence_tone?: string;
};

/**
 * Run cost rollup (Track P2). Energy + dollar estimate computed from
 * wall-clock duration × backend nominal power. `source: "measured"` means
 * a real per-run sample fed into the estimator; "nominal" means the
 * backend nominal-power table. The UI renders the source as an "estimate"
 * badge so users know the difference.
 */
export type RunCost = {
  duration_seconds: number;
  duration_hours: number;
  power_watts_estimated: number;
  energy_kwh: number;
  cost_usd: number;
  cost_per_kwh: number;
  backend: string;
  source: "measured" | "nominal" | string;
};

export type RunDetail = {
  id: string;
  run_id: string;
  modality: string;
  model_name: string;
  status: string;
  timestamp?: string | null;
  headline?: string;
  next_step?: string;
  top_issue?: string | null;
  user_summary?: RunUserSummary;
  metrics_summary?: RunMetricsSummary;
  recovery?: Record<string, unknown>;
  details?: {
    cycles_executed?: number;
    seed?: number;
    resume_from_cycle?: number;
    final_model_available?: boolean;
    cycle_metrics?: CycleMetric[];
    cycle_losses?: number[];
    yield_diagnostics?: Record<string, unknown>;
    cost?: RunCost;
    [k: string]: unknown;
  };
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

export type RunLogs = {
  available: boolean;
  lines: string[];
  reason: string | null;
  log_path: string | null;
  tail: number;
  total_lines_returned?: number;
};

export type RunSample = {
  prompt?: string;
  completion?: string;
  reward?: number;
  success?: boolean;
  details?: Record<string, unknown>;
  /** mlx-flavored chat-format records may surface this instead of prompt+completion */
  messages?: Array<{ role: string; content: string }>;
  [k: string]: unknown;
};

export type RunSamples = {
  available: boolean;
  samples: RunSample[];
  reason: string | null;
  cycle: number | null;
  kind: "samples" | "accepted" | string;
  available_cycles?: number[];
  limit?: number;
  total_returned?: number;
  source_path?: string;
};

export const api = {
  health: () => request<{ ok: boolean }>("/health"),
  backendInfo: () => request<BackendInfo>("/backend"),
  telemetry: () => request<TelemetrySample>("/telemetry"),
  dashboard: () => request<DashboardSummary>("/dashboard"),
  runCancel: (runId: string) =>
    request<{ ok: boolean; reason: string | null; run_id: string; status: string | null }>(
      `/runs/${encodeURIComponent(runId)}/cancel`,
      { method: "POST" },
    ),
  runLogs: (runId: string, tail = 200) =>
    request<RunLogs>(`/runs/${encodeURIComponent(runId)}/logs?tail=${tail}`),
  runSamples: (
    runId: string,
    params: { cycle?: number; kind?: "samples" | "accepted"; limit?: number } = {},
  ) => {
    const search = new URLSearchParams();
    if (params.cycle !== undefined) search.set("cycle", String(params.cycle));
    if (params.kind) search.set("kind", params.kind);
    if (params.limit) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<RunSamples>(
      `/runs/${encodeURIComponent(runId)}/samples${qs ? `?${qs}` : ""}`,
    );
  },
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
  /**
   * DB-backed run search (Track F-G commit 2). Filter / sort / paginate
   * the SQLite-backed run index. Repeating `modality` or `status` in the
   * params object becomes an IN-list. The response carries `facets` —
   * the distinct values currently indexed — so filter-chip UIs can
   * render their chip set without a second request.
   */
  searchRuns: (params?: RunSearchParams) => {
    const search = new URLSearchParams();
    for (const m of params?.modality ?? []) search.append("modality", m);
    for (const s of params?.status ?? []) search.append("status", s);
    if (params?.model) search.set("model", params.model);
    if (params?.since) search.set("since", params.since);
    if (params?.until) search.set("until", params.until);
    if (params?.hasEval !== undefined) search.set("has_eval", String(params.hasEval));
    if (params?.weightsUpdated !== undefined)
      search.set("weights_updated", String(params.weightsUpdated));
    if (params?.sortBy) search.set("sort_by", params.sortBy);
    if (params?.sortDir) search.set("sort_dir", params.sortDir);
    if (params?.limit !== undefined) search.set("limit", String(params.limit));
    if (params?.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<RunSearchResponse>(`/runs/search${qs ? `?${qs}` : ""}`);
  },
  runDetail: (runId: string) => request<RunDetail>(`/runs/${encodeURIComponent(runId)}`),
  /**
   * Cohort eval (F-K): runs × tasks grid pulled from each run's
   * `lm_eval_summary.json`. Missing-eval runs return `available: false`
   * so the dashboard renders em-dashes instead of failing.
   */
  evalCohort: (runIds: string[]) => {
    const search = new URLSearchParams();
    for (const id of runIds) search.append("run_ids", id);
    return request<EvalCohortResponse>(`/eval/cohort?${search.toString()}`);
  },
  runEval: (runId: string) =>
    request<RunEvalResponse>(`/runs/${encodeURIComponent(runId)}/eval`),
};
