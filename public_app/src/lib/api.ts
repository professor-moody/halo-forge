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
export const AUTH_REQUIRED_EVENT = "halo-forge:auth-required";

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

export function isLoopbackHost(hostname: string | undefined = window.location.hostname): boolean {
  const h = (hostname || "").trim().toLowerCase();
  return h === "localhost" || h === "::1" || h === "[::1]" || h.startsWith("127.") || h === "";
}

export function connectionMode(): "local" | "remote" {
  return isLoopbackHost() ? "local" : "remote";
}

export function reportAuthRequired(detail?: unknown): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT, { detail }));
}

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

export function isAuthRequiredError(error: unknown): boolean {
  if (!(error instanceof ApiError) || error.status !== 401) return false;
  const payload = error.payload;
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (detail && typeof detail === "object" && "error" in detail) {
      return (detail as { error?: unknown }).error === "invalid_token";
    }
  }
  return true;
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
    if (res.status === 401) {
      reportAuthRequired(payload);
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
  mps_to_cpu_fallbacks_60s: number | null;
  chip: {
    generation: number;
    variant: "base" | "Pro" | "Max" | "Ultra" | null;
    gpu_cores: number | null;
    nominal_memory_bandwidth_gbps: number | null;
    brand: string;
  } | null;
  note: string | null;
};

export type BackendInfo = {
  name: string;
  device: string;
  chip: TelemetrySample["chip"];
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
    supports_neural_accelerators: boolean;
  };
  training_defaults: Record<string, unknown>;
  inference_defaults: Record<string, unknown>;
  mlx_readiness: MLXReadiness;
};

export type WorkspaceInfo = {
  default_run_root: string;
  runs_dir: string;
  writable: boolean;
  message: string;
};

export type MLXReadiness = {
  status: "ready" | "unavailable" | "error";
  executable: boolean;
  package_versions: Record<string, string | null>;
  chip: (TelemetrySample["chip"] & { raw_brand?: string | null }) | { brand?: string | null; raw_brand?: string | null } | null;
  macos_version: string | null;
  metal_device: Record<string, unknown> | null;
  errors: string[];
  warnings: string[];
  suggested_fixes: string[];
  probe: Record<string, unknown>;
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
  timestamp?: string | null;
  output_dir?: string | null;
  cycles_executed?: number;
  weights_updated?: boolean;
  final_model_available?: boolean;
  artifact_path?: string | null;
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
    /** Per-modality run counts. Modalities with zero runs are omitted. */
    modality_counts?: Record<string, number>;
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
 * Model registry entry (Track F-J). A named bundle of run_ids the
 * user wants to compare / promote / share as a unit.
 */
export type RegistryEntry = {
  id: number;
  name: string;
  description: string | null;
  base_model: string | null;
  run_ids: string[];
  tags: string[];
  created_at: string;
  updated_at: string;
};

export type RegistryEntryCreate = {
  name: string;
  description?: string | null;
  base_model?: string | null;
  run_ids?: string[];
  tags?: string[];
};

export type RegistryEntryPatch = Partial<{
  description: string | null;
  base_model: string | null;
  run_ids: string[];
  tags: string[];
}>;

/**
 * Playground chat (Track F-S). Forwarded to a `halo-forge serve`
 * endpoint via the public API proxy.
 */
export type PlaygroundMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type PlaygroundChatRequest = {
  messages: PlaygroundMessage[];
  model?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  serve_url?: string;
  api_key?: string;
};

export type PlaygroundChatChoice = {
  index: number;
  message: PlaygroundMessage;
  finish_reason?: string;
};

/**
 * Run lineage edge (Track F-Q). One row of the runs.run_lineage table
 * tagged with depth from the queried run.
 */
export type LineageEdge = {
  parent_run_id?: string;
  child_run_id?: string;
  forked_at_cycle: number | null;
  notes: string | null;
  depth: number;
};

export type RunLineage = {
  run_id: string;
  ancestors: LineageEdge[];
  descendants: LineageEdge[];
};

export type RecordForkPayload = {
  parent_run_id: string;
  forked_at_cycle?: number | null;
  notes?: string | null;
};

export type PlaygroundChatResponse = {
  id?: string;
  object?: string;
  created?: number;
  model?: string;
  choices?: PlaygroundChatChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  // Surfaces upstream errors so the UI can render them inline.
  upstream_error?: boolean;
  status?: number;
  detail?: unknown;
  message?: string;
  error_kind?: string;
  action?: string;
  model_id?: string;
  model_url?: string;
};

export type HuggingFaceStatus = {
  present: boolean;
  source: "env" | "keyring" | "file" | "none" | string;
  verified: boolean;
  username: string | null;
  status: "not_connected" | "connected" | "needs_attention" | string;
  message: string;
  can_clear: boolean;
};

export type HuggingFaceModelAccess = {
  model_id: string;
  status: "available" | "gated" | "auth_required" | "missing" | "network_error" | string;
  available: boolean;
  message: string;
  model_url?: string | null;
  license_url?: string | null;
  action?: string | null;
};

export type ServeStatus = {
  running: boolean;
  state: "idle" | "starting" | "running" | "unhealthy" | "exited" | string;
  active_action?: "loading_model" | "serving" | "check_logs" | "review_logs" | string | null;
  pid: number | null;
  model: string | null;
  backend: string | null;
  host: string;
  port: number;
  url: string;
  started_at: number | null;
  exit_code: number | null;
  log_path: string | null;
  logs_available: boolean;
  last_error: string | null;
  error_hint?: string | null;
  healthy: boolean;
  message: string;
};

export type ServeStartPayload = {
  model: string;
  backend?: string | null;
  host?: string;
  port?: number;
  trust_remote_code?: boolean;
};

export type ServeLogs = {
  available: boolean;
  lines: string[];
  path: string | null;
  reason?: string;
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

export type RunLive = {
  id: string;
  status: string;
  progress_percent: number | null;
  current_step: number | null;
  total_steps: number | null;
  current_epoch: number | null;
  total_epochs: number | null;
  current_cycle: number | null;
  total_cycles: number | null;
  latest_loss: number | null;
  latest_learning_rate: number | null;
  latest_grad_norm: number | null;
  headline?: string | null;
  next_step?: string | null;
  top_issue?: string | null;
  user_summary?: RunUserSummary;
  metrics_summary?: RunMetricsSummary;
  primary_action?: Record<string, unknown> | null;
  research_sections?: unknown[];
};

export type TrainingDataset = {
  key: string;
  huggingface_id: string;
  description: string;
  domain: "code" | "vlm" | "audio" | "reasoning" | "agentic" | string;
  size_hint: string;
  default_split: string;
};

export type TrainingMode =
  | "sft"
  | "raft"
  | "dpo"
  | "orpo"
  | "rm"
  | "grpo"
  | "vlm"
  | "audio"
  | "reasoning"
  | "agentic";

export type TrainingLaunchPayload =
  | ({ mode: "sft"; model: string; dataset: string; output_dir: string } & Record<string, unknown>)
  | ({ mode: "raft"; model: string; prompts: string; output_dir: string } & Record<string, unknown>)
  | ({ mode: "dpo" | "orpo" | "rm" | "grpo"; model: string; dataset: string; output_dir: string } & Record<string, unknown>)
  | ({ mode: "vlm" | "audio" | "reasoning" | "agentic"; model: string; dataset: string; output_dir: string } & Record<string, unknown>);

export type TrainingVerifier = {
  key: string;
  label: string;
  toolchain: string;
  modality: string;
  platforms: string[];
};

export type TrainingPreflight = {
  mode: string;
  ok: boolean;
  resolved_paths: Record<string, string>;
  errors: string[];
  warnings: string[];
  suggested_fixes: string[];
  user_summary?: {
    headline?: string;
    why_it_matters?: string;
    next_step?: string;
    confidence_tone?: string;
  };
  details?: Record<string, unknown>;
};

export type VerifierCatalogEntry = {
  name: string;
  cls: string;
  origin: "builtin" | "user_plugin" | "entry_point";
  module: string;
  doc: string | null;
  base: string;
};

export type VerifierCatalog = {
  items: VerifierCatalogEntry[];
  counts: Record<string, number>;
  plugin_dir: string;
  total: number;
};

export type DiagnosticsLaunch = {
  output_dir: string;
  status: "completed" | "orphan";
  has_summary: boolean;
  launched_at: string | null;
  command: string[] | null;
  args: Record<string, unknown>;
  log_files: string[];
  summary_mtime: number | null;
  launch_mtime: number | null;
};

export type DiagnosticsLogFile = {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
};

export type DiagnosticsSummary = {
  base_path: string;
  launches: {
    total: number;
    orphan: number;
    completed: number;
    most_recent_orphan: DiagnosticsLaunch | null;
  };
  logs: {
    total: number;
    newest: DiagnosticsLogFile | null;
  };
};

export type TrainingTemplate = {
  id: string;
  name: string;
  category: string;
  intent: string;
  modality: string;
  model_hint: string;
  dataset_hint: string;
  verifier: string | null;
  hyperparams: Record<string, unknown>;
  expected_runtime: string;
  learn_more: string | null;
  cli_hint: string | null;
};

export type TrainingTemplateCategory = {
  id: string;
  label: string;
  description: string;
};

export type TrainingTemplateGallery = {
  categories: TrainingTemplateCategory[];
  items: TrainingTemplate[];
};

export type TrainingTemplateDetail = TrainingTemplate & {
  cli: string;
};

export type DiagnosticsLogTail = {
  available: boolean;
  lines: string[];
  reason: string | null;
  path: string;
  tail: number;
  truncated_head?: boolean;
  size_bytes?: number;
};

export type SuggestedModel = {
  id: string;
  label: string;
  provider: string;
  family: string;
  parameter_count: string;
  modalities: string[];
  tasks: string[];
  trainer_support: string[];
  backend_support: string[];
  memory_tier: string;
  recommended_use: string;
  known_caveats: string[];
  trust_remote_code_required: boolean;
  mlx_variant: string | null;
  status: string;
  recommended_first_run: boolean;
  estimated_memory_gb: number | null;
  license_note: string | null;
  download_note: string | null;
  fit_notes: string[];
  risk_level: "safe" | "caveated" | "experimental" | string;
  last_verified: string;
  catalog_version: string;
};

export type ModelCatalogEntry = SuggestedModel;

export type ModelCatalogResponse = {
  catalog_version: string;
  items: ModelCatalogEntry[];
  total: number;
  facets: Record<string, string[]>;
  filters: Record<string, string>;
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
  workspaceInfo: () => request<WorkspaceInfo>("/workspace"),
  huggingFaceStatus: () => request<HuggingFaceStatus>("/huggingface/status"),
  huggingFaceSaveToken: (token: string) =>
    request<HuggingFaceStatus>("/huggingface/token", {
      method: "POST",
      body: JSON.stringify({ token }),
    }),
  huggingFaceClearToken: () =>
    request<HuggingFaceStatus>("/huggingface/token", {
      method: "DELETE",
    }),
  huggingFaceCheckModel: (model_id: string) =>
    request<HuggingFaceModelAccess>("/huggingface/check-model", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    }),
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
  runLive: (runId: string) => request<RunLive>(`/runs/${encodeURIComponent(runId)}/live`),
  trainingPresets: () => request<{ items: TrainingPreset[] }>("/train/presets"),
  trainingDatasets: () => request<{ items: TrainingDataset[] }>("/train/datasets"),
  trainingVerifiers: () => request<{ items: TrainingVerifier[] }>("/train/verifiers"),
  trainingTemplates: () => request<TrainingTemplateGallery>("/train/templates"),
  trainingTemplate: (id: string) =>
    request<TrainingTemplateDetail>(`/train/templates/${encodeURIComponent(id)}`),
  modelCatalog: (params: Record<string, string | undefined> = {}) => {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value) search.set(key, value);
    });
    const qs = search.toString();
    return request<ModelCatalogResponse>(`/models${qs ? `?${qs}` : ""}`);
  },
  verifierCatalog: () => request<VerifierCatalog>("/verifiers"),
  diagnosticsSummary: () => request<DiagnosticsSummary>("/diagnostics/summary"),
  diagnosticsLaunches: () =>
    request<{ items: DiagnosticsLaunch[] }>("/diagnostics/launches"),
  diagnosticsLogs: () => request<{ items: DiagnosticsLogFile[] }>("/diagnostics/logs"),
  diagnosticsLogTail: (path: string, tail = 200) => {
    const qs = new URLSearchParams({ path, tail: String(tail) });
    return request<DiagnosticsLogTail>(`/diagnostics/log?${qs.toString()}`);
  },
  trainingModels: (params: { mode?: string; modality?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.mode) search.set("mode", params.mode);
    if (params.modality) search.set("modality", params.modality);
    const qs = search.toString();
    return request<{ items: SuggestedModel[] }>(`/train/models${qs ? `?${qs}` : ""}`);
  },
  trainingPreflight: (payload: Record<string, unknown>) =>
    request<TrainingPreflight>("/train/preflight", {
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

  // ----- run lineage (Track F-Q) -----------------------------------------
  /**
   * Walk the run lineage table BFS up + down for `runId`.
   * Returns `{run_id, ancestors, descendants}` with depth-tagged edges.
   */
  getRunLineage: (runId: string) =>
    request<RunLineage>(`/runs/${encodeURIComponent(runId)}/lineage`),
  /**
   * Record that `runId` (the child) forked from `parentRunId`.
   * Idempotent on the (child, parent) pair.
   */
  recordRunFork: (runId: string, payload: RecordForkPayload) =>
    request<RunLineage>(`/runs/${encodeURIComponent(runId)}/lineage`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  removeRunFork: (runId: string, parentRunId: string) =>
    request<{ deleted: boolean }>(
      `/runs/${encodeURIComponent(runId)}/lineage/${encodeURIComponent(parentRunId)}`,
      { method: "DELETE" },
    ),

  // ----- playground chat (Track F-S) -------------------------------------
  /**
   * Forward a chat request to a `halo-forge serve` endpoint via the
   * public API proxy. Avoids CORS by routing through the same origin
   * the rest of the app uses; lets the playground hit any
   * OpenAI-compatible serve URL (local, remote, hosted) under one
   * auth + origin model.
   */
  playgroundChat: (payload: PlaygroundChatRequest) =>
    request<PlaygroundChatResponse>(`/playground/chat`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ----- managed local serving ------------------------------------------
  serveStatus: () => request<ServeStatus>("/serve/status"),
  serveStart: (payload: ServeStartPayload) =>
    request<ServeStatus>("/serve/start", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  serveStop: () =>
    request<ServeStatus>("/serve/stop", {
      method: "POST",
    }),
  serveLogs: (tail = 200) =>
    request<ServeLogs>(`/serve/logs?tail=${encodeURIComponent(String(tail))}`),
  serveHealth: () => request<ServeStatus>("/serve/health"),

  // ----- model registry (Track F-J) -------------------------------------
  listRegistry: () =>
    request<{ items: RegistryEntry[] }>(`/registry`),
  getRegistryEntry: (id: number) =>
    request<RegistryEntry>(`/registry/${id}`),
  createRegistryEntry: (payload: RegistryEntryCreate) =>
    request<RegistryEntry>(`/registry`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateRegistryEntry: (id: number, payload: RegistryEntryPatch) =>
    request<RegistryEntry>(`/registry/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteRegistryEntry: (id: number) =>
    request<{ deleted: boolean; id: number }>(`/registry/${id}`, {
      method: "DELETE",
    }),
};
