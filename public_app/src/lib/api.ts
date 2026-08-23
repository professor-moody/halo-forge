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
        const rawDetail = (payload as { detail: unknown }).detail;
        detail = (
          rawDetail
          && typeof rawDetail === "object"
          && "message" in rawDetail
          && typeof (rawDetail as { message?: unknown }).message === "string"
        )
          ? String((rawDetail as { message: string }).message)
          : typeof rawDetail === "string"
            ? rawDetail
            : rawDetail == null
              ? detail
              : JSON.stringify(rawDetail);
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

export type VersionInfo = {
  package_version: string;
  display_version: string;
  release_channel: string;
  git_sha?: string;
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
  ready_state?: "idle" | "starting_server" | "server_ready" | "loading_model" | "chat_ready" | "failed" | string;
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
  model_ready?: boolean;
  adapter_loaded?: boolean;
  load_error?: {
    status?: number;
    error_kind?: string;
    message?: string;
    model?: string;
    model_id?: string;
    action?: string;
    model_url?: string;
    hint?: string;
    detail?: unknown;
  } | null;
  load_error_kind?: string | null;
  load_error_message?: string | null;
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

export type RunFailureSummary = {
  kind: string;
  headline: string;
  message: string;
  next_action: string;
  log_path?: string | null;
  log_tail?: string[];
  retry_route?: string | null;
  docs_url?: string | null;
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
  failure_summary?: RunFailureSummary | null;
  /** Immutable dataset bindings recorded when the run was launched. */
  datasets?: DatasetBinding[];
  /** Persistent evaluations attached to this run or one of its checkpoints. */
  evaluations?: Evaluation[];
  parent_run_id?: string | null;
  run_group_id?: string | null;
  checkpoint_policy_revision_id?: string | null;
  /** V8 reward-integrity binding resolved for verifier-guided training. */
  reward_integrity?: ResolvedRewardBinding | null;
  /** Latest published boundary audit, when training-signal capture was enabled. */
  latest_reward_audit?: RewardIntegrityAudit | null;
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
  stage?: {
    key: string;
    label: string;
    message: string;
    progress_percent: number | null;
    started_at?: string | null;
  } | null;
  last_event?: string | null;
  elapsed_seconds?: number | null;
  eta_seconds?: number | null;
  artifact_state?: "none" | "checkpoint" | "final_model" | "failed" | string | null;
  metric_points?: Array<{
    step: number;
    timestamp: string;
    train_loss?: number | null;
    eval_loss?: number | null;
    learning_rate?: number | null;
    grad_norm?: number | null;
    throughput?: number | null;
  }>;
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
  | "cpt"
  | "raft"
  | "dpo"
  | "orpo"
  | "rm"
  | "grpo"
  | "vlm"
  | "audio"
  | "reasoning"
  | "agentic"
  | "classify"
  | "embed"
  | "rerank";

export type TrainingLaunchPayload =
  | ({ mode: "sft"; model: string; dataset: string; output_dir: string } & Record<string, unknown>)
  | ({ mode: "raft"; model: string; prompts: string; output_dir: string } & Record<string, unknown>)
  | ({ mode: "dpo" | "orpo" | "rm" | "grpo"; model: string; dataset: string; output_dir: string } & Record<string, unknown>)
  | ({ mode: "vlm" | "audio" | "reasoning" | "agentic" | "classify" | "embed" | "rerank"; model: string; dataset: string; output_dir: string } & Record<string, unknown>);

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
  /** False while a managed Dataset Lab artifact is being rendered. */
  ready?: boolean;
  accepted?: boolean;
  status?: "preparing_dataset" | string;
  job_id?: string | null;
  run_id?: string | null;
  dataset_version_id?: string | null;
  dataset_bindings?: DatasetBinding[];
  artifact_preparation?: (TrainingDatasetArtifact & {
    job_id: string;
    job_url: string;
  }) | null;
  message?: string;
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

export type TrainingLaunchResult = Record<string, unknown> & {
  status?: "preparing_dataset" | string;
  ready?: boolean;
  accepted?: boolean;
  job_id?: string | null;
  run_id?: string | null;
  artifact_preparation?: (TrainingDatasetArtifact & {
    job_id: string;
    job_url: string;
  }) | null;
  message?: string;
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

// ----- Verifier Reliability and Reward Studio --------------------------

export type VerifierFamily = "deterministic" | "llm_judge" | "reward_model" | "chain";
export type VerifierQualificationState =
  | "pass"
  | "warn"
  | "fail"
  | "unqualified"
  | "legacy_unqualified"
  | "stale_runtime"
  | string;

export type VerifierRewardContract = {
  minimum: number;
  maximum: number;
  direction: "maximize" | "minimize";
  threshold?: number | null;
  tie_policy?: "pass" | "fail" | "tie" | "error" | string;
  probability_semantics?: boolean;
  error_behavior?: "fail_closed" | "fail_open" | "error" | "abstain" | string;
};

export type VerifierCapabilityDescriptor = {
  id: string;
  family: VerifierFamily;
  label: string;
  description?: string | null;
  implementation?: string | null;
  implementation_fingerprint?: string | null;
  origin?: "builtin" | "user_plugin" | "entry_point" | "artifact" | string;
  fingerprintable?: boolean;
  modalities: string[];
  task_types: string[];
  supports_probability?: boolean;
  supports_seed?: boolean;
  runtime_requirements?: Record<string, unknown>;
  compatible_consumers?: string[];
  reliability_adapter_id?: string;
  reliability_adapter_version?: string;
};

export type VerifierRevisionComponent = {
  id?: string;
  child_revision_id: string;
  ordinal: number;
  weight?: number;
  veto?: boolean;
  aggregation_rule?: string | null;
  child?: VerifierProfileRevision | null;
};

export type VerifierProfile = {
  id: string;
  name: string;
  description?: string | null;
  latest_revision_id?: string | null;
  latest_revision?: VerifierProfileRevision | null;
  revision_count?: number;
  created_at?: string | null;
  updated_at?: string | null;
};

export type VerifierProfileRevision = {
  id: string;
  profile_id: string;
  revision_number: number;
  family: VerifierFamily;
  modality: string;
  task_type: string;
  implementation_id?: string | null;
  implementation_fingerprint?: string | null;
  reliability_adapter_id?: string | null;
  reliability_adapter_version?: number | string | null;
  input_mapping?: Record<string, unknown>;
  output_contract?: Record<string, unknown>;
  reward_contract: VerifierRewardContract;
  rubric?: string | null;
  prompt_template?: string | null;
  parser?: Record<string, unknown> | string | null;
  model_revision?: string | null;
  tokenizer_revision?: string | null;
  artifact_id?: string | null;
  artifact_hash?: string | null;
  endpoint_type?: string | null;
  generation_settings?: Record<string, unknown>;
  runtime_requirements?: Record<string, unknown>;
  components?: VerifierRevisionComponent[];
  content_hash?: string;
  qualification_state?: VerifierQualificationState;
  alias?: "candidate" | "approved" | string | null;
  overridden?: boolean;
  runtime_compatible?: boolean;
  created_at?: string | null;
};

export type ResolvedVerifierBinding = {
  profile_revision_id?: string | null;
  profile_revision_hash?: string | null;
  adapter_id?: string | null;
  implementation_fingerprint?: string | null;
  sanitized_configuration_hash?: string | null;
  reward_contract?: VerifierRewardContract | null;
  qualification_scope?: Record<string, unknown> | null;
  runtime_compatibility?: VerifierRuntimeCompatibility | null;
  legacy_warning_state?: "legacy_unqualified" | null;
};

export type VerifierCalibrationProtocolRevision = {
  id: string;
  protocol_id?: string;
  revision_number?: number;
  name: string;
  family?: VerifierFamily | "all";
  repeats?: number;
  seeds?: number[];
  temperature?: number;
  top_p?: number;
  concurrency?: number;
  confirmation_requested?: boolean;
  confirmation_fraction?: number;
  partition_seed?: number;
  bootstrap_resamples?: number;
  bootstrap_seed?: number;
  perturbations?: string[];
  settings?: Record<string, unknown>;
  content_hash?: string;
  created_at?: string | null;
};

export type VerifierQualificationProfileRevision = {
  id: string;
  profile_id?: string;
  revision_number?: number;
  name: string;
  template?: "strict_oracle" | "human_aligned" | "exploratory" | string;
  task_type?: string | null;
  thresholds?: Record<string, unknown>;
  minimum_evidence?: Record<string, number>;
  promotable?: boolean;
  content_hash?: string;
  created_at?: string | null;
};

export type VerifierObservation = {
  reward?: number | null;
  passed?: boolean | null;
  parsed_value?: unknown;
  raw_output?: unknown;
  details?: Record<string, unknown>;
  component_trace?: Array<Record<string, unknown>>;
  latency_ms?: number | null;
  error?: string | null;
  runtime_identity?: Record<string, unknown>;
};

export type VerifierCalibration = {
  id: string;
  profile_revision_id: string;
  profile_revision?: VerifierProfileRevision | null;
  source_kind: "label_set_revision" | "benchmark_suite_revision" | string;
  source_revision_id: string;
  source_purpose?: string | null;
  source_hash?: string | null;
  source_name?: string | null;
  protocol_revision_id: string;
  qualification_profile_revision_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" | string;
  stage?: string | null;
  processed_records?: number;
  total_records?: number;
  progress_percent?: number;
  primary_metric?: VerifierCalibrationMetric | null;
  qualification?: VerifierQualificationDecision | null;
  decisions?: VerifierQualificationDecision[];
  metrics?: VerifierCalibrationMetric[];
  work_item_id?: string | null;
  evidence_hash?: string | null;
  runtime_hash?: string | null;
  request_hash?: string | null;
  runtime_compatibility?: Record<string, unknown> | null;
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type VerifierCalibrationSample = {
  id: string;
  calibration_id: string;
  record_id: string;
  record_hash?: string | null;
  group_id?: string | null;
  split?: "calibration" | "confirmation" | string;
  task_type?: string;
  orientation?: string | null;
  perturbation?: string | null;
  repeat_index?: number;
  seed?: number | null;
  expected?: unknown;
  input?: unknown;
  observation: VerifierObservation;
  agreement?: boolean | null;
  subgroup?: Record<string, string>;
};

export type VerifierCalibrationMetric = {
  id?: string;
  calibration_id?: string;
  name: string;
  value: number | null;
  direction?: "maximize" | "minimize";
  lower_ci?: number | null;
  upper_ci?: number | null;
  record_count?: number;
  subgroup?: string | null;
  split?: string | null;
  available?: boolean;
  reason?: string | null;
  details?: Record<string, unknown>;
};

export type VerifierQualificationDecision = {
  id: string;
  calibration_id: string;
  profile_revision_id: string;
  qualification_profile_revision_id: string;
  decision: "pass" | "warn" | "fail";
  scope?: "development" | "operational" | "confirmation" | string;
  reasons: string[];
  evidence_count?: number;
  metrics?: VerifierCalibrationMetric[];
  override?: boolean;
  override_note?: string | null;
  created_at?: string | null;
};

export type VerifierAlias = {
  id: string;
  profile_id: string;
  alias: "candidate" | "approved" | string;
  profile_revision_id: string;
  previous_revision_id?: string | null;
  note?: string | null;
  override?: boolean;
  created_at?: string | null;
};

export type VerifierCalibrationComparison = {
  base_calibration_id: string;
  candidate_calibration_id: string;
  compatible: boolean;
  compatibility_reasons?: string[];
  task_type?: string;
  metrics: Array<{
    name: string;
    base_value: number | null;
    candidate_value: number | null;
    raw_delta: number | null;
    favorable_delta?: number | null;
    direction?: "maximize" | "minimize";
  }>;
  sample_counts?: Record<string, number>;
};

export type VerifierRuntimeCompatibility = {
  profile_revision_id: string;
  status: "compatible" | "stale_runtime" | "missing" | string;
  compatible: boolean;
  expected?: Record<string, unknown>;
  observed?: Record<string, unknown>;
  differences?: Array<{ field: string; expected: unknown; observed: unknown }>;
  checked_at?: string | null;
};

export type VerifierUsage = {
  items: Array<{
    id: string;
    kind: "dataset" | "run" | "evaluation" | "suggestion" | "evidence_bundle" | string;
    role?: string | null;
    label?: string | null;
    created_at?: string | null;
  }>;
  total: number;
};

// ----- Reward Integrity and Training Signal Studio --------------------

export type TrainingSignalCaptureFidelity =
  | "exact"
  | "sampled"
  | "aggregate_only"
  | "unavailable"
  | string;

export type TrainingSignalCapabilityDescriptor = {
  id: string;
  version: number | string;
  trainer_mode: TrainingMode | string;
  backend_family: string;
  boundary_unit: "step" | "cycle" | "epoch" | "full_trial" | string;
  resumable: boolean;
  audit_boundaries?: string[];
  capture_fidelity: TrainingSignalCaptureFidelity;
  candidate_multiplicity?: "single" | "multiple" | string;
  mappings?: {
    identity?: string[];
    input?: string[];
    output?: string[];
    reference?: string[];
    media?: string[];
    verifier?: string[];
  };
  unavailable_fields?: string[];
  reason?: string | null;
};

export type TrainingRecordRef = {
  record_id: string;
  record_hash?: string | null;
  instance_id?: string | null;
  identity_kind?: "managed" | "virtual" | "manual" | string;
  dataset_version_id?: string | null;
  split?: string | null;
  group_id?: string | null;
};

export type TrainingSignalSnapshot = {
  id: string;
  snapshot_id?: string;
  shard_id: string;
  run_id: string;
  segment_id?: string | null;
  boundary_index: number;
  boundary_value?: number | null;
  record: TrainingRecordRef;
  candidate_ordinal?: number;
  occurrence_id?: string | null;
  identity_mode?: "trainer_occurrence" | "legacy_content_fallback" | string;
  prompt?: unknown;
  context?: unknown;
  output?: unknown;
  expected?: unknown;
  media?: Array<{ kind: string; hash: string; path?: string | null; metadata?: Record<string, unknown> }>;
  generation_settings?: Record<string, unknown>;
  training_observation?: VerifierObservation | null;
  selection?: "kept" | "dropped" | "diagnostic" | string;
  selection_reason?: string | null;
  capture_stratum?: "uniform_core" | "diagnostic" | string;
  producer_model_hash?: string | null;
  producer_model_identity?: Record<string, unknown>;
  checkpoint_hash?: string | null;
  runtime_identity?: Record<string, unknown>;
  created_at?: string | null;
};

export type TrainingSignalShard = {
  id: string;
  run_id: string;
  segment_id?: string | null;
  direct_run_segment_id?: string | null;
  trial_segment_id?: string | null;
  reward_system_revision_id?: string;
  protocol_revision_id?: string;
  capability_id?: string;
  boundary_index: number;
  boundary_value?: number | null;
  boundary_unit?: string | null;
  status: "open" | "sealed" | "quarantined" | "corrupt" | string;
  capture_fidelity: TrainingSignalCaptureFidelity;
  observed_count: number;
  retained_count: number;
  core_count?: number;
  diagnostic_count?: number;
  aggregate_statistics?: Record<string, number | null>;
  dataset_identity?: Record<string, unknown>;
  producer_model_hash?: string | null;
  checkpoint_hash?: string | null;
  runtime_identity?: Record<string, unknown>;
  retained_set_hash?: string | null;
  trace_hash?: string | null;
  manifest_path?: string | null;
  sealed_at?: string | null;
};

export type RewardSystemAuditor = {
  id?: string;
  reward_system_revision_id?: string;
  verifier_profile_revision_id?: string;
  /** Canonical v11 wire name; the profile-prefixed alias remains readable. */
  verifier_revision_id?: string;
  role: "primary_sentinel" | "diagnostic" | string;
  ordinal: number;
  implementation_fingerprint?: string | null;
  verifier_chain_leaf_fingerprints?: string[];
  correlated?: boolean;
  correlation_reasons?: string[];
};

export type RewardSystemRevision = {
  id: string;
  reward_system_id?: string;
  system_id?: string;
  revision_number: number;
  name?: string;
  optimizer_verifier_profile_revision_id?: string;
  optimizer_verifier_revision_id?: string;
  optimizer_verifier_hash?: string | null;
  modality: string;
  task_type: string;
  input_mapping?: Record<string, unknown>;
  reward_normalization?: Record<string, unknown>;
  reward_mapping?: Record<string, unknown>;
  threshold?: number | null;
  failure_behavior?: string | null;
  shaping?: Record<string, unknown>;
  definition?: Record<string, unknown>;
  compatible_capabilities?: string[];
  auditors: RewardSystemAuditor[];
  qualification_state?: "ready" | "correlated" | "stale_runtime" | "unqualified" | string;
  content_hash: string;
  created_at?: string | null;
};

export type RewardSystem = {
  id: string;
  name: string;
  description?: string | null;
  latest_revision_id?: string | null;
  latest_revision?: RewardSystemRevision | null;
  revision_count?: number;
  created_at?: string | null;
  updated_at?: string | null;
};

export type RewardAuditProtocolRevision = {
  id: string;
  protocol_id?: string;
  revision_number?: number;
  name: string;
  template: "balanced_256" | "broad_512" | "exhaustive" | string;
  uniform_core_limit?: number;
  diagnostic_limit?: number;
  seed?: number;
  boundaries?: Array<number | string>;
  capture_required_for_gating?: boolean;
  definition?: Record<string, unknown>;
  content_hash?: string;
  created_at?: string | null;
};

export type RewardIntegrityProfileRevision = {
  id: string;
  profile_id?: string;
  revision_number?: number;
  name: string;
  template: "strict_integrity" | "human_aligned_integrity" | "exploratory" | string;
  thresholds?: Record<string, { pass?: number; warn?: number; direction?: "min" | "max" }>;
  minimum_pass_records?: number;
  minimum_report_records?: number;
  bootstrap_resamples?: number;
  bootstrap_seed?: number;
  promotable?: boolean;
  requirements?: Record<string, unknown>;
  content_hash?: string;
  created_at?: string | null;
};

export type RewardIntegrityBinding = {
  reward_system_revision_id: string;
  reward_audit_protocol_revision_id: string;
  reward_integrity_profile_revision_id: string;
  audit_boundaries?: Array<number | string>;
  development_suite_revision_id?: string | null;
};

export type ResolvedRewardBinding = RewardIntegrityBinding & {
  reward_system_hash?: string | null;
  optimizer_verifier_profile_revision_id?: string | null;
  primary_sentinel_verifier_profile_revision_id?: string | null;
  capability?: TrainingSignalCapabilityDescriptor | null;
  capture_fidelity?: TrainingSignalCaptureFidelity;
  boundary_unit?: string | null;
  resolved_boundaries?: number[];
  ready?: boolean;
  warnings?: string[];
  errors?: string[];
};

export type RewardIntegrityMetric = {
  id?: string;
  audit_id?: string;
  name: string;
  value: number | null;
  direction?: "maximize" | "minimize";
  lower_ci?: number | null;
  upper_ci?: number | null;
  record_count?: number;
  population?: "uniform_core" | "diagnostic" | string;
  subgroup?: string | null;
  available?: boolean;
  reason?: string | null;
  details?: Record<string, unknown>;
};

export type RewardIntegrityDecision = {
  id: string;
  audit_id: string;
  decision: "pass" | "warn" | "fail" | "incomplete_evidence" | string;
  action?: "continue" | "awaiting_review" | "stop" | "fork" | string;
  reasons: string[];
  record_count?: number;
  automatic?: boolean;
  override?: boolean;
  review_action?: "continue" | "stop" | "fork" | string | null;
  review_reason?: string | null;
  reviewed_at?: string | null;
  evidence?: Record<string, unknown>;
  created_at?: string | null;
};

export type RewardIntegrityForkContext = {
  audit_id: string;
  decision: RewardIntegrityDecision;
  parent_run_id: string;
  checkpoint: {
    content_hash: string;
    path?: string | null;
    occurrence_id?: string | null;
    artifact?: Record<string, unknown> | null;
    snapshot_path?: string | null;
    boundary_unit?: string | null;
    boundary_value?: number | null;
    segment_id?: string | null;
    integrity_source?: string | null;
    blockers?: string[];
  };
  reward_system_revision_id: string;
  reward_audit_protocol_revision_id: string;
  reward_integrity_profile_revision_id: string;
  signal_capability?: Record<string, unknown> | null;
  resume_mode: "resume_boundary" | "initialize_from_checkpoint" | string;
  train_context: Record<string, unknown>;
  datasets?: Array<Record<string, unknown>>;
  launch_ready: boolean;
  blockers: string[];
  href: string;
  replay_sync?: Record<string, unknown>;
};

export type RewardIntegrityReviewResult = RewardIntegrityDecision | RewardIntegrityForkContext;

export type RewardIntegrityObservation = {
  id: string;
  audit_id: string;
  snapshot_id: string;
  record: TrainingRecordRef;
  boundary_index: number;
  candidate_ordinal?: number;
  prompt?: unknown;
  context?: unknown;
  output?: unknown;
  expected?: unknown;
  media?: TrainingSignalSnapshot["media"];
  optimizer_observation?: VerifierObservation | null;
  sentinel_observation?: VerifierObservation | null;
  diagnostic_observations?: Array<{ verifier_profile_revision_id: string; observation: VerifierObservation }>;
  normalized_optimizer_reward?: number | null;
  normalized_sentinel_reward?: number | null;
  reward_gap?: number | null;
  classification?: "agreement" | "optimizer_only_accept" | "sentinel_only_accept" | "both_reject" | "error" | string;
  capture_stratum?: string;
};

export type RewardIntegrityAudit = {
  id: string;
  run_id: string;
  segment_id?: string | null;
  boundary_index: number;
  boundary_value?: number | null;
  boundary_unit?: string | null;
  checkpoint_artifact_id?: string | null;
  training_signal_shard_id: string;
  signal_shard_id?: string;
  reward_system_revision_id: string;
  protocol_revision_id: string;
  integrity_profile_revision_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" | string;
  stage?: string | null;
  processed_records?: number;
  total_records?: number;
  distinct_record_count?: number;
  progress_percent?: number;
  capture_fidelity?: TrainingSignalCaptureFidelity;
  metrics?: RewardIntegrityMetric[];
  decision?: RewardIntegrityDecision | null;
  work_item_id?: string | null;
  evidence_hash?: string | null;
  trace_hash?: string | null;
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type RewardIntegrityComparison = {
  base_audit_id: string;
  candidate_audit_id: string;
  compatible: boolean;
  comparison_kind?: "paired_snapshot" | "matched_input" | "aggregate_only" | string;
  pairing_reason: string;
  compatibility_reasons?: string[];
  shared_record_count?: number;
  shared_snapshot_count?: number;
  unmatched_base: number;
  unmatched_candidate: number;
  pairs: RewardIntegrityComparisonPair[];
  pair_total: number;
  limit: number;
  offset: number;
  metrics: Array<{
    name: string;
    base_value: number | null;
    candidate_value: number | null;
    raw_delta: number | null;
    favorable_delta?: number | null;
    direction?: "maximize" | "minimize";
  }>;
};

export type RewardIntegrityComparisonPair = {
  id: string;
  pairing: "paired_snapshot" | "matched_input" | string;
  record_id: string;
  snapshot_id?: string | null;
  base_snapshot_id: string;
  candidate_snapshot_id: string;
  same_output: boolean;
  base: RewardIntegrityObservation;
  candidate: RewardIntegrityObservation;
};

export type EvaluationBatch = {
  id: string;
  suite_revision_id: string;
  verifier_profile_revision_id?: string | null;
  base: EvaluationSubject;
  candidates: EvaluationSubject[];
  evaluations?: Evaluation[];
  status?: string;
  work_item_ids?: string[];
  created_at?: string | null;
};

export type ReviewQueueSummary = ReviewQueue & {
  statistics: ReviewQueueStatistics;
  next_item_id?: string | null;
  next_item_ordinal?: number | null;
};

export type ReviewItemNeighbors = {
  item_id: string;
  position: number;
  total: number;
  previous_id?: string | null;
  next_id?: string | null;
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
  model_url: string | null;
  license_url: string | null;
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
 * Dataset Lab
 * ----------------------------------------------------------------------- */

export type DatasetSource = {
  id?: string;
  dataset_id?: string;
  kind: "local" | "huggingface" | string;
  uri: string;
  config?: string | null;
  split?: string | null;
  revision?: string | null;
  fingerprint?: string | null;
  row_count?: number | null;
  size_bytes?: number | null;
  created_at?: string | null;
  [key: string]: unknown;
};

export type DatasetCreatePayload = {
  name?: string;
  description?: string;
  modality?: string;
  canonical_schema?: string | Record<string, unknown>;
  source: Pick<DatasetSource, "kind" | "uri"> &
    Partial<Pick<DatasetSource, "config" | "split" | "revision">>;
};

export type DatasetRecipeStep = {
  id?: string;
  kind: string;
  label?: string;
  enabled?: boolean;
  [key: string]: unknown;
};

export type DatasetRecipe = {
  name?: string;
  kind?: string;
  schema?: string | Record<string, unknown>;
  seed?: number;
  steps: DatasetRecipeStep[];
};

export type DatasetVersion = {
  id: string;
  dataset_id: string;
  version?: number | string;
  label?: string | null;
  status: "queued" | "building" | "ready" | "failed" | "cancelled" | string;
  content_hash?: string | null;
  recipe_hash?: string | null;
  storage_path?: string | null;
  row_count?: number | null;
  size_bytes?: number | null;
  split_counts?: Record<string, number>;
  statistics?: Record<string, unknown>;
  rejections?: Record<string, unknown> | Array<Record<string, unknown>>;
  contamination?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  source_fingerprints?: Record<string, string> | Array<Record<string, unknown> | string>;
  assets_materialized?: boolean;
  recipe?: DatasetRecipe | Record<string, unknown> | null;
  compatible_trainers?: TrainerCompatibility[];
  training_artifacts?: TrainingDatasetArtifact[];
  created_at?: string | null;
  updated_at?: string | null;
  [key: string]: unknown;
};

export type DatasetJob = {
  id: string;
  dataset_id: string;
  version_id?: string | null;
  kind?: string;
  job_type?: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | string;
  stage?: string | null;
  progress?: number | null;
  progress_percent?: number | null;
  completed?: number | null;
  total?: number | null;
  processed_records?: number | null;
  total_records?: number | null;
  accepted_records?: number | null;
  rejected_records?: number | null;
  output_size_bytes?: number | null;
  logs?: string[];
  error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  [key: string]: unknown;
};

export type DatasetRecord = {
  id: string;
  name: string;
  description?: string | null;
  modality?: string | null;
  canonical_schema?: string | Record<string, unknown>;
  latest_version_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  sources: DatasetSource[];
  source?: DatasetSource | null;
  row_count?: number | null;
  size_bytes?: number | null;
  versions?: DatasetVersion[];
  jobs?: DatasetJob[];
  job?: DatasetJob | null;
  latest_version?: DatasetVersion | null;
  active_job?: DatasetJob | null;
  [key: string]: unknown;
};

export type DatasetPreview = {
  items: Array<Record<string, unknown>>;
  total: number;
  offset: number;
  limit: number;
  split?: string | null;
};

export type DatasetBuildPayload = {
  recipe: DatasetRecipe;
  source_id?: string;
};

export type DatasetJobAccepted = {
  id?: string;
  job_id?: string;
  status: string;
  dataset_id: string;
  version_id?: string | null;
};

/* -------------------------------------------------------------------------
 * Guided own-data training (Dataset Lab v9)
 * ----------------------------------------------------------------------- */

export type InterfaceCapabilityDescriptor = {
  id: string;
  label: string;
  kind: "execution_surface" | "training" | "source" | string;
  available: boolean;
  status?: "supported" | "alpha" | "preview" | "unavailable" | string;
  reason?: string | null;
  execution_surface?: "desktop" | "local_browser" | "remote_browser" | "cli" | string | null;
  modality?: string | null;
  canonical_shape?: string | null;
  trainer_mode?: TrainingMode | string | null;
  backends?: string[];
  model_families?: string[];
  metadata?: Record<string, unknown>;
};

export type TrainingScenarioField = {
  name: string;
  label?: string;
  description?: string | null;
  required?: boolean;
  value_type?: string;
  aliases?: string[];
  example?: unknown;
};

export type TrainingScenarioDescriptor = {
  id: string;
  revision_id: string;
  revision?: number;
  label: string;
  description: string;
  modality: "text" | "image" | "audio" | "preference" | "tool" | string;
  canonical_shape: "sft" | "chat" | "preference" | "rlvr" | "tool" | "vlm" | "audio" | string;
  task_type?: string | null;
  available: boolean;
  verified?: boolean;
  unavailable_reason?: string | null;
  required_fields: Array<string | TrainingScenarioField>;
  optional_fields?: Array<string | TrainingScenarioField>;
  accepted_aliases?: Record<string, string[]>;
  source_layouts?: string[];
  compatible_trainers?: TrainerCompatibility[];
  trainer_modes?: Array<TrainingMode | string>;
  model_families?: string[];
  recommended_model?: string | null;
  default_recipe?: DatasetRecipe;
  proof_run?: {
    max_samples: number;
    epochs?: number;
    cycles?: number;
    seed: number;
  };
  documentation_anchor?: string | null;
  common_failures?: string[];
  example_count?: number;
};

export type ScenarioAdviceRequest = {
  goal: string;
  modality?: string | null;
  source_fields?: string[];
  source_layout?: string | null;
  sample_values?: Record<string, unknown>;
  include_unavailable?: boolean;
};

export type ScenarioAdviceRecommendation = {
  scenario_id: string;
  scenario_revision_id: string;
  label: string;
  score: number;
  confidence: "high" | "medium" | "low" | string;
  why_fit: string[];
  cautions?: string[];
  required_fields?: string[];
  optional_fields?: string[];
  trainer_modes?: Array<TrainingMode | string>;
  expected_outcome?: string;
  available: boolean;
  unavailable_reason?: string | null;
  requires_confirmation: boolean;
};

export type ScenarioAdviceQuestion = {
  id: string;
  label: string;
  help?: string | null;
  options?: string[];
};

export type ScenarioAdviceResult = {
  registry_revision?: string;
  recommendations: ScenarioAdviceRecommendation[];
  questions?: ScenarioAdviceQuestion[];
  unavailable?: ScenarioAdviceRecommendation[];
  explanation?: string;
  requires_confirmation: boolean;
};

export type GuidedExampleDescriptor = {
  id: string;
  scenario_id: string;
  scenario_revision_id: string;
  label: string;
  description: string;
  expected_source_shape: string;
  expected_outcome: string;
  hardware_guidance: string;
  fixture_format: string;
  fixture_filename: string;
  record_count: number;
  modality: string;
  trainer_modes: Array<TrainingMode | string>;
  documentation_anchor?: string;
};

export type TrainingScenarioExample = {
  id: string;
  scenario_revision_id: string;
  label: string;
  description?: string | null;
  format: string;
  filename: string;
  records?: Array<Record<string, unknown>>;
  content?: string | null;
  size_bytes?: number | null;
  checksum?: string | null;
};

export type DatasetImportFile = {
  id: string;
  import_id: string;
  relative_path: string;
  size_bytes: number;
  uploaded_bytes: number;
  content_hash?: string | null;
  status: "pending" | "uploading" | "uploaded" | "verified" | "failed" | string;
  error?: string | null;
};

export type DatasetImportSession = {
  id: string;
  status: "draft" | "uploading" | "ready" | "inspecting" | "completed" | "failed" | "cancelled" | string;
  source_kind: "reference" | "upload" | "huggingface" | "example" | string;
  source_uri?: string | null;
  source_config?: string | null;
  source_split?: string | null;
  source_revision?: string | null;
  resolved_revision?: string | null;
  fingerprint?: string | null;
  scenario_revision_id?: string | null;
  files?: DatasetImportFile[];
  total_files?: number;
  total_bytes?: number;
  uploaded_bytes?: number;
  inspection_id?: string | null;
  work_item_id?: string | null;
  published_dataset_id?: string | null;
  published_source_id?: string | null;
  disk_forecast?: {
    ready: boolean;
    requires_override: boolean;
    blockers: string[];
    warnings: string[];
    remedy?: string | null;
    stages: Record<string, {
      phase: string;
      available: boolean;
      allowed: boolean;
      capacity_sufficient?: boolean | null;
      overridden: boolean;
      override_reason?: string | null;
      projected_disk_bytes: number;
      current_free_bytes?: number | null;
      projected_free_bytes?: number | null;
      required_reserve_bytes?: number | null;
      blockers: string[];
    }>;
    override_history?: Array<Record<string, unknown>>;
  };
  readiness?: {
    ready: boolean;
    requires_capacity_override: boolean;
    blockers: string[];
    warnings: string[];
    remedy?: string | null;
  };
  expires_at?: string | null;
  error?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type HuggingFaceDatasetOptions = {
  repo_id: string;
  requested_revision: string;
  resolved_revision: string;
  items: Array<{ config: string | null; splits: string[] }>;
  total: number;
  limit?: number;
  offset?: number;
};

export type InspectedSourceField = {
  name: string;
  value_type?: string;
  coverage: number;
  present_count?: number;
  null_count?: number;
  examples?: unknown[];
};

export type SchemaCandidate = {
  scenario_id: string;
  scenario_revision_id: string;
  label?: string;
  confidence: "high" | "medium" | "low" | "ambiguous" | string;
  score?: number;
  coverage?: number;
  required_coverage?: Record<string, number>;
  suggested_mapping?: Record<string, string | FieldMappingExpression>;
  safe_transforms?: string[];
  missing_fields?: string[];
  reasons?: string[];
};

export type DatasetSourceInspection = {
  id: string;
  import_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | string;
  stage?: string | null;
  progress_percent?: number | null;
  source_fingerprint?: string | null;
  row_count?: number | null;
  valid_records?: number | null;
  invalid_records?: number | null;
  size_bytes?: number | null;
  sample_count?: number;
  preview_policy?: string | null;
  preview_policy_details?: Record<string, unknown> | null;
  fields: InspectedSourceField[];
  preview_records: Array<Record<string, unknown>>;
  schema_candidates: SchemaCandidate[];
  parse_errors?: Array<Record<string, unknown>>;
  media_summary?: Record<string, unknown>;
  extraction_summary?: {
    document_count?: number;
    extracted?: number;
    failed?: number;
    quarantined?: number;
    empty?: number;
    encrypted?: number;
    image_only?: number;
    source_types?: Record<string, number>;
    [key: string]: unknown;
  };
  statistics?: {
    extraction_summary?: DatasetSourceInspection["extraction_summary"];
    media_summary?: Record<string, unknown>;
    [key: string]: unknown;
  };
  warnings?: string[];
  error?: string | null;
  work_item_id?: string | null;
};

export type FieldMappingExpression =
  | { kind: "direct"; source: string }
  | { kind: "constant"; value: unknown }
  | { kind: "concat"; sources: string[]; separator?: string }
  | { kind: "nested_path"; source: string; path: string }
  | { kind: "conversation"; source: string; role_field?: string; content_field?: string; role_map?: Record<string, string> }
  | { kind: "media_root"; source: string; root: string };

export type FieldMappingPlan = {
  version: 2;
  scenario_revision_id: string;
  mappings: Record<string, FieldMappingExpression>;
  confirmed: boolean;
};

export type MappingPreviewItem = {
  ordinal: number;
  source: Record<string, unknown>;
  canonical: Record<string, unknown>;
  issues: Array<{ field?: string; code?: string; message: string; severity?: string }>;
};

export type MappingPreview = {
  items: MappingPreviewItem[];
  total_sampled: number;
  valid_count: number;
  invalid_count: number;
  field_coverage?: Record<string, number>;
  ready: boolean;
  warnings?: string[];
};

export type SemanticRecordPreview = {
  kind: "sft" | "chat" | "preference" | "tool" | "vlm" | "audio" | "corpus" | string;
  ordinal: number;
  title: string;
  summary: string;
  source: Record<string, unknown>;
  canonical: Record<string, unknown>;
  presentation: {
    turns?: Array<{
      index?: number;
      role: string;
      content: string;
      tool_calls?: unknown[];
    }>;
    tools?: unknown[];
    expected_calls?: unknown[];
    expected_results?: unknown[];
    prompt?: unknown;
    chosen?: unknown;
    rejected?: unknown;
    system?: unknown;
    image?: unknown;
    response?: unknown;
    ground_truth?: unknown;
    alternatives?: unknown[];
    audio?: unknown;
    task?: unknown;
    transcript?: unknown;
    label?: unknown;
    title?: unknown;
    text?: unknown;
    source_ref?: unknown;
    source_spans?: unknown[];
    metadata?: Record<string, unknown>;
    reference_answer?: unknown;
    [key: string]: unknown;
  };
  issues?: Array<{ field?: string; code?: string; message: string; severity?: string }>;
  provenance?: Record<string, unknown>;
};

export type SemanticPreviewResponse = {
  items: SemanticRecordPreview[];
  total: number;
  limit: number;
  offset: number;
  canonical_schema: string;
  sampled: boolean;
};

export type DatasetPreparationPlan = {
  scenario_revision_id: string;
  mapping_plan: FieldMappingPlan;
  recipe: DatasetRecipe;
  sampled: boolean;
  estimates?: {
    accepted?: number;
    quarantined?: number;
    duplicates?: number;
    split_counts?: Record<string, number>;
    token_count?: number;
  };
  warnings?: string[];
  split_policy?: {
    method?: string;
    group_field?: string | null;
    ratios?: Record<string, number>;
    seed?: number;
    [key: string]: unknown;
  };
};

export type RemediationAction = {
  id: string;
  label: string;
  action: string;
  description: string;
  target?: string | null;
  payload?: Record<string, unknown>;
  destructive?: boolean;
};

export type ReadinessFinding = {
  code?: string;
  message: string;
  severity?: "error" | "warning" | "info" | string;
  action_id?: string;
  remedy?: string;
  action?: string;
  why_it_matters?: string;
};

export type DatasetReadiness = {
  ready: boolean;
  status?: string;
  scope?: "inspection" | "version" | string;
  subject_id?: string;
  scenario_revision_id?: string | null;
  sampled?: boolean;
  summary?: {
    headline?: string;
    detail?: string;
    next_step?: string;
    source_records?: number;
    preview_records?: number;
    valid_preview_records?: number;
    invalid_preview_records?: number;
    estimated_accepted_records?: number;
    estimated_quarantined_records?: number;
    exact_duplicate_preview_records?: number;
    token_count_is_estimated?: boolean;
    [key: string]: unknown;
  };
  blockers: ReadinessFinding[];
  warnings: ReadinessFinding[];
  actions?: RemediationAction[];
  rejected_examples?: Array<{
    ordinal?: number;
    source?: Record<string, unknown>;
    issues?: Array<Record<string, unknown>>;
  }>;
  distributions?: Record<string, unknown>;
  split_balance?: Record<string, {
    ratio?: number;
    estimated_records?: number;
    [key: string]: unknown;
  }>;
  media?: Record<string, unknown>;
  extraction?: Record<string, unknown>;
  minimum_data?: {
    required_for_default_split?: number;
    estimated_available?: number;
    satisfied?: boolean;
    scientific_quality_threshold?: number | null;
    note?: string;
    [key: string]: unknown;
  };
  compatible_trainers?: TrainerCompatibility[];
  recommended_model?: ModelCatalogEntry | null;
};

export type DocumentExtractorDescriptor = {
  id: string;
  label: string;
  version?: string;
  available: boolean;
  source_kinds?: string[];
  media_types?: string[];
  extensions?: string[];
  preserves?: string[];
  limitations?: string[];
  reason?: string | null;
  metadata?: Record<string, unknown>;
};

export type DocumentExtraction = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | string;
  source_fingerprint?: string | null;
  extractor_version?: string | null;
  config_hash?: string | null;
  content_hash?: string | null;
  bundle_path?: string | null;
  manifest_hash?: string | null;
  document_count?: number;
  item_count?: number;
  quarantined_count?: number;
  error?: string | null;
  work_item_id?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  summary?: Record<string, unknown>;
};

export type DocumentExtractionPreview = {
  extraction?: DocumentExtraction;
  items: Array<{
    id?: string;
    title?: string | null;
    text?: string;
    source_uri?: string | null;
    source_kind?: string | null;
    media_type?: string | null;
    ordinal?: number;
    provenance?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
    issues?: Array<Record<string, unknown>>;
  }>;
  total: number;
  limit?: number;
  offset?: number;
};

export type CorpusProfile = {
  document_count: number;
  character_count: number;
  paragraph_count: number;
  byte_count: number;
  language_hints?: Record<string, number>;
  length_distribution?: Record<string, unknown>;
  duplicate_documents?: number;
  quarantined_documents?: number;
  extraction_failures?: number;
  source_types?: Record<string, number>;
};

export type CorpusPackingRequest = {
  model: string;
  adaptation: "lora" | "full";
  max_sequence_length: number;
  packing: string;
  budget_mode: "tokens" | "passes";
  target_tokens?: number | null;
  corpus_passes?: number | null;
  effective_batch_size?: number;
  seed?: number;
};

export type CorpusPackingPlan = {
  tokenizer_id: string;
  tokenizer_revision?: string | null;
  tokenizer_hash?: string;
  max_sequence_length: number;
  separator: string;
  packing: string;
  budget_mode: "tokens" | "passes" | string;
  target_tokens?: number | null;
  corpus_passes?: number | null;
  train_tokens: number;
  validation_tokens: number;
  train_blocks: number;
  validation_blocks: number;
  padding_tokens: number;
  utilization: number;
  estimated_steps: number;
  effective_batch_size: number;
  artifact_hash?: string | null;
  warnings?: string[];
};

export type CorpusPackingPlanResponse = CorpusPackingPlan | {
  status: "preparing" | "queued" | "running" | string;
  ready?: boolean;
  job_id?: string | null;
  work_item_id?: string | null;
  message?: string | null;
  progress_percent?: number | null;
  packing_plan?: CorpusPackingPlan | null;
};

export type CorpusTrainingConfig = CorpusPackingRequest & {
  dataset_version_id: string;
  model_revision?: string | null;
  model_hash?: string | null;
  tokenizer_revision?: string | null;
  tokenizer_hash?: string | null;
  learning_rate?: number | null;
  output?: string | null;
};

export type DomainWorkResult<T> = {
  work_item_id?: string | null;
  import?: DatasetImportSession;
  inspection?: DatasetSourceInspection;
  extraction?: DocumentExtraction;
  dataset?: DatasetRecord;
  source?: DatasetSource;
  readiness?: DatasetReadiness;
  result?: T;
} & Partial<T>;

export type DatasetExportPayload = {
  output?: string;
  format?: "jsonl" | "parquet" | "csv" | string;
  split?: string;
};

/** A role-specific immutable dataset selection used by training and replay. */
export type DatasetBinding = {
  role: "train" | "validation" | "test" | "canary" | string;
  dataset_version_id: string;
  split: string;
  dataset_id?: string | null;
  dataset_name?: string | null;
  content_hash?: string | null;
  training_artifact_id?: string | null;
  artifact_hash?: string | null;
};

export type TrainerCompatibility = {
  adapter_id: string;
  adapter_version?: string;
  trainer_mode: TrainingMode | string;
  compatible: boolean;
  reason?: string | null;
  required_schema?: string | null;
};

export type TrainingDatasetArtifact = {
  id: string;
  dataset_version_id: string;
  status: "queued" | "rendering" | "ready" | "failed" | "cancelled" | string;
  stage?: string | null;
  progress_percent?: number | null;
  adapter_id: string;
  adapter_version: string;
  trainer_mode: TrainingMode | string;
  model?: string | null;
  tokenizer_revision?: string | null;
  chat_template_hash?: string | null;
  bindings: DatasetBinding[];
  paths?: Record<string, string>;
  asset_root?: string | null;
  row_counts?: Record<string, number>;
  token_statistics?: Record<string, unknown>;
  artifact_hash?: string | null;
  storage_path?: string | null;
  derived_validation?: boolean;
  created_at?: string | null;
  error?: string | null;
  /** Dataset artifact format v3 adds a consumable canonical-row lineage index. */
  format_version?: number;
  lineage_index_path?: string | null;
  lineage_index_hash?: string | null;
};

export type TrainingArtifactCreatePayload = {
  adapter_id: string;
  trainer_mode: TrainingMode | string;
  model?: string;
  tokenizer_revision?: string;
  chat_template?: string;
  bindings?: DatasetBinding[];
  validation_fraction?: number;
};

export type DatasetVersionComparison = {
  base_version_id: string;
  other_version_id: string;
  summary: Record<string, number | string | boolean | null>;
  added?: Array<Record<string, unknown>>;
  removed?: Array<Record<string, unknown>>;
  changed?: Array<Record<string, unknown>>;
  repeated?: Array<Record<string, unknown>>;
  split_moved?: Array<Record<string, unknown>>;
  recipe_diff?: Record<string, unknown>;
  statistics_diff?: Record<string, unknown>;
  source_contribution_diff?: Record<string, unknown>;
};

/* -------------------------------------------------------------------------
 * Benchmark suites and persistent evaluation
 * ----------------------------------------------------------------------- */

export type EvaluationSubject = {
  kind: "model" | "run" | "final_model" | "checkpoint" | string;
  value: string;
  run_id?: string | null;
  checkpoint?: string | null;
  revision?: string | null;
  subject_hash?: string | null;
};

export type BenchmarkSuiteItem = {
  id?: string;
  adapter: string;
  task?: string;
  dataset_version_id?: string | null;
  split?: string | null;
  config?: Record<string, unknown>;
  weight?: number;
};

export type BenchmarkSuiteRevision = {
  id: string;
  suite_id: string;
  revision: number | string;
  items: BenchmarkSuiteItem[];
  generation_settings: Record<string, unknown>;
  evaluator_versions?: Record<string, string>;
  primary_metric: string;
  direction: "maximize" | "minimize";
  content_hash?: string | null;
  created_at?: string | null;
};

export type BenchmarkSuite = {
  id: string;
  name: string;
  purpose?: "development" | "holdout" | "unspecified" | string;
  description?: string | null;
  latest_revision_id?: string | null;
  latest_revision?: BenchmarkSuiteRevision | null;
  revisions?: BenchmarkSuiteRevision[];
  revision_count?: number;
  created_at?: string | null;
  updated_at?: string | null;
};

export type BenchmarkSuiteCreatePayload = {
  name: string;
  description?: string;
  purpose?: "development" | "operational" | "holdout" | "unspecified";
  items: BenchmarkSuiteItem[];
  generation_settings?: Record<string, unknown>;
  primary_metric: string;
  direction: "maximize" | "minimize";
};

export type EvaluationMetric = {
  name: string;
  value: number | null;
  direction: "maximize" | "minimize";
  suite_item_id?: string | null;
  n_samples?: number | null;
  error?: string | null;
};

export type EvaluationSample = {
  id?: string;
  evaluation_id?: string;
  suite_item_id: string;
  record_id: string;
  input: unknown;
  expected: unknown;
  output: unknown;
  score: number | null;
  passed: boolean | null;
  latency_ms: number | null;
  error: string | null;
  verifier_trace: unknown;
  evidence_kind?: string;
  valid?: boolean;
  mineable?: boolean;
  generation_seed?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  finish_reason?: string | null;
  runtime_versions?: Record<string, unknown> | null;
  template_hash?: string | null;
  score_direction?: "maximize" | "minimize" | null;
  score_threshold?: number | null;
  coverage?: Record<string, unknown> | null;
  task?: string | null;
  category?: string | null;
  failure_reason?: string | null;
};

export type Evaluation = {
  id: string;
  suite_revision_id: string;
  suite_id?: string;
  suite_name?: string | null;
  subject: EvaluationSubject;
  subject_hash?: string | null;
  run_id?: string | null;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | string;
  stage?: string | null;
  progress_percent?: number | null;
  processed_samples?: number | null;
  total_samples?: number | null;
  metrics?: EvaluationMetric[];
  primary_metric?: EvaluationMetric | null;
  artifact_path?: string | null;
  reused_from_id?: string | null;
  error?: string | null;
  logs?: string[];
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
};

export type EvaluationHistoryItem = Evaluation & {
  history_ordinal: number;
  primary_value?: number | null;
};

export type EvaluationHistoryResponse = {
  items: EvaluationHistoryItem[];
  total: number;
  subject_ref?: string | null;
  suite_revision_id?: string | null;
  limit: number;
};

export type EvaluationCreatePayload = {
  suite_revision_id: string;
  subject: EvaluationSubject;
  reuse_completed?: boolean;
};

export type EvaluationSampleDelta = {
  record_id: string;
  suite_item_id: string;
  classification: "regression" | "improvement" | "unchanged_failure" | "unchanged_pass" | string;
  base?: EvaluationSample | null;
  candidate?: EvaluationSample | null;
  delta?: number | null;
};

export type EvaluationComparison = {
  base_id: string;
  candidate_id: string;
  suite_revision_id: string;
  primary_metric: string;
  direction: "maximize" | "minimize";
  base_value?: number | null;
  candidate_value?: number | null;
  delta?: number | null;
  counts: Record<string, number>;
  metrics?: Array<{
    name: string;
    direction: "maximize" | "minimize";
    base: number | null;
    candidate: number | null;
    delta: number | null;
  }>;
  samples: EvaluationSampleDelta[];
  suite_purpose?: "development" | "holdout" | "unspecified" | string;
  failure_mining_allowed?: boolean;
  evidence_gaps?: Array<Record<string, unknown>>;
  evidence_summary?: Record<string, number | boolean>;
  sample_total?: number;
  evidence_gap_total?: number;
  offset?: number;
  limit?: number;
};

export type EvaluationDrift = EvaluationComparison & {
  classification: "improved" | "regressed" | "practically_equivalent" | "unavailable" | string;
  practical_delta: number;
  compatible: boolean;
  history_contract: {
    suite_revision_id: string;
    direction: "maximize" | "minimize";
    comparison: string;
  };
};

export type FailureMiningSelector = {
  kind:
    | "candidate_failure"
    | "regression"
    | "improvement"
    | "verifier_disagreement"
    | string;
  task?: string;
  category?: string;
  failure_reason?: string;
  min_score?: number;
  max_score?: number;
};

export type FailureMiningPreview = {
  items: EvaluationSampleDelta[];
  total: number;
  selector: FailureMiningSelector;
  exclusions_hash?: string | null;
};

/* -------------------------------------------------------------------------
 * Reproducible experiment operations
 * ----------------------------------------------------------------------- */

export type WorkItem = {
  id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" | string;
  priority: number;
  stage?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
  run_group_id?: string | null;
  trial_id?: string | null;
  run_id?: string | null;
  attempt?: number;
  max_attempts?: number;
  error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  heartbeat_at?: string | null;
  payload?: Record<string, unknown>;
};

/** Durable scheduler worker. Fields are intentionally nullable because a
 * dashboard can reconnect while an older worker is still being reconciled. */
export type Worker = {
  id: string;
  name?: string | null;
  status: "online" | "offline" | "draining" | "needs_reconciliation" | string;
  pid?: number | null;
  process_start_identity?: string | null;
  started_at?: string | null;
  heartbeat_at?: string | null;
  current_work_item_id?: string | null;
  capabilities?: string[] | Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

export type WorkAttempt = {
  id: string;
  work_item_id: string;
  attempt: number;
  status: string;
  output_dir?: string | null;
  worker_id?: string | null;
  pid?: number | null;
  process_start_identity?: string | null;
  started_at?: string | null;
  heartbeat_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  retry_reason?: string | null;
  telemetry_rollup?: Record<string, number | null>;
};

export type WorkEvent = {
  id: string;
  work_item_id: string;
  attempt_id?: string | null;
  sequence?: number;
  type: string;
  status?: string | null;
  message?: string | null;
  payload?: Record<string, unknown>;
  created_at?: string | null;
};

export type ActivityItem = {
  id: string;
  work_item_id?: string | null;
  domain_id?: string | null;
  domain_type?: string | null;
  kind: string;
  title?: string | null;
  status: string;
  stage?: string | null;
  priority?: number;
  progress_current?: number | null;
  progress_total?: number | null;
  progress_percent?: number | null;
  queue_position?: number | null;
  eta_seconds?: number | null;
  blockers?: string[];
  resource_requirements?: Record<string, unknown>;
  worker_id?: string | null;
  attempt?: number;
  max_attempts?: number;
  attempts?: WorkAttempt[];
  events?: WorkEvent[];
  logs?: string[];
  error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  heartbeat_at?: string | null;
  telemetry_rollup?: Record<string, number | null>;
  summary_metrics?: Record<string, number | null>;
  next_actions?: string[];
  action_links?: Array<{
    id: string;
    label: string;
    href: string;
  }>;
};

export type TelemetrySeries = {
  work_item_id: string;
  interval_seconds?: number;
  retained_until?: string | null;
  samples: Array<{
    timestamp: string | number;
    cpu_util_percent?: number | null;
    gpu_util_percent?: number | null;
    process_memory_bytes?: number | null;
    system_memory_bytes?: number | null;
    device_memory_bytes?: number | null;
    disk_free_bytes?: number | null;
  }>;
  aggregate?: Record<string, number | null>;
};

export type ExperimentTrialRun = {
  id: string;
  trial_id: string;
  run_id?: string | null;
  seed: number;
  status: string;
  objective_value?: number | null;
  evaluation_id?: string | null;
  model_artifact_id?: string | null;
  segment_count?: number;
  error?: string | null;
};

export type ExperimentTrial = {
  id: string;
  run_group_id: string;
  ordinal: number;
  status: string;
  parameters: Record<string, unknown>;
  config_hash?: string | null;
  aggregate?: {
    count?: number;
    mean?: number | null;
    stddev?: number | null;
    minimum?: number | null;
    maximum?: number | null;
    median?: number | null;
    direction?: "maximize" | "minimize";
  } | null;
  pruned?: boolean;
  prune_reason?: string | null;
  runs?: ExperimentTrialRun[];
};

export type CheckpointSchedule = {
  kind?: "final_only" | "percentages" | "interval" | "explicit" | string;
  mode?: "final" | "percentages" | "interval" | "explicit" | string;
  unit?: "step" | "cycle" | "epoch" | string;
  percentages?: number[];
  boundaries?: number[];
  interval?: number | null;
  [key: string]: unknown;
};

export type CheckpointRetentionPolicy = {
  keep_last: number;
  keep_every_n_boundaries: number | null;
  keep_best: number;
  protect_evaluated: boolean;
  protect_decision_referenced: boolean;
  protect_lineage_referenced: boolean;
  review_before_cleanup: boolean;
};

export type CheckpointPolicyRule = {
  kind: "guardrail" | "plateau" | "threshold" | string;
  metric?: string | null;
  direction?: "maximize" | "minimize" | string | null;
  threshold?: number | null;
  practical_delta?: number | null;
  minimum_delta?: number | null;
  patience?: number | null;
  action?: "continue" | "pause" | "stop" | string;
  on_breach?: "pause" | "stop" | string;
  comparison?: "absolute" | "baseline" | "previous" | "best" | string;
  suite_revision_id?: string | null;
  required?: boolean;
  [key: string]: unknown;
};

export type CheckpointPolicyRevision = {
  id?: string;
  policy_id: string;
  revision_number: number;
  name: string;
  description?: string | null;
  development_suite_revision_id: string;
  primary_metric: string;
  direction: "maximize" | "minimize";
  schedule: CheckpointSchedule;
  rules: CheckpointPolicyRule[];
  retention?: CheckpointRetentionPolicy;
  guardrail_suite_revision_ids?: string[];
  automatic_actions: boolean;
  compatible_capabilities?: string[];
  version?: number;
  content_hash: string;
  created_at?: string | null;
};

export type ResolvedCheckpointPlan = {
  policy_revision_id: string;
  policy_hash: string;
  trainer_mode: string;
  unit: string;
  total_budget: number;
  boundaries: number[];
  required_suite_revision_ids: string[];
  automatic_actions: boolean;
  capability_notes?: string[];
  estimated_checkpoint_count?: number;
  estimated_evaluation_count?: number;
  estimated_storage_bytes?: number | null;
  content_hash: string;
};

export type TrainerExecutionCapability = {
  capability_id: string;
  version: number;
  trainer_mode: string;
  backend_family: string;
  segment_unit: "step" | "cycle" | "full_trial" | string;
  supports_gated_execution: boolean;
  resume_parameter?: string | null;
  resume_cli_flag?: string | null;
  checkpoint_pattern?: string | null;
  checkpoint_index?: string | null;
  reason?: string | null;
};

export type CheckpointGateDecision = {
  id: string;
  run_group_id?: string | null;
  run_id?: string | null;
  checkpoint_artifact_id?: string | null;
  policy_revision_id: string;
  plan_hash: string;
  boundary_index: number;
  boundary_value?: number | null;
  action: "continue" | "pause" | "stop" | "await_review" | string;
  status?: "decided" | "awaiting_review" | "overridden" | string;
  automatic: boolean;
  reasons: string[];
  evidence?: Record<string, unknown>;
  review_reason?: string | null;
  reviewed_at?: string | null;
  content_hash: string;
  created_at?: string | null;
};

export type CheckpointTrajectoryPoint = {
  id: string;
  run_id?: string | null;
  trial_id?: string | null;
  seed?: number | null;
  boundary_index: number;
  boundary_value: number;
  boundary_unit: string;
  status: string;
  checkpoint_artifact_id?: string | null;
  evaluation_id?: string | null;
  gate_decision_id?: string | null;
  gate_action?: CheckpointGateDecision["action"] | null;
  metric_value?: number | null;
  metrics?: Record<string, number | null>;
  reason?: string | null;
  resource_use?: Record<string, number | null>;
  created_at?: string | null;
};

export type RunGroupTrajectory = {
  run_group_id: string;
  policy_revision?: CheckpointPolicyRevision | null;
  resolved_plan?: ResolvedCheckpointPlan | null;
  points: CheckpointTrajectoryPoint[];
  gate_decisions?: CheckpointGateDecision[];
  summary?: Record<string, number | string | null>;
};

export type EvidenceCompatibility = {
  compatible: boolean;
  reasons?: string[];
  matched_seed_count?: number;
  required_seed_count?: number;
  suite_revision_id?: string | null;
  settings_hash?: string | null;
};

export type CohortAnalysisSnapshot = {
  id: string;
  run_group_id?: string | null;
  request: Record<string, unknown>;
  analysis: {
    classification?: "improved" | "regressed" | "practically_equivalent" | "inconclusive" | "insufficient_evidence" | string;
    primary_metric?: string | null;
    direction?: "maximize" | "minimize" | string | null;
    matched_seed_count?: number;
    practical_delta?: number | null;
    interval?: { lower?: number | null; upper?: number | null; confidence?: number | null };
    summary?: Record<string, number | string | null>;
    observations?: Array<Record<string, unknown>>;
    pareto?: Array<Record<string, unknown>>;
    replicate_unit?: string;
    subjects?: Record<string, Record<string, unknown>>;
    comparisons?: Record<string, {
      classification?: string;
      matched_seed_count?: number;
      complete_required_seed_coverage?: boolean;
      confidence_interval?: { lower?: number | null; upper?: number | null } | null;
      mean_delta?: number | null;
      reason?: string | null;
      [key: string]: unknown;
    }>;
    compatibility?: EvidenceCompatibility;
    [key: string]: unknown;
  };
  status: string;
  content_hash: string;
  work_item_id?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
};

export type ResearchDecisionRecord = {
  id: string;
  analysis_snapshot_id: string;
  selected_subject: Record<string, unknown>;
  rejected_subjects?: Array<Record<string, unknown>>;
  exclusions?: Array<Record<string, unknown>>;
  rationale: string;
  override_reason?: string | null;
  fork_spec?: Record<string, unknown> | null;
  content_hash: string;
  created_at?: string | null;
};

export type EvidenceBundle = {
  id: string;
  analysis_snapshot_id: string;
  research_decision_id?: string | null;
  status: string;
  storage_path?: string | null;
  manifest?: Record<string, unknown>;
  content_hash?: string | null;
  work_item_id?: string | null;
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type LongitudinalEvidenceSeries = {
  subject_id: string;
  suite_revision_id: string;
  compatible: boolean;
  warnings?: string[];
  points: Array<{ timestamp?: string | null; checkpoint_artifact_id?: string | null; metric: string; value: number | null; practical_delta?: number | null }>;
};

export type WorkspaceDraft<T = Record<string, unknown>> = {
  id?: string;
  kind: string;
  owner?: string | null;
  name: string;
  content: T;
  content_hash?: string | null;
  expires_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type GlobalSearchResult = {
  id: string;
  type: "dataset" | "dataset_version" | "run" | "suite" | "run_group" | "artifact" | "checkpoint_policy" | "activity" | string;
  label: string;
  description?: string | null;
  status?: string | null;
  short_hash?: string | null;
  short_id?: string | null;
  target?: string | null;
  url?: string | null;
  metadata?: Record<string, unknown>;
};

export type RunGroup = {
  id: string;
  name: string;
  kind: "repeat" | "sweep" | string;
  status: string;
  trainer_mode: TrainingMode | string;
  suite_revision_id: string;
  primary_metric: string;
  direction: "maximize" | "minimize";
  base_config: Record<string, unknown>;
  search_space?: Record<string, unknown>;
  seeds: number[];
  n_trials: number;
  sampler?: string | null;
  pruning?: Record<string, unknown>;
  checkpoint_policy_revision_id?: string | null;
  checkpoint_policy?: CheckpointPolicyRevision | null;
  resolved_checkpoint_plan?: ResolvedCheckpointPlan | null;
  latest_analysis?: CohortAnalysisSnapshot | null;
  awaiting_review_count?: number;
  config_hash?: string | null;
  best_trial_id?: string | null;
  best_value?: number | null;
  completed_trials?: number;
  failed_trials?: number;
  pruned_trials?: number;
  trials?: ExperimentTrial[];
  work_items?: WorkItem[];
  created_at?: string | null;
  updated_at?: string | null;
};

export type RunGroupCreatePayload = {
  version?: 1 | 2;
  name: string;
  kind: "repeat" | "sweep";
  trainer_mode: TrainingMode | string;
  suite_revision_id: string;
  base_config: Record<string, unknown>;
  seeds?: number[];
  n_trials?: number;
  search_space?: Record<string, unknown>;
  sampler?: "random" | "grid" | "tpe" | string;
  sampler_seed?: number;
  pruning?: { enabled?: boolean; reduction_factor?: number; budgets?: number[] };
  checkpoint_policy_revision_id?: string | null;
  resolved_checkpoint_plan?: ResolvedCheckpointPlan | null;
  priority?: number;
};

export type ArtifactBlob = {
  id: string;
  content_hash: string;
  format?: "huggingface" | "mlx" | "gguf" | "onnx" | string | null;
  dtype?: string | null;
  quantization?: string | null;
  size_bytes?: number | null;
  integrity?: "verified" | "unverified" | "invalid" | string;
  manifest?: Record<string, unknown>;
  created_at?: string | null;
};

export type ArtifactLocation = {
  id: string;
  blob_id?: string | null;
  path: string;
  kind?: "referenced" | "managed" | "trash" | string;
  available?: boolean;
  verified_at?: string | null;
  created_at?: string | null;
};

export type ArtifactAlias = {
  id: string;
  alias: string;
  artifact_id: string;
  previous_artifact_id?: string | null;
  note?: string | null;
  created_at?: string | null;
  superseded_at?: string | null;
};

export type ArtifactEdge = {
  id: string;
  parent_artifact_id: string;
  child_artifact_id: string;
  ordinal: number;
  relationship?: string;
  created_at?: string | null;
};

export type ModelArtifactOccurrence = {
  id: string;
  occurrence_id?: string;
  kind: "checkpoint" | "adapter" | "final_model" | "merged" | "converted" | "quantized" | "export_bundle" | string;
  content_hash: string;
  path: string;
  blob_id?: string | null;
  blob?: ArtifactBlob | null;
  locations?: ArtifactLocation[];
  run_id?: string | null;
  run_group_id?: string | null;
  trial_id?: string | null;
  segment_id?: string | null;
  model_name?: string | null;
  tokenizer_revision?: string | null;
  step?: number | null;
  cycle?: number | null;
  format?: string | null;
  dtype?: string | null;
  quantization?: string | null;
  size_bytes?: number | null;
  integrity?: string | null;
  pinned?: boolean;
  tags?: string[];
  notes?: string | null;
  aliases?: Array<ArtifactAlias | string>;
  parents?: ArtifactEdge[];
  metadata?: Record<string, unknown>;
  specialized_task?: {
    task_kind: string;
    modality: string;
    label_schema_revision_id?: string | null;
    model_head_hash: string;
    processor_hash: string;
    loss_adapter: string;
    loss_adapter_version: string;
    retrieval_corpus_hash?: string | null;
    metadata?: Record<string, unknown>;
    created_at?: string | null;
  } | null;
  created_at?: string | null;
};

/** Backwards-compatible name used by the v3 run and experiment views. */
export type ModelArtifact = ModelArtifactOccurrence;

export type ArtifactOperation = {
  id: string;
  kind: "merge" | "bake" | "convert" | "quantize" | "verify" | "qualify" | "serve" | "export" | "cleanup" | string;
  status: string;
  input_artifact_ids: string[];
  output_artifact_id?: string | null;
  work_item_id?: string | null;
  config?: Record<string, unknown>;
  resolved_inputs?: Array<Record<string, unknown>>;
  tool_versions?: Record<string, string>;
  logs?: string[];
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type QualificationProfileRevision = {
  id: string;
  profile_id?: string | null;
  revision?: number;
  name?: string | null;
  development_suite_revision_id: string;
  operational_suite_revision_id: string;
  holdout_suite_revision_id?: string | null;
  metrics: Array<{
    name?: string;
    metric?: string;
    stage?: "development" | "operational" | "holdout" | string;
    direction: "maximize" | "minimize";
    threshold?: number | null;
    pass_threshold?: number | null;
    allowed_delta?: number | null;
    maximum_regression?: number | null;
    required?: boolean;
  }>;
  target_backend: string;
  generation_settings?: Record<string, unknown>;
  performance_settings?: Record<string, unknown>;
  description?: string | null;
  content_hash?: string | null;
  created_at?: string | null;
};

export type QualificationProfileCreatePayload = {
  profile_id?: string;
  name: string;
  description?: string;
  development_suite_revision_id: string;
  operational_suite_revision_id: string;
  holdout_suite_revision_id?: string | null;
  thresholds: Array<{
    stage: "development" | "operational" | "holdout";
    metric: string;
    direction: "maximize" | "minimize";
    pass_threshold?: number | null;
    maximum_regression?: number | null;
    required?: boolean;
  }>;
  target_backend: string;
  generation_settings?: Record<string, unknown>;
  performance_settings?: Record<string, unknown>;
};

export type ArtifactQualification = {
  id: string;
  artifact_id: string;
  parent_artifact_id?: string | null;
  profile_revision_id: string;
  status: "queued" | "running" | "pass" | "warn" | "fail" | string;
  decision?: "pass" | "warn" | "fail" | null;
  reasons?: string[];
  metrics?: Record<string, number | null>;
  quality_deltas?: Record<string, number | null>;
  performance?: Record<string, number | null>;
  work_item_id?: string | null;
  override_note?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type QualificationComparison = {
  profile_revision_id: string;
  profile_content_hash?: string | null;
  base_qualification_id: string;
  candidate_qualification_id: string;
  parent_artifact_hash?: string | null;
  candidate_artifact_hash?: string | null;
  deltas: Array<{
    stage?: string;
    metric: string;
    direction: "maximize" | "minimize" | string;
    parent_value?: number | null;
    candidate_value?: number | null;
    raw_delta?: number | null;
    favorable_delta?: number | null;
  }>;
};

export type ServingProfileRevision = {
  id: string;
  profile_id?: string | null;
  revision?: number;
  artifact_id: string;
  backend: string;
  host?: string;
  port?: number;
  chat_template?: string | null;
  generation_defaults?: Record<string, unknown>;
  resource_expectations?: Record<string, unknown>;
  created_at?: string | null;
};

export type StorageInventory = {
  generated_at?: string | null;
  root?: string | null;
  total_bytes?: number | null;
  used_bytes?: number | null;
  free_bytes?: number | null;
  minimum_free_bytes?: number | null;
  projected_free_bytes?: number | null;
  low_disk?: boolean;
  artifact_bytes?: number | null;
  cache_bytes?: number | null;
  temporary_bytes?: number | null;
  import_staging_bytes?: number | null;
  import_staging_items?: Array<{
    id: string;
    name: string;
    status: string;
    size_bytes: number;
    expires_at?: string | null;
    cleanup_eligible: boolean;
    resource_type: "dataset_import_staging" | string;
  }>;
  trash_bytes?: number | null;
  protected_bytes?: number | null;
  managed_locations?: ArtifactLocation[];
  cache_items?: ArtifactLocation[];
  forecast?: Record<string, unknown>;
};

export type CleanupPlan = {
  id: string;
  status: "preview" | "approved" | "running" | "completed" | "failed" | string;
  reclaimable_bytes?: number | null;
  items: Array<{
    id: string;
    path?: string | null;
    size_bytes?: number | null;
    reason?: string | null;
    protected?: boolean;
  }>;
  work_item_id?: string | null;
  trash_retention_days?: number;
  created_at?: string | null;
};

export type PlaygroundSessionMessage = PlaygroundMessage & {
  id?: string;
  artifact_id?: string | null;
  generation?: Record<string, unknown>;
  evidence?: Record<string, unknown>;
  created_at?: string | null;
};

export type PlaygroundSession = {
  id: string;
  name: string;
  artifact_id?: string | null;
  compare_artifact_id?: string | null;
  endpoint?: string | null;
  seed?: number | null;
  generation_settings?: Record<string, unknown>;
  settings?: Record<string, unknown>;
  messages: PlaygroundSessionMessage[];
  created_at?: string | null;
  updated_at?: string | null;
  archived?: boolean;
};

export type PlaygroundReviewResult = {
  id?: string;
  kind: "benchmark_suite" | "dataset_source" | string;
  status?: string;
  benchmark_suite_revision_id?: string | null;
  dataset_source_draft_id?: string | null;
  review_queue_id?: string | null;
  pairing_count?: number;
  reviewed_turn_count?: number;
  starts_training?: boolean;
  work_item_id?: string | null;
  created_at?: string | null;
};

export type PlaygroundReviewPairing = {
  prompt_message_id: string;
  base_message_id: string;
  candidate_message_id: string;
};

export type ActivitySnapshot = {
  items: ActivityItem[];
  worker?: Worker | null;
  workers?: Worker[];
  resource_lease?: Record<string, unknown> | null;
  storage?: StorageInventory | null;
  source?: "activity" | "work-items";
};

export type PaginatedResponse<T> = {
  items: T[];
  total?: number;
  limit?: number;
  offset?: number;
  next_cursor?: string | null;
};

// ----- Review Studio -----------------------------------------------------

export type AnnotationModality = "text" | "vlm" | "audio" | string;
export type AnnotationTaskType =
  | "binary"
  | "categorical"
  | "multi_label"
  | "scalar"
  | "text_correction"
  | "structured_correction"
  | "pairwise"
  | "ranking"
  | string;

export type AnnotationSchema = {
  id: string;
  name: string;
  description?: string | null;
  archived?: boolean;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AnnotationSchemaRevision = {
  id: string;
  schema_id: string;
  revision_number: number;
  content_hash?: string;
  modality: AnnotationModality;
  task_type: AnnotationTaskType;
  definition: Record<string, unknown>;
  created_at?: string | null;
};

export type AcquisitionStrategy = {
  kind: string;
  quota?: number | null;
  options?: Record<string, unknown>;
};

export type AcquisitionSource = {
  kind: "evaluation" | "evaluation_comparison" | "dataset_version" | "playground_session" | "run_samples" | "import" | "jsonl" | string;
  ref?: string;
  id?: string;
  base_id?: string;
  candidate_id?: string;
  split?: string;
  purpose?: string;
  [key: string]: unknown;
};

export type AcquisitionRequest = {
  name: string;
  seed: number;
  sources?: AcquisitionSource[];
  source?: AcquisitionSource;
  records?: Array<Record<string, unknown>>;
  strategies: AcquisitionStrategy[];
  metadata?: Record<string, unknown>;
};

export type AcquisitionBatch = {
  id: string;
  name: string;
  status: string;
  request?: Record<string, unknown>;
  source_hash?: string | null;
  content_hash?: string | null;
  seed?: number;
  row_count?: number;
  eligibility?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  work_item_id?: string | null;
  stage?: string | null;
  processed_records?: number | null;
  total_records?: number | null;
  progress_percent?: number | null;
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
};

export type AcquisitionCandidate = {
  id: string;
  batch_id: string;
  ordinal: number;
  record_id: string;
  record_hash?: string;
  source_kind?: string;
  source_ref?: string | null;
  source_record_id?: string | null;
  record: Record<string, unknown>;
  evidence?: Record<string, unknown>;
  source?: Record<string, unknown>;
  stratum?: string | null;
  score?: number | null;
  created_at?: string | null;
};

export type ReviewPolicy = {
  mode: "one_pass" | "two_pass" | string;
  blind_second_pass?: boolean;
  allow_suggestions?: boolean;
  require_adjudication?: boolean;
};

export type ReviewOutputAdapterDescriptor = {
  id: string;
  version: number;
  modalities: string[];
  task_types: string[];
  build_modes: string[];
  default_build_mode: string;
};

export type ReviewCapabilities = {
  modalities: Array<{ id: string; task_types: string[] }>;
  acquisition_strategies: string[];
  acquisition_source_kinds?: string[];
  verifier_failure_selectors?: string[];
  review_policies: string[];
  event_types: string[];
  output_adapters: ReviewOutputAdapterDescriptor[];
  max_event_batch_size: number;
  protected_suite_purposes: string[];
  protected_splits: string[];
};

export type ReviewQueue = {
  id: string;
  name: string;
  status: string;
  acquisition_batch_id: string;
  schema_revision_id: string;
  policy: ReviewPolicy;
  content_hash?: string;
  current_pass: number;
  latest_label_set_revision_id?: string | null;
  statistics?: ReviewQueueStatistics;
  created_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
};

export type ReviewQueueStatistics = {
  total?: number;
  resolved?: number;
  coverage?: number;
  pending?: number;
  completed?: number;
  excluded?: number;
  flagged?: number;
  needs_adjudication?: number;
  conflicts?: number;
  status_counts?: Record<string, number>;
  two_pass_compared?: number;
  two_pass_agreements?: number;
  two_pass_agreement_rate?: number | null;
  event_counts?: Record<string, number>;
  correction_rate?: number;
  unpublished_changes?: boolean;
  by_status?: Record<string, number>;
  [key: string]: unknown;
};

export type ReviewItem = {
  id: string;
  queue_id: string;
  candidate_id: string;
  ordinal: number;
  status: string;
  active_event_id?: string | null;
  projection?: Record<string, unknown>;
  record?: Record<string, unknown>;
  evidence?: Record<string, unknown>;
  source?: Record<string, unknown>;
  record_id?: string;
  record_hash?: string;
  created_at?: string | null;
  updated_at?: string | null;
};

export type ReviewEvent = {
  id: string;
  queue_id: string;
  item_id: string;
  event_type: string;
  pass_number: number;
  idempotency_key: string;
  request_hash?: string;
  expected_active_event_id?: string | null;
  payload: Record<string, unknown>;
  supersedes_event_id?: string | null;
  created_at?: string | null;
};

export type ReviewSuggestion = {
  id: string;
  item_id: string;
  pass_number: number;
  status?: string;
  work_item_id?: string;
  provider?: string;
  model_revision?: string;
  content_hash?: string;
  output: Record<string, unknown> | null;
  provenance?: Record<string, unknown>;
  created_at?: string | null;
};

export type LabelSet = {
  id: string;
  queue_id: string;
  name: string;
  latest_revision_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type LabelSetRevision = {
  id: string;
  label_set_id: string;
  revision_number: number;
  content_hash?: string;
  storage_path?: string;
  row_count: number;
  excluded_count?: number;
  manifest?: Record<string, unknown>;
  created_at?: string | null;
};

export type LabelSetPublicationAccepted = {
  id: string;
  publication_id: string;
  queue_id: string;
  status: "queued" | string;
  work_item_id: string;
  label_set_revision_id: string | null;
};

export type LabelSetItem = {
  revision_id: string;
  ordinal: number;
  review_item_id: string;
  record_id: string;
  record_hash?: string;
  annotation: Record<string, unknown>;
  output_records?: Array<Record<string, unknown>>;
  lineage?: Record<string, unknown>;
  excluded?: boolean;
  exclusion_reason?: string | null;
};

export type DatasetBuildPreview = {
  label_set_revision_id: string;
  revision_id?: string;
  dataset_id: string;
  parent_version_id?: string | null;
  build_mode: string;
  target_split?: string;
  source_count?: number;
  output_count?: number;
  added_count?: number;
  removed_count?: number;
  replaced_count?: number;
  annotated_count?: number;
  excluded_count?: number;
  quarantined_count?: number;
  split_counts?: Record<string, number>;
  moved_from_splits?: Record<string, number>;
  contamination?: Record<string, unknown>;
  sample?: Array<Record<string, unknown>>;
  items?: Array<Record<string, unknown>>;
  warnings?: string[];
  total?: number;
  limit?: number;
  offset?: number;
  new_dataset?: boolean;
  starts_training: false;
};

export type SpecFieldDescriptor = {
  name: string;
  label: string;
  value_type: string;
  required?: boolean;
  default?: unknown;
  options?: Array<string | { value: string; label?: string }>;
  description?: string;
  placeholder?: string;
  visible_when?: Record<string, unknown>;
};

export type SpecDescriptor = {
  kind: string;
  id: string;
  version: string | number;
  label: string;
  description?: string;
  fields: SpecFieldDescriptor[];
};

export type SpecValidationResult = {
  valid: boolean;
  value: Record<string, unknown>;
  errors: Array<{ field?: string; message: string }>;
};

type DatasetRecordWire = {
  id: string;
  name: string;
  description?: string | null;
  modality?: string | null;
  canonical_schema?: string | Record<string, unknown>;
  latest_version_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  sources?: DatasetSource[];
  source?: DatasetSource | null;
  row_count?: number | null;
  size_bytes?: number | null;
  versions?: DatasetVersion[];
  jobs?: DatasetJob[];
  job?: DatasetJob | null;
  latest_version?: DatasetVersion | null;
  active_job?: DatasetJob | null;
  [key: string]: unknown;
};

function normalizeDatasetRecord(record: DatasetRecordWire): DatasetRecord {
  const sources = record.sources ?? (record.source ? [record.source] : []);
  const jobs = record.jobs ?? (record.job ? [record.job] : []);
  const activeJob =
    record.active_job ??
    jobs.find((job) => ["queued", "running", "building", "materializing"].includes(job.status)) ??
    record.job ??
    null;
  return { ...record, sources, jobs, active_job: activeJob };
}

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

function isMissingEndpoint(error: unknown): boolean {
  return error instanceof ApiError && [404, 405, 501].includes(error.status);
}

function workItemToActivity(item: WorkItem, index: number): ActivityItem {
  const progressPercent =
    typeof item.progress_current === "number" &&
    typeof item.progress_total === "number" &&
    item.progress_total > 0
      ? Math.min(100, (item.progress_current / item.progress_total) * 100)
      : null;
  const payloadTitle =
    typeof item.payload?.name === "string"
      ? item.payload.name
      : typeof item.payload?.model === "string"
        ? item.payload.model
        : null;
  return {
    id: item.id,
    work_item_id: item.id,
    domain_id: item.run_id ?? item.run_group_id ?? item.trial_id ?? null,
    domain_type: item.run_id ? "run" : item.run_group_id ? "run_group" : null,
    kind: item.kind,
    title: payloadTitle ?? item.stage ?? item.kind.replace(/[_-]/g, " "),
    status: item.status,
    stage: item.stage,
    priority: item.priority,
    progress_current: item.progress_current,
    progress_total: item.progress_total,
    progress_percent: progressPercent,
    queue_position: item.status === "queued" ? index + 1 : null,
    attempt: item.attempt,
    max_attempts: item.max_attempts,
    error: item.error,
    created_at: item.created_at,
    started_at: item.started_at,
    completed_at: item.completed_at,
    heartbeat_at: item.heartbeat_at,
    next_actions: ["failed", "interrupted", "needs_reconciliation"].includes(item.status)
      ? ["retry"]
      : ["queued", "running", "blocked"].includes(item.status)
        ? ["cancel"]
        : [],
  };
}

async function loadActivitySnapshot(limit = 100): Promise<ActivitySnapshot> {
  try {
    const response = await request<ActivitySnapshot>(`/activity?limit=${encodeURIComponent(String(limit))}`);
    return { ...response, items: response.items ?? [], source: "activity" };
  } catch (error) {
    if (!isMissingEndpoint(error)) throw error;
    const [work, workers, storage] = await Promise.allSettled([
      request<{ items: WorkItem[]; active_lease?: Record<string, unknown> | null }>(
        `/work-items?limit=${encodeURIComponent(String(limit))}`,
      ),
      request<{ items: Worker[] }>("/workers"),
      request<StorageInventory>("/storage"),
    ]);
    if (work.status === "rejected") throw work.reason;
    const workerItems = workers.status === "fulfilled" ? workers.value.items : [];
    return {
      items: (work.value.items ?? []).map(workItemToActivity),
      worker: workerItems[0] ?? null,
      workers: workerItems,
      resource_lease: work.value.active_lease ?? null,
      storage: storage.status === "fulfilled" ? storage.value : null,
      source: "work-items",
    };
  }
}

type JsonRecord = Record<string, unknown>;

function jsonRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function jsonString(value: unknown, fallback = ""): string {
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

function jsonStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => jsonString(item)).filter(Boolean) : [];
}

function normalizeVerifierCapability(value: unknown): VerifierCapabilityDescriptor {
  const raw = jsonRecord(value);
  const id = jsonString(raw.id ?? raw.key ?? raw.adapter_id, "unknown-verifier");
  return {
    id,
    family: jsonString(raw.family, "deterministic") as VerifierFamily,
    label: jsonString(raw.label ?? raw.display_name, id),
    description: jsonString(raw.description ?? raw.warning ?? raw.implementation) || null,
    implementation: jsonString(raw.implementation) || null,
    implementation_fingerprint: jsonString(raw.implementation_fingerprint ?? raw.fingerprint) || null,
    origin: jsonString(raw.origin) || undefined,
    fingerprintable: typeof raw.fingerprintable === "boolean" ? raw.fingerprintable : raw.qualifiable !== false,
    modalities: jsonStrings(raw.modalities),
    task_types: jsonStrings(raw.task_types ?? raw.tasks),
    supports_probability: typeof raw.supports_probability === "boolean" ? raw.supports_probability : undefined,
    supports_seed: typeof raw.supports_seed === "boolean" ? raw.supports_seed : undefined,
    runtime_requirements: jsonRecord(raw.runtime_requirements),
    compatible_consumers: jsonStrings(raw.compatible_consumers),
    reliability_adapter_id: jsonString(raw.reliability_adapter_id ?? raw.adapter_id) || undefined,
    reliability_adapter_version: jsonString(raw.reliability_adapter_version ?? raw.adapter_version) || undefined,
  };
}

function normalizeVerifierRevision(
  value: unknown,
  aliases: string[] = [],
  overriddenAliases: string[] = [],
): VerifierProfileRevision {
  const raw = jsonRecord(value);
  const definition = jsonRecord(raw.definition);
  const configuration = jsonRecord(definition.configuration);
  const runtimeContract = jsonRecord(raw.runtime_contract ?? raw.runtime_requirements ?? definition.runtime_contract);
  const reward = jsonRecord(raw.reward_contract ?? definition.reward_contract);
  const alias = aliases.includes("approved") ? "approved" : aliases.includes("candidate") ? "candidate" : null;
  const overriddenAlias = overriddenAliases.includes("approved") ? "approved" : overriddenAliases.includes("candidate") ? "candidate" : null;
  const componentValues = Array.isArray(raw.components) ? raw.components : [];
  return {
    id: jsonString(raw.id),
    profile_id: jsonString(raw.profile_id),
    revision_number: Number(raw.revision_number ?? 1),
    family: jsonString(raw.family ?? definition.family, "deterministic") as VerifierFamily,
    modality: jsonString(raw.modality ?? definition.modality, "text"),
    task_type: jsonString(raw.task_type ?? definition.task_type, "binary"),
    implementation_id: jsonString(raw.implementation_id ?? raw.implementation_ref ?? definition.implementation_id) || null,
    implementation_fingerprint: jsonString(raw.implementation_fingerprint) || null,
    reliability_adapter_id: jsonString(raw.reliability_adapter_id) || null,
    reliability_adapter_version: jsonString(raw.reliability_adapter_version) || null,
    input_mapping: jsonRecord(raw.input_mapping ?? definition.input_mapping),
    output_contract: jsonRecord(raw.output_contract ?? definition.output_contract),
    reward_contract: {
      minimum: Number(reward.minimum ?? reward.min ?? 0),
      maximum: Number(reward.maximum ?? reward.max ?? 1),
      direction: jsonString(reward.direction, "maximize") as "maximize" | "minimize",
      threshold: reward.threshold === null ? null : Number(reward.threshold ?? 0.5),
      tie_policy: jsonString(reward.tie_policy, "error"),
      probability_semantics: reward.probability_semantics === true,
      error_behavior: jsonString(reward.error_behavior, "fail_closed"),
    },
    rubric: jsonString(raw.rubric ?? definition.rubric) || null,
    prompt_template: jsonString(raw.prompt_template ?? definition.prompt_template) || null,
    parser: typeof (raw.parser ?? definition.parser) === "string"
      ? String(raw.parser ?? definition.parser)
      : Object.keys(jsonRecord(raw.parser ?? definition.parser)).length
        ? jsonRecord(raw.parser ?? definition.parser)
        : null,
    model_revision: jsonString(raw.model_revision ?? definition.model_revision ?? configuration.model_revision) || null,
    tokenizer_revision: jsonString(raw.tokenizer_revision ?? definition.tokenizer_revision ?? configuration.tokenizer_revision) || null,
    artifact_id: jsonString(raw.artifact_id ?? definition.artifact_id) || null,
    artifact_hash: jsonString(raw.artifact_hash ?? definition.artifact_hash) || null,
    endpoint_type: jsonString(raw.endpoint_type ?? definition.endpoint_type ?? configuration.endpoint_type) || null,
    generation_settings: jsonRecord(raw.generation_settings ?? definition.generation_settings),
    runtime_requirements: runtimeContract,
    components: componentValues.map((item) => {
      const component = jsonRecord(item);
      return {
        id: jsonString(component.id ?? component.revision_id) || undefined,
        child_revision_id: jsonString(component.child_revision_id),
        ordinal: Number(component.ordinal ?? 0),
        weight: Number(component.weight ?? 1),
        veto: component.veto === true,
        aggregation_rule: jsonString(component.aggregation_rule) || null,
      };
    }),
    content_hash: jsonString(raw.content_hash),
    qualification_state: alias ? "pass" : overriddenAlias ? "warn" : raw.qualifiable === false ? "unqualified" : jsonString(raw.qualification_state, "unqualified"),
    alias: alias ?? overriddenAlias,
    overridden: Boolean(overriddenAlias && !alias),
    runtime_compatible: typeof raw.runtime_compatible === "boolean" ? raw.runtime_compatible : undefined,
    created_at: jsonString(raw.created_at) || null,
  };
}

function normalizeVerifierProfile(value: unknown): VerifierProfile & { revisions?: VerifierProfileRevision[] } {
  const raw = jsonRecord(value);
  const profile = jsonRecord(raw.profile && typeof raw.profile === "object" ? raw.profile : raw);
  const aliases = Array.isArray(raw.aliases)
    ? raw.aliases.map((item) => typeof item === "string" ? item : jsonString(jsonRecord(item).alias)).filter(Boolean)
    : [];
  const overriddenAliases = jsonStrings(raw.overridden_aliases);
  const latestRaw = raw.latest_revision ?? raw.revision;
  const revisionsRaw = Array.isArray(raw.revisions) ? raw.revisions : [];
  const revisions = revisionsRaw.map((item, index) => normalizeVerifierRevision(item, index === 0 ? aliases : [], index === 0 ? overriddenAliases : []));
  const latest = latestRaw ? normalizeVerifierRevision(latestRaw, aliases, overriddenAliases) : revisions[0] ?? null;
  return {
    id: jsonString(profile.id),
    name: jsonString(profile.name, "Unnamed verifier"),
    description: jsonString(profile.description) || null,
    latest_revision_id: jsonString(profile.latest_revision_id ?? latest?.id) || null,
    latest_revision: latest,
    revision_count: revisions.length || (latest ? 1 : 0),
    created_at: jsonString(profile.created_at) || null,
    updated_at: jsonString(profile.updated_at) || null,
    revisions: revisions.length ? revisions : latest ? [latest] : [],
  };
}

function normalizeVerifierProtocol(value: unknown): VerifierCalibrationProtocolRevision {
  const raw = jsonRecord(value);
  const head = jsonRecord(raw.protocol && typeof raw.protocol === "object" ? raw.protocol : raw);
  const revision = jsonRecord(raw.revision ?? raw.latest_revision);
  const definition = jsonRecord(revision.definition ?? raw.definition);
  return {
    id: jsonString(revision.id ?? head.latest_revision_id ?? head.id),
    protocol_id: jsonString(revision.protocol_id ?? head.id),
    revision_number: Number(revision.revision_number ?? 1),
    name: jsonString(head.name ?? raw.name, "Calibration protocol"),
    family: jsonString(definition.family, "all") as VerifierFamily | "all",
    repeats: typeof definition.repeats === "number" ? definition.repeats : undefined,
    seeds: Array.isArray(definition.seeds ?? definition.stochastic_seeds) ? (definition.seeds ?? definition.stochastic_seeds) as number[] : undefined,
    temperature: typeof definition.temperature === "number" ? definition.temperature : undefined,
    top_p: typeof definition.top_p === "number" ? definition.top_p : undefined,
    concurrency: typeof definition.concurrency === "number" ? definition.concurrency : undefined,
    confirmation_requested: definition.confirmation_requested === true,
    confirmation_fraction: typeof definition.confirmation_fraction === "number" ? definition.confirmation_fraction : undefined,
    partition_seed: typeof definition.partition_seed === "number" ? definition.partition_seed : undefined,
    bootstrap_resamples: typeof definition.bootstrap_resamples === "number" ? definition.bootstrap_resamples : undefined,
    bootstrap_seed: typeof definition.bootstrap_seed === "number" ? definition.bootstrap_seed : undefined,
    perturbations: jsonStrings(definition.perturbations),
    settings: definition,
    content_hash: jsonString(revision.content_hash),
    created_at: jsonString(revision.created_at ?? head.created_at) || null,
  };
}

function normalizeVerifierQualificationProfile(value: unknown): VerifierQualificationProfileRevision {
  const raw = jsonRecord(value);
  const head = jsonRecord(raw.profile && typeof raw.profile === "object" ? raw.profile : raw);
  const revision = jsonRecord(raw.revision ?? raw.latest_revision);
  const requirements = jsonRecord(revision.requirements ?? raw.requirements);
  return {
    id: jsonString(revision.id ?? head.latest_revision_id ?? head.id),
    profile_id: jsonString(revision.profile_id ?? head.id),
    revision_number: Number(revision.revision_number ?? 1),
    name: jsonString(head.name ?? raw.name, jsonString(revision.template_kind, "Qualification profile")),
    template: jsonString(revision.template_kind ?? raw.template_kind, "human_aligned"),
    thresholds: requirements,
    minimum_evidence: jsonRecord(requirements.minimum_evidence) as Record<string, number>,
    promotable: revision.promotable !== false,
    content_hash: jsonString(revision.content_hash),
    created_at: jsonString(revision.created_at ?? head.created_at) || null,
  };
}

function finiteNumber(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function optionalNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizePage<T>(value: unknown, normalize: (item: unknown) => T): PaginatedResponse<T> {
  const raw = jsonRecord(value);
  const sourceItems = Array.isArray(raw.items) ? raw.items : [];
  return {
    items: sourceItems.map(normalize),
    total: finiteNumber(raw.total, sourceItems.length),
    limit: finiteNumber(raw.limit, sourceItems.length || 100),
    offset: finiteNumber(raw.offset, 0),
  };
}

function normalizeTrainingSignalCapability(value: unknown): TrainingSignalCapabilityDescriptor {
  const raw = jsonRecord(value);
  const mappings = jsonRecord(raw.mappings);
  const mappingKeys = (name: string) => Object.keys(jsonRecord(mappings[name] ?? raw[`${name}_mapping`]));
  return {
    id: jsonString(raw.id, "unknown-training-signal-capability"),
    version: typeof raw.version === "number" ? raw.version : jsonString(raw.version, "1"),
    trainer_mode: jsonString(raw.trainer_mode ?? raw.trainer, "unknown"),
    backend_family: jsonString(raw.backend_family ?? raw.backend, "unknown"),
    boundary_unit: jsonString(raw.boundary_unit, "full_trial"),
    resumable: raw.resumable === true,
    audit_boundaries: jsonStrings(raw.audit_boundaries ?? raw.available_boundaries),
    capture_fidelity: jsonString(raw.capture_fidelity ?? raw.fidelity, "unavailable"),
    candidate_multiplicity: jsonString(raw.candidate_multiplicity, "one"),
    mappings: {
      identity: mappingKeys("identity"),
      input: mappingKeys("input"),
      output: mappingKeys("output"),
      reference: mappingKeys("reference"),
      media: mappingKeys("media"),
      verifier: mappingKeys("verifier"),
    },
    unavailable_fields: jsonStrings(raw.unavailable_fields),
    reason: jsonString(raw.reason) || null,
  };
}

function normalizeRewardSystemRevision(value: unknown, fallbackName = ""): RewardSystemRevision {
  const wrapper = jsonRecord(value);
  const raw = jsonRecord(wrapper.revision && typeof wrapper.revision === "object" ? wrapper.revision : value);
  const definition = jsonRecord(raw.definition);
  const sourceAuditors = Array.isArray(raw.auditors) ? raw.auditors : [];
  const auditors = sourceAuditors.map((item, index): RewardSystemAuditor => {
    const auditor = jsonRecord(item);
    return {
      id: jsonString(auditor.id) || undefined,
      reward_system_revision_id: jsonString(auditor.reward_system_revision_id ?? raw.id) || undefined,
      verifier_profile_revision_id: jsonString(auditor.verifier_profile_revision_id) || undefined,
      verifier_revision_id: jsonString(auditor.verifier_revision_id ?? auditor.verifier_profile_revision_id) || undefined,
      role: jsonString(auditor.role, index === 0 ? "primary_sentinel" : "diagnostic"),
      ordinal: finiteNumber(auditor.ordinal, index),
      implementation_fingerprint: jsonString(auditor.implementation_fingerprint) || null,
      verifier_chain_leaf_fingerprints: jsonStrings(auditor.verifier_chain_leaf_fingerprints),
      correlated: auditor.correlated === true,
      correlation_reasons: jsonStrings(auditor.correlation_reasons),
    };
  });
  const primary = auditors.find((item) => item.role === "primary_sentinel");
  return {
    id: jsonString(raw.id),
    reward_system_id: jsonString(raw.reward_system_id ?? raw.system_id) || undefined,
    system_id: jsonString(raw.system_id ?? raw.reward_system_id) || undefined,
    revision_number: finiteNumber(raw.revision_number, 1),
    name: jsonString(raw.name, fallbackName) || undefined,
    optimizer_verifier_profile_revision_id: jsonString(raw.optimizer_verifier_profile_revision_id) || undefined,
    optimizer_verifier_revision_id: jsonString(raw.optimizer_verifier_revision_id ?? raw.optimizer_verifier_profile_revision_id) || undefined,
    optimizer_verifier_hash: jsonString(raw.optimizer_verifier_hash) || null,
    modality: jsonString(raw.modality, "unknown"),
    task_type: jsonString(raw.task_type, "unknown"),
    input_mapping: jsonRecord(raw.input_mapping),
    reward_normalization: jsonRecord(raw.reward_normalization),
    reward_mapping: jsonRecord(raw.reward_mapping),
    threshold: optionalNumber(raw.threshold),
    failure_behavior: jsonString(raw.failure_behavior ?? definition.failure_behavior) || null,
    shaping: jsonRecord(raw.shaping ?? definition.shaping),
    definition,
    compatible_capabilities: jsonStrings(raw.compatible_capabilities),
    auditors,
    qualification_state: jsonString(raw.qualification_state, primary?.correlated ? "correlated" : "published"),
    content_hash: jsonString(raw.content_hash),
    created_at: jsonString(raw.created_at) || null,
  };
}

function normalizeRewardSystem(value: unknown): RewardSystem {
  const raw = jsonRecord(value);
  const head = jsonRecord(raw.system && typeof raw.system === "object" ? raw.system : raw);
  const revisionRaw = raw.revision ?? raw.latest_revision;
  const revision = revisionRaw ? normalizeRewardSystemRevision(revisionRaw, jsonString(head.name)) : null;
  const revisions = jsonRecord(raw.revisions);
  return {
    id: jsonString(head.id),
    name: jsonString(head.name, "Unnamed reward system"),
    description: jsonString(head.description) || null,
    latest_revision_id: jsonString(head.latest_revision_id ?? revision?.id) || null,
    latest_revision: revision,
    revision_count: finiteNumber(raw.revision_count ?? revisions.total, revision ? 1 : 0),
    created_at: jsonString(head.created_at) || null,
    updated_at: jsonString(head.updated_at) || null,
  };
}

function normalizeRewardAuditProtocol(value: unknown): RewardAuditProtocolRevision {
  const raw = jsonRecord(value);
  const head = jsonRecord(raw.protocol && typeof raw.protocol === "object" ? raw.protocol : raw);
  const revision = jsonRecord(raw.revision ?? raw.latest_revision ?? (raw.capture_mode ? raw : undefined));
  const definition = jsonRecord(revision.definition ?? raw.definition);
  const template = jsonString(revision.capture_mode ?? definition.capture_mode ?? head.name, "balanced_256");
  return {
    id: jsonString(revision.id ?? head.latest_revision_id ?? head.id),
    protocol_id: jsonString(revision.protocol_id ?? head.id) || undefined,
    revision_number: finiteNumber(revision.revision_number, 1),
    name: jsonString(head.name ?? raw.name, template),
    template,
    uniform_core_limit: optionalNumber(definition.uniform_core_limit) ?? undefined,
    diagnostic_limit: optionalNumber(definition.diagnostic_limit) ?? undefined,
    seed: optionalNumber(definition.seed) ?? undefined,
    boundaries: Array.isArray(definition.boundaries) ? definition.boundaries as Array<number | string> : undefined,
    capture_required_for_gating: definition.full_snapshot_required_for_gating !== false,
    definition,
    content_hash: jsonString(revision.content_hash) || undefined,
    created_at: jsonString(revision.created_at ?? head.created_at) || null,
  };
}

function normalizeRewardIntegrityProfile(value: unknown): RewardIntegrityProfileRevision {
  const raw = jsonRecord(value);
  const head = jsonRecord(raw.profile && typeof raw.profile === "object" ? raw.profile : raw);
  const revision = jsonRecord(raw.revision ?? raw.latest_revision ?? (raw.template_kind ? raw : undefined));
  const requirements = jsonRecord(revision.requirements ?? raw.requirements);
  const minimum = jsonRecord(requirements.minimum_records);
  const template = jsonString(revision.template_kind ?? head.name, "human_aligned_integrity");
  return {
    id: jsonString(revision.id ?? head.latest_revision_id ?? head.id),
    profile_id: jsonString(revision.profile_id ?? head.id) || undefined,
    revision_number: finiteNumber(revision.revision_number, 1),
    name: jsonString(head.name ?? raw.name, template),
    template,
    thresholds: jsonRecord(requirements.metrics) as RewardIntegrityProfileRevision["thresholds"],
    minimum_pass_records: optionalNumber(minimum.pass) ?? undefined,
    minimum_report_records: optionalNumber(minimum.warn) ?? undefined,
    bootstrap_resamples: optionalNumber(requirements.bootstrap_resamples) ?? undefined,
    bootstrap_seed: optionalNumber(requirements.bootstrap_seed) ?? undefined,
    promotable: revision.promotable !== false && requirements.report_only !== true,
    requirements,
    content_hash: jsonString(revision.content_hash) || undefined,
    created_at: jsonString(revision.created_at ?? head.created_at) || null,
  };
}

function normalizeRewardObservation(value: unknown): VerifierObservation | null {
  if (!value || typeof value !== "object") return null;
  const raw = jsonRecord(value);
  return {
    reward: optionalNumber(raw.reward),
    passed: typeof raw.passed === "boolean" ? raw.passed : null,
    parsed_value: raw.parsed_value,
    raw_output: raw.raw_output,
    details: jsonRecord(raw.details),
    component_trace: Array.isArray(raw.component_trace) ? raw.component_trace.map(jsonRecord) : [],
    latency_ms: optionalNumber(raw.latency_ms),
    error: jsonString(raw.error) || null,
    runtime_identity: jsonRecord(raw.runtime_identity),
  };
}

function normalizeRewardIntegritySample(value: unknown): RewardIntegrityObservation {
  const raw = jsonRecord(value);
  const recordRaw = jsonRecord(raw.record);
  const input = raw.input ?? raw.prompt ?? raw.context;
  const optimizerRaw = jsonRecord(raw.optimizer_observation);
  const sentinelRaw = jsonRecord(raw.sentinel_observation ?? raw.primary_sentinel_observation);
  const optimizer = normalizeRewardObservation(optimizerRaw);
  const sentinel = normalizeRewardObservation(sentinelRaw);
  const optimizerReward = optionalNumber(raw.normalized_optimizer_reward ?? optimizerRaw.normalized_reward);
  const sentinelReward = optionalNumber(raw.normalized_sentinel_reward ?? sentinelRaw.normalized_reward);
  const optimizerPassed = optimizer?.passed;
  const sentinelPassed = sentinel?.passed;
  const hasError = Boolean(optimizer?.error || sentinel?.error);
  const classification = jsonString(raw.classification) || (hasError
    ? "error"
    : optimizerPassed === true && sentinelPassed === false
      ? "optimizer_only_accept"
      : optimizerPassed === false && sentinelPassed === true
        ? "sentinel_only_accept"
        : optimizerPassed !== null && optimizerPassed !== undefined && optimizerPassed === sentinelPassed
          ? "agreement"
          : "unclassified");
  const diagnosticsRaw = Array.isArray(raw.diagnostic_observations) ? raw.diagnostic_observations : [];
  const media = Array.isArray(raw.media) ? raw.media.map((item, index) => {
    const mediaItem = jsonRecord(item);
    return {
      kind: jsonString(mediaItem.kind ?? mediaItem.type, `media-${index + 1}`),
      hash: jsonString(mediaItem.hash ?? mediaItem.content_hash ?? mediaItem.media_hash, "hash unavailable"),
      path: jsonString(mediaItem.path ?? mediaItem.reference) || null,
      metadata: jsonRecord(mediaItem.metadata),
    };
  }) : [];
  return {
    id: jsonString(raw.id ?? raw.snapshot_id ?? raw.ordinal),
    audit_id: jsonString(raw.audit_id),
    snapshot_id: jsonString(raw.snapshot_id ?? raw.id),
    record: {
      record_id: jsonString(recordRaw.record_id ?? raw.record_id ?? raw.snapshot_id),
      record_hash: jsonString(recordRaw.record_hash ?? raw.record_hash) || null,
      instance_id: jsonString(recordRaw.instance_id ?? raw.instance_id) || null,
      group_id: jsonString(recordRaw.group_id ?? raw.group_id) || null,
      identity_kind: jsonString(recordRaw.identity_kind) || undefined,
    },
    boundary_index: finiteNumber(raw.boundary_index, 0),
    candidate_ordinal: finiteNumber(raw.candidate_ordinal, 0),
    prompt: input,
    context: raw.context,
    output: raw.output,
    expected: raw.expected,
    media,
    optimizer_observation: optimizer,
    sentinel_observation: sentinel,
    diagnostic_observations: diagnosticsRaw.map((item) => {
      const diagnostic = jsonRecord(item);
      return {
        verifier_profile_revision_id: jsonString(diagnostic.verifier_profile_revision_id ?? diagnostic.verifier_revision_id),
        observation: normalizeRewardObservation(diagnostic.observation ?? diagnostic) ?? {},
      };
    }),
    normalized_optimizer_reward: optimizerReward,
    normalized_sentinel_reward: sentinelReward,
    reward_gap: optionalNumber(raw.reward_gap) ?? (optimizerReward !== null && sentinelReward !== null ? optimizerReward - sentinelReward : null),
    classification,
    capture_stratum: jsonString(raw.capture_stratum ?? raw.selection_class, raw.diagnostic === true ? "diagnostic" : "uniform_core"),
  };
}

function normalizeRewardIntegrityMetric(value: unknown): RewardIntegrityMetric {
  const raw = jsonRecord(value);
  const direction = jsonString(raw.direction);
  return {
    id: jsonString(raw.id) || undefined,
    audit_id: jsonString(raw.audit_id) || undefined,
    name: jsonString(raw.name, "unnamed_metric"),
    value: optionalNumber(raw.value),
    direction: direction === "maximize" || direction === "minimize" ? direction : undefined,
    lower_ci: optionalNumber(raw.lower_ci ?? raw.ci_low),
    upper_ci: optionalNumber(raw.upper_ci ?? raw.ci_high),
    record_count: optionalNumber(raw.record_count) ?? undefined,
    population: jsonString(raw.population, "uniform_core"),
    subgroup: jsonString(raw.subgroup) || null,
    available: raw.available !== false,
    reason: jsonString(raw.reason ?? raw.missing_reason) || null,
    details: jsonRecord(raw.details ?? raw.metadata),
  };
}

function normalizeRewardIntegrityDecision(value: unknown): RewardIntegrityDecision {
  const raw = jsonRecord(value);
  const evidence = jsonRecord(raw.evidence);
  return {
    id: jsonString(raw.id),
    audit_id: jsonString(raw.audit_id),
    decision: jsonString(raw.decision, "incomplete_evidence"),
    action: jsonString(raw.action) || undefined,
    reasons: jsonStrings(raw.reasons),
    record_count: optionalNumber(raw.record_count ?? evidence.record_count) ?? undefined,
    automatic: raw.override !== true,
    override: raw.override === true,
    review_action: raw.override === true ? jsonString(raw.action) || null : null,
    review_reason: jsonString(raw.override_note) || null,
    reviewed_at: raw.override === true ? jsonString(raw.created_at) || null : null,
    evidence,
    created_at: jsonString(raw.created_at) || null,
  };
}

function normalizeRewardIntegrityForkContext(value: unknown): RewardIntegrityForkContext {
  const raw = jsonRecord(value);
  const checkpoint = jsonRecord(raw.checkpoint);
  return {
    audit_id: jsonString(raw.audit_id),
    decision: normalizeRewardIntegrityDecision(raw.decision),
    parent_run_id: jsonString(raw.parent_run_id),
    checkpoint: {
      content_hash: jsonString(checkpoint.content_hash),
      path: jsonString(checkpoint.path) || null,
      occurrence_id: jsonString(checkpoint.occurrence_id) || null,
      artifact: Object.keys(jsonRecord(checkpoint.artifact)).length ? jsonRecord(checkpoint.artifact) : null,
      snapshot_path: jsonString(checkpoint.snapshot_path) || null,
      boundary_unit: jsonString(checkpoint.boundary_unit) || null,
      boundary_value: optionalNumber(checkpoint.boundary_value),
      segment_id: jsonString(checkpoint.segment_id) || null,
      integrity_source: jsonString(checkpoint.integrity_source) || null,
      blockers: jsonStrings(checkpoint.blockers),
    },
    reward_system_revision_id: jsonString(raw.reward_system_revision_id),
    reward_audit_protocol_revision_id: jsonString(raw.reward_audit_protocol_revision_id),
    reward_integrity_profile_revision_id: jsonString(raw.reward_integrity_profile_revision_id),
    signal_capability: Object.keys(jsonRecord(raw.signal_capability)).length ? jsonRecord(raw.signal_capability) : null,
    resume_mode: jsonString(raw.resume_mode, "initialize_from_checkpoint"),
    train_context: jsonRecord(raw.train_context),
    datasets: Array.isArray(raw.datasets) ? raw.datasets.map(jsonRecord) : [],
    launch_ready: raw.launch_ready === true,
    blockers: jsonStrings(raw.blockers),
    href: jsonString(raw.href),
    replay_sync: Object.keys(jsonRecord(raw.replay_sync)).length ? jsonRecord(raw.replay_sync) : undefined,
  };
}

function normalizeRewardIntegrityReviewResult(value: unknown): RewardIntegrityReviewResult {
  const raw = jsonRecord(value);
  return raw.decision && typeof raw.decision === "object"
    ? normalizeRewardIntegrityForkContext(raw)
    : normalizeRewardIntegrityDecision(raw);
}

function normalizeRewardIntegrityAudit(value: unknown): RewardIntegrityAudit {
  const raw = jsonRecord(value);
  const audit = jsonRecord(raw.audit && typeof raw.audit === "object" ? raw.audit : raw);
  const shard = jsonRecord(raw.signal_shard ?? audit.signal_shard);
  const metricPage = jsonRecord(raw.metrics);
  const decisionPage = jsonRecord(raw.decisions);
  const metrics = Array.isArray(metricPage.items) ? metricPage.items.map(normalizeRewardIntegrityMetric) : Array.isArray(raw.metrics) ? raw.metrics.map(normalizeRewardIntegrityMetric) : [];
  const decisions = Array.isArray(decisionPage.items) ? decisionPage.items.map(normalizeRewardIntegrityDecision) : [];
  const embeddedDecision = raw.latest_decision ?? audit.decision;
  const latestDecision = embeddedDecision ? normalizeRewardIntegrityDecision(embeddedDecision) : decisions.at(-1) ?? null;
  const processed = finiteNumber(audit.processed_records ?? audit.processed_samples, 0);
  const total = optionalNumber(audit.total_records ?? audit.total_samples);
  const progress = optionalNumber(audit.progress_percent) ?? (total && total > 0 ? Math.min(100, processed / total * 100) : undefined);
  return {
    id: jsonString(audit.id),
    run_id: jsonString(audit.run_id),
    segment_id: jsonString(audit.segment_id ?? audit.direct_run_segment_id ?? audit.trial_segment_id) || null,
    boundary_index: finiteNumber(audit.boundary_index ?? shard.boundary_index, 0),
    boundary_value: optionalNumber(audit.boundary_value ?? shard.boundary_value),
    boundary_unit: jsonString(audit.boundary_unit ?? shard.boundary_unit) || null,
    checkpoint_artifact_id: jsonString(audit.checkpoint_artifact_id ?? shard.checkpoint_artifact_id) || null,
    training_signal_shard_id: jsonString(audit.training_signal_shard_id ?? audit.signal_shard_id),
    signal_shard_id: jsonString(audit.signal_shard_id ?? audit.training_signal_shard_id) || undefined,
    reward_system_revision_id: jsonString(audit.reward_system_revision_id),
    protocol_revision_id: jsonString(audit.protocol_revision_id),
    integrity_profile_revision_id: jsonString(audit.integrity_profile_revision_id),
    status: jsonString(audit.status, "unknown"),
    stage: jsonString(audit.stage) || null,
    processed_records: processed,
    total_records: total ?? undefined,
    distinct_record_count: optionalNumber(audit.distinct_record_count) ?? undefined,
    progress_percent: progress,
    capture_fidelity: jsonString(audit.capture_fidelity ?? shard.capture_fidelity) || undefined,
    metrics,
    decision: latestDecision,
    work_item_id: jsonString(audit.work_item_id) || null,
    evidence_hash: jsonString(audit.evidence_hash ?? audit.manifest_hash) || null,
    trace_hash: jsonString(audit.trace_hash ?? shard.trace_hash) || null,
    error: jsonString(audit.error) || null,
    created_at: jsonString(audit.created_at) || null,
    completed_at: jsonString(audit.completed_at) || null,
  };
}

function normalizeTrainingSignalShard(value: unknown): TrainingSignalShard {
  const raw = jsonRecord(value);
  const aggregate = jsonRecord(raw.aggregate ?? raw.aggregate_statistics);
  const eventCount = finiteNumber(raw.event_count ?? raw.observed_count, 0);
  return {
    id: jsonString(raw.id),
    run_id: jsonString(raw.run_id),
    segment_id: jsonString(raw.segment_id ?? raw.direct_run_segment_id ?? raw.trial_segment_id) || null,
    direct_run_segment_id: jsonString(raw.direct_run_segment_id) || null,
    trial_segment_id: jsonString(raw.trial_segment_id) || null,
    reward_system_revision_id: jsonString(raw.reward_system_revision_id) || undefined,
    protocol_revision_id: jsonString(raw.protocol_revision_id) || undefined,
    capability_id: jsonString(raw.capability_id) || undefined,
    boundary_index: finiteNumber(raw.boundary_index, 0),
    boundary_value: optionalNumber(raw.boundary_value),
    boundary_unit: jsonString(raw.boundary_unit) || null,
    status: jsonString(raw.status, raw.sealed === true ? "sealed" : "open"),
    capture_fidelity: jsonString(raw.capture_fidelity, "unavailable"),
    observed_count: eventCount,
    retained_count: finiteNumber(raw.retained_count ?? raw.event_count, eventCount),
    core_count: optionalNumber(raw.core_count ?? aggregate.uniform_core_count) ?? undefined,
    diagnostic_count: optionalNumber(raw.diagnostic_count ?? aggregate.diagnostic_count) ?? undefined,
    aggregate_statistics: aggregate as Record<string, number | null>,
    dataset_identity: jsonRecord(raw.dataset_identity),
    producer_model_hash: jsonString(raw.producer_model_hash) || null,
    checkpoint_hash: jsonString(raw.checkpoint_hash) || null,
    runtime_identity: jsonRecord(raw.runtime_identity),
    retained_set_hash: jsonString(raw.retained_set_hash) || null,
    trace_hash: jsonString(raw.trace_hash) || null,
    manifest_path: jsonString(raw.manifest_path ?? raw.storage_path) || null,
    sealed_at: raw.sealed === true ? jsonString(raw.created_at) || null : null,
  };
}

function normalizeRewardIntegrityComparison(value: unknown): RewardIntegrityComparison {
  const raw = jsonRecord(value);
  const explicit = Array.isArray(raw.metrics) ? raw.metrics : [];
  const deltaMap = jsonRecord(raw.metric_deltas);
  const metrics = explicit.length ? explicit.map((item) => {
    const metric = jsonRecord(item);
    return {
      name: jsonString(metric.name),
      base_value: optionalNumber(metric.base_value),
      candidate_value: optionalNumber(metric.candidate_value),
      raw_delta: optionalNumber(metric.raw_delta),
      favorable_delta: optionalNumber(metric.favorable_delta) ?? undefined,
      direction: jsonString(metric.direction) as "maximize" | "minimize" || undefined,
    };
  }) : Object.entries(deltaMap).map(([name, delta]) => ({
    name,
    base_value: null,
    candidate_value: null,
    raw_delta: optionalNumber(delta),
  }));
  const pairing = jsonString(raw.comparison_kind ?? raw.pairing, "aggregate_only");
  const normalizedPairing = pairing === "exact" ? "paired_snapshot" : pairing;
  const pairsRaw = Array.isArray(raw.pairs) ? raw.pairs : [];
  return {
    base_audit_id: jsonString(raw.base_audit_id ?? raw.left_audit_id),
    candidate_audit_id: jsonString(raw.candidate_audit_id ?? raw.right_audit_id),
    compatible: raw.compatible !== false,
    comparison_kind: normalizedPairing,
    pairing_reason: jsonString(raw.pairing_reason, normalizedPairing === "matched_input"
      ? "Evidence shares a stable input identity only; this comparison is distributional and non-causal."
      : normalizedPairing === "paired_snapshot"
        ? "Evidence shares the exact immutable snapshot identity."
        : "Only aggregate metrics are available; no sample pairs can be returned."),
    compatibility_reasons: jsonStrings(raw.compatibility_reasons),
    shared_record_count: finiteNumber(raw.shared_record_count ?? raw.shared_snapshot_count, 0),
    shared_snapshot_count: finiteNumber(raw.shared_snapshot_count, 0),
    unmatched_base: finiteNumber(raw.unmatched_base ?? raw.unmatched_left, 0),
    unmatched_candidate: finiteNumber(raw.unmatched_candidate ?? raw.unmatched_right, 0),
    pairs: pairsRaw.map((item) => {
      const pair = jsonRecord(item);
      const base = normalizeRewardIntegritySample(pair.base ?? pair.left);
      const candidate = normalizeRewardIntegritySample(pair.candidate ?? pair.right);
      return {
        id: jsonString(pair.id, `${base.snapshot_id}:${candidate.snapshot_id}`),
        pairing: jsonString(pair.pairing, normalizedPairing),
        record_id: jsonString(pair.record_id, base.record.record_id),
        snapshot_id: jsonString(pair.snapshot_id) || null,
        base_snapshot_id: jsonString(pair.base_snapshot_id ?? pair.left_snapshot_id, base.snapshot_id),
        candidate_snapshot_id: jsonString(pair.candidate_snapshot_id ?? pair.right_snapshot_id, candidate.snapshot_id),
        same_output: pair.same_output === true,
        base,
        candidate,
      };
    }),
    pair_total: finiteNumber(raw.pair_total, pairsRaw.length),
    limit: finiteNumber(raw.limit, Math.max(1, pairsRaw.length || 100)),
    offset: finiteNumber(raw.offset, 0),
    metrics,
  };
}

function normalizeVerifierDecision(value: unknown): VerifierQualificationDecision {
  const raw = jsonRecord(value);
  const evidence = jsonRecord(raw.evidence);
  return {
    id: jsonString(raw.id),
    calibration_id: jsonString(raw.calibration_id),
    profile_revision_id: jsonString(raw.profile_revision_id ?? raw.verifier_revision_id),
    qualification_profile_revision_id: jsonString(raw.qualification_profile_revision_id),
    decision: jsonString(raw.decision, "warn") as "pass" | "warn" | "fail",
    scope: jsonString(raw.scope, "development"),
    reasons: jsonStrings(raw.reasons),
    evidence_count: typeof raw.evidence_count === "number" ? raw.evidence_count : typeof evidence.record_count === "number" ? evidence.record_count : undefined,
    override: raw.override === true,
    override_note: jsonString(raw.override_note) || null,
    created_at: jsonString(raw.created_at) || null,
  };
}

function normalizeVerifierMetric(value: unknown): VerifierCalibrationMetric {
  const raw = jsonRecord(value);
  return {
    id: jsonString(raw.id) || undefined,
    calibration_id: jsonString(raw.calibration_id) || undefined,
    name: jsonString(raw.name),
    value: typeof raw.value === "number" ? raw.value : null,
    direction: jsonString(raw.direction) as "maximize" | "minimize" || undefined,
    lower_ci: typeof raw.lower_ci === "number" ? raw.lower_ci : typeof raw.ci_low === "number" ? raw.ci_low : null,
    upper_ci: typeof raw.upper_ci === "number" ? raw.upper_ci : typeof raw.ci_high === "number" ? raw.ci_high : null,
    record_count: typeof raw.record_count === "number" ? raw.record_count : undefined,
    subgroup: jsonString(raw.subgroup) || null,
    split: jsonString(raw.split ?? raw.partition) || null,
    available: raw.available !== false,
    reason: jsonString(raw.reason ?? raw.missing_reason) || null,
    details: jsonRecord(raw.details ?? raw.metadata),
  };
}

function normalizeVerifierCalibration(value: unknown): VerifierCalibration {
  const raw = jsonRecord(value);
  const decisions = Array.isArray(raw.decisions) ? raw.decisions.map(normalizeVerifierDecision) : [];
  const metrics = Array.isArray(raw.metrics) ? raw.metrics.map(normalizeVerifierMetric) : [];
  const total = typeof raw.total_records === "number" ? raw.total_records : undefined;
  const processed = Number(raw.processed_records ?? 0);
  return {
    id: jsonString(raw.id),
    profile_revision_id: jsonString(raw.profile_revision_id ?? raw.verifier_revision_id),
    profile_revision: raw.profile_revision ? normalizeVerifierRevision(raw.profile_revision) : null,
    source_kind: jsonString(raw.source_kind) as VerifierCalibration["source_kind"],
    source_revision_id: jsonString(raw.source_revision_id),
    source_purpose: jsonString(raw.source_purpose) || null,
    source_hash: jsonString(raw.source_hash) || null,
    source_name: jsonString(raw.source_name) || null,
    protocol_revision_id: jsonString(raw.protocol_revision_id),
    qualification_profile_revision_id: jsonString(raw.qualification_profile_revision_id),
    status: jsonString(raw.status, "queued"),
    stage: jsonString(raw.stage) || null,
    processed_records: processed,
    total_records: total,
    progress_percent: typeof raw.progress_percent === "number" ? raw.progress_percent : total ? processed / total * 100 : 0,
    primary_metric: raw.primary_metric ? normalizeVerifierMetric(raw.primary_metric) : metrics.find((item) => item.details?.primary === true) ?? null,
    qualification: decisions.at(-1) ?? (raw.qualification ? normalizeVerifierDecision(raw.qualification) : null),
    decisions,
    metrics,
    work_item_id: jsonString(raw.work_item_id) || null,
    evidence_hash: jsonString(raw.evidence_hash ?? raw.manifest_hash) || null,
    runtime_hash: jsonString(raw.runtime_hash ?? raw.runtime_identity_hash) || null,
    request_hash: jsonString(raw.request_hash ?? raw.reuse_key) || null,
    runtime_compatibility: raw.runtime_compatibility ? jsonRecord(raw.runtime_compatibility) : null,
    error: jsonString(raw.error) || null,
    created_at: jsonString(raw.created_at) || null,
    completed_at: jsonString(raw.completed_at) || null,
  };
}

function normalizeVerifierSample(value: unknown): VerifierCalibrationSample {
  const raw = jsonRecord(value);
  const reference = jsonRecord(raw.reference);
  const metadata = jsonRecord(raw.metadata);
  return {
    id: jsonString(raw.id, `${jsonString(raw.calibration_id)}:${Number(raw.ordinal ?? 0)}`),
    calibration_id: jsonString(raw.calibration_id),
    record_id: jsonString(raw.record_id),
    record_hash: jsonString(raw.record_hash) || null,
    group_id: jsonString(raw.group_id) || null,
    split: jsonString(raw.split ?? raw.partition, "calibration"),
    task_type: jsonString(raw.task_type) || undefined,
    orientation: jsonString(raw.orientation) || null,
    perturbation: jsonString(raw.perturbation ?? raw.probe_kind) || null,
    repeat_index: Number(raw.repeat_index ?? 0),
    seed: typeof raw.seed === "number" ? raw.seed : null,
    expected: raw.expected ?? reference.expected ?? reference.label ?? reference.value ?? reference,
    input: raw.input ?? reference.input ?? metadata.input ?? metadata.record ?? reference,
    observation: jsonRecord(raw.observation) as VerifierObservation,
    agreement: typeof raw.agreement === "boolean" ? raw.agreement : undefined,
    subgroup: jsonRecord(raw.subgroup ?? metadata.subgroup) as Record<string, string>,
  };
}

function normalizeVerifierComparison(value: unknown): VerifierCalibrationComparison {
  const raw = jsonRecord(value);
  const deltas = Array.isArray(raw.metrics) ? raw.metrics : Array.isArray(raw.metric_deltas) ? raw.metric_deltas : [];
  return {
    base_calibration_id: jsonString(raw.base_calibration_id),
    candidate_calibration_id: jsonString(raw.candidate_calibration_id),
    compatible: raw.compatible === true,
    compatibility_reasons: jsonStrings(raw.compatibility_reasons),
    task_type: jsonString(raw.task_type) || undefined,
    metrics: deltas.map((item) => {
      const metric = jsonRecord(item);
      const delta = typeof metric.raw_delta === "number" ? metric.raw_delta : typeof metric.delta === "number" ? metric.delta : null;
      const direction = jsonString(metric.direction, "maximize") as "maximize" | "minimize";
      return {
        name: jsonString(metric.name),
        base_value: typeof metric.base_value === "number" ? metric.base_value : typeof metric.base === "number" ? metric.base : null,
        candidate_value: typeof metric.candidate_value === "number" ? metric.candidate_value : typeof metric.candidate === "number" ? metric.candidate : null,
        raw_delta: delta,
        favorable_delta: typeof metric.favorable_delta === "number" ? metric.favorable_delta : delta === null ? null : direction === "minimize" ? -delta : delta,
        direction,
      };
    }),
    sample_counts: jsonRecord(raw.sample_counts) as Record<string, number>,
  };
}

function verifierProfileDefinition(payload: Record<string, unknown>): Record<string, unknown> {
  const definition = { ...payload };
  const implementationId = definition.implementation_id;
  const modelRevision = definition.model_revision;
  const endpointType = definition.endpoint_type;
  delete definition.name;
  delete definition.description;
  delete definition.implementation_id;
  delete definition.model_revision;
  delete definition.endpoint_type;
  const family = jsonString(payload.family, "deterministic");
  const reference = family === "reward_model" ? modelRevision : implementationId;
  const rewardContract = jsonRecord(definition.reward_contract);
  if (Object.keys(rewardContract).length) {
    definition.reward_contract = {
      ...rewardContract,
      tie_policy: rewardContract.tie_policy === "allow" ? "tie" : rewardContract.tie_policy,
      error_behavior: rewardContract.error_behavior === "propagate" ? "error" : rewardContract.error_behavior,
    };
  }
  return {
    ...definition,
    implementation: {
      kind: family === "reward_model" ? "artifact" : family === "chain" ? "chain" : "builtin",
      ref: jsonString(reference, family === "chain" ? "ordered_chain" : ""),
    },
    configuration: {
      ...jsonRecord(payload.configuration),
      model_revision: modelRevision,
      endpoint_type: endpointType,
    },
  };
}

function normalizeVerifierAlias(value: unknown): VerifierAlias {
  const raw = jsonRecord(value);
  const alias = jsonRecord(raw.alias && typeof raw.alias === "object" ? raw.alias : raw);
  return {
    id: jsonString(alias.id, `${jsonString(alias.profile_id)}:${jsonString(alias.alias)}:${jsonString(alias.revision_id)}`),
    profile_id: jsonString(alias.profile_id),
    alias: jsonString(alias.alias),
    profile_revision_id: jsonString(alias.profile_revision_id ?? alias.revision_id),
    previous_revision_id: jsonString(alias.previous_revision_id) || null,
    note: jsonString(alias.note) || null,
    override: alias.override === true,
    created_at: jsonString(alias.created_at ?? alias.updated_at) || null,
  };
}

export type TrainingOutcomeAssessment = {
  id: string;
  proof_run_id: string;
  scenario_revision_id: string;
  profile_id: string;
  status: "queued" | "running" | "improved" | "regressed" | "mixed" | "no_clear_change" | "incomplete_evidence" | "technical_failure" | "failed" | "cancelled";
  stage?: string;
  progress?: Record<string, unknown>;
  technical_status: string;
  quality_status: string;
  base_evaluation_id?: string | null;
  candidate_evaluation_id?: string | null;
  resource_projection: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  summary: Record<string, unknown>;
  work_item_id?: string | null;
};

export type GuidedAction = {
  id: string;
  label: string;
  href?: string | null;
  method?: string | null;
  payload?: Record<string, unknown>;
  requires_confirmation?: boolean;
  tone?: string;
};

export type ActionableGuidance = {
  context_kind: string;
  context_id: string;
  display_status: string;
  summary: string;
  primary_action: GuidedAction;
  secondary_actions: GuidedAction[];
  blockers: string[];
  technical_details: Record<string, unknown>;
};

export type OutcomePreparation = {
  status: string;
  assessment?: TrainingOutcomeAssessment;
  base_evaluation?: Record<string, unknown>;
  proof_evaluation?: Record<string, unknown>;
  work_item_id?: string | null;
  guidance: ActionableGuidance;
};

export type StudyLaunchPlan = {
  protocol_revision_id: string;
  arm_count: number;
  seed_count: number;
  run_count: number;
  estimated_seconds_low?: number | null;
  estimated_seconds_high?: number | null;
  estimated_storage_bytes?: number | null;
  blockers: string[];
  work_item_id?: string | null;
};

export type GroundingGenerationPreview = {
  profile_revision_id: string;
  preset: "quick" | "standard" | "thorough";
  candidate_limit: number;
  preview_items: Record<string, unknown>[];
  teacher: Record<string, unknown>;
  verifier: Record<string, unknown>;
  request_estimate: Record<string, unknown>;
  blockers: string[];
};

export type EnvironmentPermissionSummary = {
  local_files: boolean;
  local_sqlite: boolean;
  loopback_services: boolean;
  external_writes: boolean;
  max_steps: number;
  timeout_seconds: number;
  notes: string[];
};

export type AdaptationStudy = {
  id: string;
  name: string;
  description?: string | null;
  status: string;
  latest_protocol_revision_id?: string | null;
  created_at: string;
  updated_at: string;
};

export type GroundedGenerationBatch = {
  id: string;
  profile_revision_id: string;
  status: string;
  stage: string;
  intended_destination: string;
  candidate_count: number;
  accepted_count: number;
  rejected_count: number;
  coverage: Record<string, unknown>;
  work_item_id?: string | null;
};

export type SpecializedTaskDescriptor = {
  id: string;
  label: string;
  task_kind: string;
  modality: string;
  canonical_schema: string;
  trainer_mode: string;
  metrics: string[];
  available: boolean;
  unavailable_reason?: string | null;
};

export type AgentEnvironment = {
  id: string;
  name: string;
  description?: string | null;
  latest_revision_id?: string | null;
  archived: boolean;
  created_at: string;
  updated_at: string;
};

export type AgentEpisode = {
  id: string;
  suite_revision_id: string;
  suite_item_id: string;
  subject_type: string;
  subject_ref: string;
  subject_hash: string;
  seed: number;
  status: string;
  metrics: Record<string, unknown>;
  work_item_id?: string | null;
};

// ----- V17 readiness, repair overlays, and support ----------------------

export type SetupRemediation = {
  id: string;
  label: string;
  description: string;
  automatic: boolean;
  action: string;
  blocker?: string | null;
};

export type DistributionCapability = {
  platform: string;
  architecture: string;
  execution_surfaces: string[];
  desktop_package?: string | null;
  desktop_status: "supported" | "candidate" | "unavailable" | string;
  signature_state: string;
  runtime_version: string;
  supported_backends: string[];
  unavailable_reason?: string | null;
};

export type WorkstationReadiness = {
  id: string;
  status: "ready" | "attention" | "blocked" | string;
  display_status: string;
  summary: string;
  checks: Array<{
    id: string;
    label: string;
    status: string;
    summary: string;
    technical?: Record<string, unknown>;
  }>;
  remediations: SetupRemediation[];
  capability: DistributionCapability;
  content_hash: string;
  created_at: string;
  primary_action?: { id: string; label: string; href?: string } | null;
};

export type DatasetIssue = {
  id: string;
  session_id: string;
  ordinal: number;
  record_id?: string | null;
  source_index?: number | null;
  code: string;
  category: string;
  severity: string;
  field_path?: string | null;
  message: string;
  suggested_actions: string[];
  evidence: Record<string, unknown>;
};

export type DatasetRepairAction = {
  ordinal?: number;
  issue_code?: string;
  action_kind: string;
  reason: string;
  record_id?: string | null;
  source_index?: number | null;
  field_path?: string | null;
  value?: unknown;
};

export type DatasetRepairSession = {
  id: string;
  source_id?: string | null;
  inspection_id?: string | null;
  dataset_version_id?: string | null;
  source_uri: string;
  source_fingerprint: string;
  scenario_revision_id?: string | null;
  status: string;
  stage: string;
  progress: Record<string, number>;
  issue_summary: Record<string, unknown>;
  latest_plan_revision_id?: string | null;
  latest_preview_id?: string | null;
  published_repair_revision_id?: string | null;
  work_item_id?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
};

export type DatasetRepairPlanRevision = {
  id: string;
  session_id: string;
  revision_number: number;
  source_fingerprint: string;
  content_hash: string;
  actions: DatasetRepairAction[];
  created_at: string;
};

export type DatasetRepairPreview = {
  id: string;
  session_id: string;
  plan_revision_id: string;
  source_fingerprint: string;
  status: string;
  exact: boolean;
  counts: Record<string, number>;
  issue_counts: Record<string, number>;
  split_impact: Record<string, unknown>;
  sample: Array<Record<string, unknown>>;
  content_hash?: string | null;
  storage_path?: string | null;
  work_item_id?: string | null;
  error?: string | null;
};

export type SupportBundlePreview = {
  categories: string[];
  included: Array<{ id: string; description: string }>;
  excluded_by_default: string[];
  redaction_policy: string;
};

export type SupportBundle = {
  id: string;
  status: string;
  categories: string[];
  preview: SupportBundlePreview;
  manifest: Record<string, unknown>;
  storage_path?: string | null;
  content_hash?: string | null;
  work_item_id?: string | null;
  error?: string | null;
  created_at: string;
  completed_at?: string | null;
};

// ----- V18 guided training plan and capacity coach ---------------------

export type TrainingResourceForecast = {
  download_bytes?: number | null;
  scratch_bytes?: number | null;
  checkpoint_bytes?: number | null;
  peak_memory_bytes?: number | null;
  proof_seconds_range?: [number, number] | null;
  full_run_seconds_range?: [number, number] | null;
  provenance: Record<string, string>;
  confidence: string;
};

export type TrainingPlanReason = {
  code: string;
  summary: string;
  detail: string;
  kind: string;
};

export type TrainingPlanRevision = {
  id: string;
  plan_id: string;
  revision_number: number;
  status: "draft" | "resolved" | string;
  content_hash: string;
  profile_id: string;
  profile_version: string;
  dataset_version_id: string;
  scenario_revision_id?: string | null;
  trainer_mode: string;
  backend: string;
  model_id: string;
  model_revision?: string | null;
  resolved_model_commit?: string | null;
  definition: Record<string, unknown>;
  reasons: TrainingPlanReason[];
  forecast: TrainingResourceForecast;
  compute_shape_hash: string;
  runtime_hash: string;
  runtime_profile_revision_id?: string | null;
  training_path_revision_id?: string | null;
  training_path_certification_id?: string | null;
  created_at: string;
};

// ----- V19/V20 managed accelerator runtimes ---------------------------

export type ManagedRuntimeCapability = {
  accelerator_family: "rocm" | "cuda" | "native" | string;
  available: boolean;
  status: string;
  summary: string;
  runtime_revision_id?: string | null;
  qualification_id?: string | null;
  supported_trainers: string[];
  unavailable_reason?: string | null;
};

export type ManagedRuntimeRevision = {
  id: string;
  profile_id: string;
  content_hash: string;
  engine: string;
  base_image?: string | null;
  base_image_digest?: string | null;
  derived_image_ref?: string | null;
  download_bytes?: number | null;
  installed_bytes?: number | null;
  trainer_contracts: string[];
};

export type RuntimePreparation = {
  id: string;
  runtime_revision_id: string;
  status: string;
  stage: string;
  progress: Record<string, unknown>;
  work_item_id?: string | null;
  error?: string | null;
};

export type RuntimeQualification = {
  id: string;
  runtime_revision_id: string;
  status: string;
  stage: string;
  progress: Record<string, unknown>;
  work_item_id?: string | null;
  error?: string | null;
  steps: Array<{ step_id: string; label: string; status: string }>;
};

export type ManagedRuntimeView = {
  id: string;
  name: string;
  accelerator_family: string;
  description?: string | null;
  revision?: ManagedRuntimeRevision | null;
  qualification?: RuntimeQualification | null;
  preparations: RuntimePreparation[];
};

export type AcceleratorAvailability = {
  accelerator_family: string;
  state: "idle" | "busy" | "unknown" | string;
  sampled_at: string;
  utilization_percent?: number | null;
  reason?: string | null;
  owners: Array<{ pid?: number | null; executable: string; elapsed_seconds?: number | null }>;
};

export type TrainingPathCapability = {
  path_revision_id: string;
  profile_id: string;
  label: string;
  scenario_revision_id?: string | null;
  trainer_mode: string;
  model_id: string;
  runtime_family: string;
  state: "runtime_ready" | "path_verified" | "verification_in_progress" | "plan_ready" | "beta_qualified" | "unavailable" | string;
  display_status: string;
  summary: string;
  runtime_revision_id?: string | null;
  runtime_qualification_id?: string | null;
  certification_id?: string | null;
  blocker?: string | null;
  recovery_action?: { action: string; label: string; reason: string; enabled: boolean } | null;
};

export type TrainingPathCertificationMatrix = {
  runtime_family: string;
  runtime_ready: boolean;
  beta_qualified: boolean;
  paths: TrainingPathCapability[];
  recommended_path_revision_id?: string | null;
};

export type TrainingPathCertification = {
  id: string;
  path_revision_id: string;
  runtime_revision_id: string;
  status: string;
  stage: string;
  progress: Record<string, unknown>;
  work_item_id?: string | null;
  error?: string | null;
};

export type TrainingPlanRecommendation = {
  plan: { id: string; status: string; latest_revision_id?: string | null };
  revision: TrainingPlanRevision;
  alternatives: Array<{ model_id: string; label: string; estimated_memory_gb?: number | null; reason_not_selected: string }>;
  summary: string;
  primary_action: { id: string; label: string; plan_revision_id: string };
};

export type ModelPreparation = {
  id: string;
  plan_revision_id: string;
  status: string;
  resolved_commit?: string | null;
  size_bytes?: number | null;
  progress: Record<string, unknown>;
  work_item_id?: string | null;
  error?: string | null;
};

export type TrainingCapacityCheck = {
  id: string;
  plan_revision_id: string;
  model_preparation_id?: string | null;
  status: string;
  stage: string;
  selected_adjustment: Record<string, unknown>;
  forecast: TrainingResourceForecast;
  progress: Record<string, unknown>;
  primary_remedy: { id?: string; label?: string; reason?: string };
  work_item_id?: string | null;
  error?: string | null;
};

export type TrainingPlanReadiness = {
  plan_revision_id: string;
  status: string;
  display_status: string;
  summary: string;
  model_preparation?: ModelPreparation | null;
  capacity_check?: TrainingCapacityCheck | null;
  blockers: Array<{ code: string; summary: string }>;
  primary_action: { id: string; label: string };
  notices?: Array<{ code: string; summary: string }>;
};

export const api = {
  health: () => request<{ ok: boolean }>("/health"),
  versionInfo: () => request<VersionInfo>("/version"),
  backendInfo: () => request<BackendInfo>("/backend"),
  workspaceInfo: () => request<WorkspaceInfo>("/workspace"),
  workstationReadiness: () => request<WorkstationReadiness>("/setup/readiness"),
  applySetupRemediation: (action: string) =>
    request<WorkstationReadiness>(`/setup/remediations/${encodeURIComponent(action)}`, { method: "POST" }),
  distributionCapability: () => request<DistributionCapability>("/release/capability"),
  releaseStatus: () => request<{ status: string; current_version: string; latest_version?: string | null; update_available: boolean; automatic_update: boolean; message: string; release_url?: string | null }>("/release/status"),
  trainingPlanCapabilities: () => request<Record<string, unknown>>("/training-plan-capabilities"),
  managedRuntimeCapabilities: () => request<PaginatedResponse<ManagedRuntimeCapability>>("/runtime-capabilities"),
  managedRuntimes: () => request<PaginatedResponse<ManagedRuntimeView>>("/runtimes"),
  managedRuntime: (identifier: string) => request<Record<string, unknown>>(`/runtimes/${encodeURIComponent(identifier)}`),
  prepareManagedRuntime: (revisionId: string) =>
    request<RuntimePreparation>(`/runtime-revisions/${encodeURIComponent(revisionId)}/prepare`, { method: "POST", body: JSON.stringify({ confirmed: true }) }),
  qualifyManagedRuntime: (revisionId: string) =>
    request<RuntimeQualification>(`/runtime-revisions/${encodeURIComponent(revisionId)}/qualify`, { method: "POST" }),
  runtimePreparation: (preparationId: string) => request<RuntimePreparation>(`/runtime-preparations/${encodeURIComponent(preparationId)}`),
  runtimeQualification: (qualificationId: string) => request<RuntimeQualification>(`/runtime-qualifications/${encodeURIComponent(qualificationId)}`),
  acceleratorAvailability: (family: "rocm" | "cuda") => request<AcceleratorAvailability>(`/accelerator/availability?family=${encodeURIComponent(family)}`),
  trainingPaths: (family: "rocm" | "cuda") =>
    request<TrainingPathCertificationMatrix>(`/runtime/paths?family=${encodeURIComponent(family)}`),
  previewTrainingPathCertification: (pathRevisionId: string, runtimeRevisionId: string) =>
    request<Record<string, unknown>>(`/training-path-revisions/${encodeURIComponent(pathRevisionId)}/certification-preview`, {
      method: "POST",
      body: JSON.stringify({ runtime_profile_revision_id: runtimeRevisionId }),
    }),
  certifyTrainingPath: (pathRevisionId: string, runtimeRevisionId: string) =>
    request<TrainingPathCertification>(`/training-path-revisions/${encodeURIComponent(pathRevisionId)}/certify`, {
      method: "POST",
      body: JSON.stringify({ runtime_profile_revision_id: runtimeRevisionId }),
    }),
  trainingPathCertification: (certificationId: string) =>
    request<TrainingPathCertification>(`/training-path-certifications/${encodeURIComponent(certificationId)}`),
  cancelTrainingPathCertification: (certificationId: string) =>
    request<TrainingPathCertification>(`/training-path-certifications/${encodeURIComponent(certificationId)}/cancel`, { method: "POST" }),
  retryTrainingPathCertification: (certificationId: string, reason: string) =>
    request<TrainingPathCertification>(`/training-path-certifications/${encodeURIComponent(certificationId)}/retry`, { method: "POST", body: JSON.stringify({ reason }) }),
  resumeTrainingPathCertification: (certificationId: string, reason?: string) =>
    request<TrainingPathCertification>(`/training-path-certifications/${encodeURIComponent(certificationId)}/resume`, { method: "POST", body: JSON.stringify({ reason }) }),
  trainingPathCertificationEvidence: (certificationId: string) =>
    request<Record<string, unknown>>(`/training-path-certifications/${encodeURIComponent(certificationId)}/evidence`),
  recommendTrainingPlan: (payload: Record<string, unknown>) =>
    request<TrainingPlanRecommendation>("/training-plans/recommend", { method: "POST", body: JSON.stringify(payload) }),
  trainingPlan: (planId: string) =>
    request<{ plan: Record<string, unknown>; revision?: TrainingPlanRevision | null; readiness?: TrainingPlanReadiness | null }>(`/training-plans/${encodeURIComponent(planId)}`),
  trainingPlanRevision: (revisionId: string) =>
    request<TrainingPlanRevision>(`/training-plan-revisions/${encodeURIComponent(revisionId)}`),
  trainingPlanAlternatives: (revisionId: string) =>
    request<PaginatedResponse<Record<string, unknown>>>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/alternatives`),
  chooseTrainingPlanAlternative: (revisionId: string, modelId: string, reason: string) =>
    request<TrainingPlanRecommendation>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/alternatives`, { method: "POST", body: JSON.stringify({ model_id: modelId, reason }) }),
  confirmTrainingPlan: (revisionId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/confirm`, { method: "POST", body: JSON.stringify(payload) }),
  prepareTrainingPlanModel: (revisionId: string) =>
    request<ModelPreparation>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/prepare`, { method: "POST", body: JSON.stringify({ download_confirmed: true }) }),
  modelPreparation: (preparationId: string) =>
    request<ModelPreparation>(`/model-preparations/${encodeURIComponent(preparationId)}`),
  retryModelPreparation: (preparationId: string, reason: string) =>
    request<{ id: string; status: string; work_item_id: string }>(`/model-preparations/${encodeURIComponent(preparationId)}/retry`, { method: "POST", body: JSON.stringify({ reason }) }),
  createTrainingCapacityCheck: (revisionId: string) =>
    request<TrainingCapacityCheck>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/capacity-check`, { method: "POST" }),
  trainingCapacityCheck: (checkId: string) =>
    request<TrainingCapacityCheck>(`/training-capacity-checks/${encodeURIComponent(checkId)}`),
  retryTrainingCapacityCheck: (checkId: string, reason: string) =>
    request<{ id: string; status: string; work_item_id: string }>(`/training-capacity-checks/${encodeURIComponent(checkId)}/retry`, { method: "POST", body: JSON.stringify({ reason }) }),
  trainingPlanReadiness: (revisionId: string) =>
    request<TrainingPlanReadiness>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/readiness`),
  launchTrainingPlanProof: (revisionId: string, payload: Record<string, unknown> = {}) =>
    request<Record<string, unknown>>(`/training-plan-revisions/${encodeURIComponent(revisionId)}/proof`, { method: "POST", body: JSON.stringify(payload) }),
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
  trainingDatasetVersions: (mode?: TrainingMode) =>
    request<{ items: DatasetVersion[] }>(
      `/train/dataset-versions${mode ? `?mode=${encodeURIComponent(mode)}` : ""}`,
    ),
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
  verifierReliabilityCapabilities: async () => {
    const value = await request<{
      items: VerifierCapabilityDescriptor[];
      qualification_templates?: VerifierQualificationProfileRevision[];
      max_evaluation_candidates?: number;
    }>("/verifier-reliability/capabilities");
    return { ...value, items: (value.items ?? []).map(normalizeVerifierCapability) };
  },
  listVerifierProfiles: (params: { family?: string; modality?: string; taskType?: string; qualification?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.family) search.set("family", params.family);
    if (params.modality) search.set("modality", params.modality);
    if (params.taskType) search.set("task_type", params.taskType);
    if (params.qualification) search.set("qualified_only", String(params.qualification === "pass" || params.qualification === "candidate" || params.qualification === "approved"));
    if (params.q) search.set("q", params.q);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<unknown>>(`/verifier-profiles${qs ? `?${qs}` : ""}`).then((value) => ({ ...value, items: value.items.map(normalizeVerifierProfile) }));
  },
  verifierProfile: (profileId: string) =>
    request<unknown>(`/verifier-profiles/${encodeURIComponent(profileId)}`).then(normalizeVerifierProfile),
  listVerifierProfileRevisions: (profileId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<unknown>>(`/verifier-profiles/${encodeURIComponent(profileId)}/revisions${qs ? `?${qs}` : ""}`).then((value) => ({ ...value, items: value.items.map((item) => normalizeVerifierRevision(item)) }));
  },
  createVerifierProfile: (payload: Record<string, unknown>) => {
    return request<unknown>("/verifier-profiles", { method: "POST", body: JSON.stringify({ name: payload.name, description: payload.description, definition: verifierProfileDefinition(payload) }) }).then(normalizeVerifierProfile);
  },
  reviseVerifierProfile: (profileId: string, payload: Record<string, unknown>) =>
    request<unknown>(`/verifier-profiles/${encodeURIComponent(profileId)}/revisions`, { method: "POST", body: JSON.stringify({ definition: verifierProfileDefinition(payload) }) }).then((value) => normalizeVerifierRevision(jsonRecord(value).revision ?? value)),
  verifierProfileRevision: (revisionId: string) =>
    request<unknown>(`/verifier-profile-revisions/${encodeURIComponent(revisionId)}`).then((value) => normalizeVerifierRevision(jsonRecord(value).revision ?? value)),
  listVerifierProtocols: async (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    const value = await request<PaginatedResponse<unknown>>(`/verifier-calibration-protocols${qs ? `?${qs}` : ""}`);
    const items = await Promise.all(value.items.map(async (item) => {
      const normalized = normalizeVerifierProtocol(item);
      if (!normalized.id) return normalized;
      try {
        return normalizeVerifierProtocol(await request<unknown>(`/verifier-calibration-protocol-revisions/${encodeURIComponent(normalized.id)}`));
      } catch {
        return normalized;
      }
    }));
    return { ...value, items };
  },
  createVerifierProtocol: (payload: Record<string, unknown>) =>
    request<unknown>("/verifier-calibration-protocols", { method: "POST", body: JSON.stringify(payload) }).then(normalizeVerifierProtocol),
  verifierProtocol: (protocolRevisionId: string) =>
    request<unknown>(`/verifier-calibration-protocol-revisions/${encodeURIComponent(protocolRevisionId)}`).then(normalizeVerifierProtocol),
  reviseVerifierProtocol: (protocolId: string, payload: Record<string, unknown>) =>
    request<unknown>(`/verifier-calibration-protocols/${encodeURIComponent(protocolId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }).then(normalizeVerifierProtocol),
  listVerifierQualificationProfiles: async (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    const value = await request<PaginatedResponse<unknown>>(`/verifier-qualification-profiles${qs ? `?${qs}` : ""}`);
    const items = await Promise.all(value.items.map(async (item) => {
      const normalized = normalizeVerifierQualificationProfile(item);
      if (!normalized.id) return normalized;
      try {
        return normalizeVerifierQualificationProfile(await request<unknown>(`/verifier-qualification-profile-revisions/${encodeURIComponent(normalized.id)}`));
      } catch {
        return normalized;
      }
    }));
    return { ...value, items };
  },
  createVerifierQualificationProfile: (payload: Record<string, unknown>) =>
    request<unknown>("/verifier-qualification-profiles", { method: "POST", body: JSON.stringify(payload) }).then(normalizeVerifierQualificationProfile),
  verifierQualificationProfile: (profileRevisionId: string) =>
    request<unknown>(`/verifier-qualification-profile-revisions/${encodeURIComponent(profileRevisionId)}`).then(normalizeVerifierQualificationProfile),
  reviseVerifierQualificationProfile: (profileId: string, payload: Record<string, unknown>) =>
    request<unknown>(`/verifier-qualification-profiles/${encodeURIComponent(profileId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }).then(normalizeVerifierQualificationProfile),
  listVerifierCalibrations: (params: { profileRevisionId?: string; status?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.profileRevisionId) search.set("verifier_profile_revision_id", params.profileRevisionId);
    if (params.status) search.set("status", params.status);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<unknown>>(`/verifier-calibrations${qs ? `?${qs}` : ""}`).then((value) => ({ ...value, items: value.items.map(normalizeVerifierCalibration) }));
  },
  createVerifierCalibration: (payload: Record<string, unknown>) => {
    const normalized: Record<string, unknown> = {
      ...payload,
      verifier_profile_revision_id: payload.verifier_profile_revision_id ?? payload.profile_revision_id,
      confirmation: payload.confirmation ?? payload.confirmation_requested,
      source_kind: payload.source_kind === "label_set_revision" ? "label_set" : payload.source_kind === "benchmark_suite_revision" ? "benchmark_suite" : payload.source_kind,
    };
    delete normalized.profile_revision_id;
    delete normalized.confirmation_requested;
    return request<unknown>("/verifier-calibrations", { method: "POST", body: JSON.stringify(normalized) }).then(normalizeVerifierCalibration);
  },
  verifierCalibration: (calibrationId: string) =>
    request<unknown>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}`).then(normalizeVerifierCalibration),
  verifierCalibrationSamples: (calibrationId: string, params: { split?: string; outcome?: string; perturbation?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.split) search.set("partition", params.split);
    if (params.outcome) search.set("outcome", params.outcome);
    if (params.perturbation) search.set("perturbation", params.perturbation);
    if (params.q) search.set("q", params.q);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<unknown>>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}/samples${qs ? `?${qs}` : ""}`).then((value) => ({ ...value, items: value.items.map(normalizeVerifierSample) }));
  },
  verifierCalibrationMetrics: (calibrationId: string) =>
    request<PaginatedResponse<unknown>>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}/metrics`).then((value) => ({ ...value, items: value.items.map(normalizeVerifierMetric) })),
  cancelVerifierCalibration: (calibrationId: string) =>
    request<unknown>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}/cancel`, { method: "POST" }).then(normalizeVerifierCalibration),
  retryVerifierCalibration: (calibrationId: string) =>
    request<unknown>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}/retry`, { method: "POST" }).then(normalizeVerifierCalibration),
  verifyVerifierCalibration: (calibrationId: string) =>
    request<{ calibration_id: string; valid: boolean; errors?: string[]; checksums?: Record<string, string> }>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}/verify`, { method: "POST" }),
  qualifyVerifierCalibration: (calibrationId: string, payload: { scope: "development" | "operational" | "confirmation"; overrideNote?: string }) =>
    request<unknown>(`/verifier-calibrations/${encodeURIComponent(calibrationId)}/qualify`, { method: "POST", body: JSON.stringify({ scope: payload.scope, override_note: payload.overrideNote }) }).then(normalizeVerifierDecision),
  compareVerifierCalibrations: (baseCalibrationId: string, candidateCalibrationId: string) => {
    const search = new URLSearchParams({ base_id: baseCalibrationId, candidate_id: candidateCalibrationId });
    return request<unknown>(`/verifier-calibrations/compare?${search.toString()}`).then(normalizeVerifierComparison);
  },
  listVerifierQualificationDecisions: (params: { profileRevisionId?: string; calibrationId?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.profileRevisionId) search.set("verifier_profile_revision_id", params.profileRevisionId);
    if (params.calibrationId) search.set("calibration_id", params.calibrationId);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<unknown>>(`/verifier-qualifications${qs ? `?${qs}` : ""}`).then((value) => ({ ...value, items: value.items.map(normalizeVerifierDecision) }));
  },
  promoteVerifierRevision: (revisionId: string, payload: { alias: "candidate" | "approved"; note?: string; override?: boolean }) =>
    request<unknown>(`/verifier-profile-revisions/${encodeURIComponent(revisionId)}/promote`, { method: "POST", body: JSON.stringify({ alias: payload.alias, override_note: payload.note }) }).then(normalizeVerifierAlias),
  verifierAliasHistory: (profileId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<unknown>>(`/verifier-profiles/${encodeURIComponent(profileId)}/aliases${qs ? `?${qs}` : ""}`).then((value) => ({ ...value, items: value.items.map(normalizeVerifierAlias) }));
  },
  verifierRevisionUsage: (revisionId: string) =>
    request<PaginatedResponse<unknown>>(`/verifier-profile-revisions/${encodeURIComponent(revisionId)}/usage`).then((value) => ({
      ...value,
      items: value.items.map((item) => {
        const raw = jsonRecord(item);
        return {
          id: jsonString(raw.id ?? raw.domain_id),
          kind: jsonString(raw.kind ?? raw.domain_kind),
          role: jsonString(raw.role) || null,
          label: jsonString(raw.label ?? raw.domain_id) || null,
          created_at: jsonString(raw.created_at) || null,
        };
      }),
      total: value.total ?? value.items.length,
    })),
  verifierRuntimeCompatibility: (revisionId: string) =>
    request<VerifierRuntimeCompatibility>(`/verifier-profile-revisions/${encodeURIComponent(revisionId)}/runtime-compatibility`),

  // ----- reward integrity and training signals ------------------------
  rewardIntegrityCapabilities: () =>
    request<unknown>("/reward-integrity-capabilities").then((value) => {
      const raw = jsonRecord(value);
      const items = Array.isArray(raw.items) ? raw.items.map(normalizeTrainingSignalCapability) : [];
      const protocolItems = raw.default_protocols ?? raw.protocols;
      const profileItems = raw.default_profiles ?? raw.integrity_profiles;
      return {
        items,
        default_protocols: Array.isArray(protocolItems) ? protocolItems.map(normalizeRewardAuditProtocol) : [],
        default_profiles: Array.isArray(profileItems) ? profileItems.map(normalizeRewardIntegrityProfile) : [],
      };
    }),
  listRewardSystems: (params: { trainerMode?: string; backendFamily?: string; modality?: string; qualifiedOnly?: boolean; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.trainerMode) search.set("trainer_mode", params.trainerMode);
    if (params.backendFamily) search.set("backend_family", params.backendFamily);
    if (params.modality) search.set("modality", params.modality);
    if (params.qualifiedOnly !== undefined) search.set("qualified_only", String(params.qualifiedOnly));
    if (params.q) search.set("q", params.q);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/reward-systems${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeRewardSystem));
  },
  rewardSystem: (systemId: string) =>
    request<unknown>(`/reward-systems/${encodeURIComponent(systemId)}`).then(normalizeRewardSystem),
  rewardSystemRevision: (revisionId: string) =>
    request<unknown>(`/reward-system-revisions/${encodeURIComponent(revisionId)}`).then(normalizeRewardSystemRevision),
  createRewardSystem: (payload: Record<string, unknown>) =>
    request<unknown>("/reward-systems", { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardSystem),
  validateRewardSystem: (payload: Record<string, unknown>) =>
    request<{ valid: boolean; blockers: string[] }>("/reward-systems/validate", { method: "POST", body: JSON.stringify(payload) }),
  reviseRewardSystem: (systemId: string, payload: Record<string, unknown>) =>
    request<unknown>(`/reward-systems/${encodeURIComponent(systemId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardSystemRevision),
  listRewardAuditProtocols: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/reward-audit-protocols${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeRewardAuditProtocol));
  },
  rewardAuditProtocol: (revisionId: string) =>
    request<unknown>(`/reward-audit-protocol-revisions/${encodeURIComponent(revisionId)}`).then(normalizeRewardAuditProtocol),
  createRewardAuditProtocol: (payload: Record<string, unknown>) =>
    request<unknown>("/reward-audit-protocols", { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardAuditProtocol),
  reviseRewardAuditProtocol: (protocolId: string, payload: Record<string, unknown>) =>
    request<unknown>(`/reward-audit-protocols/${encodeURIComponent(protocolId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardAuditProtocol),
  listRewardIntegrityProfiles: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/reward-integrity-profiles${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeRewardIntegrityProfile));
  },
  rewardIntegrityProfile: (revisionId: string) =>
    request<unknown>(`/reward-integrity-profile-revisions/${encodeURIComponent(revisionId)}`).then(normalizeRewardIntegrityProfile),
  createRewardIntegrityProfile: (payload: Record<string, unknown>) =>
    request<unknown>("/reward-integrity-profiles", { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardIntegrityProfile),
  reviseRewardIntegrityProfile: (profileId: string, payload: Record<string, unknown>) =>
    request<unknown>(`/reward-integrity-profiles/${encodeURIComponent(profileId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardIntegrityProfile),
  resolveRewardIntegrityBinding: (payload: RewardIntegrityBinding & { trainer_mode: string; backend_family?: string; total_budget?: number; budget_unit?: string }) =>
    request<ResolvedRewardBinding>("/reward-integrity-bindings/resolve", { method: "POST", body: JSON.stringify(payload) }),
  listTrainingSignalShards: (params: { runId?: string; status?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.runId) search.set("run_id", params.runId);
    if (params.status) search.set("status", params.status);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/training-signals${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeTrainingSignalShard));
  },
  trainingSignalShard: (shardId: string) =>
    request<unknown>(`/training-signals/${encodeURIComponent(shardId)}`).then(normalizeTrainingSignalShard),
  runTrainingSignalShards: (runId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/runs/${encodeURIComponent(runId)}/training-signals${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeTrainingSignalShard));
  },
  verifyTrainingSignalShard: (shardId: string) =>
    request<{ shard_id: string; valid: boolean; errors?: string[] }>(`/training-signals/${encodeURIComponent(shardId)}/verify`, { method: "POST" }),
  listRewardIntegrityAudits: (params: { runId?: string; status?: string; profileRevisionId?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.runId) search.set("run_id", params.runId);
    if (params.status) search.set("status", params.status);
    if (params.profileRevisionId) search.set("integrity_profile_revision_id", params.profileRevisionId);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/reward-integrity-audits${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeRewardIntegrityAudit));
  },
  runRewardIntegrityAudits: (runId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/runs/${encodeURIComponent(runId)}/reward-integrity-audits${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeRewardIntegrityAudit));
  },
  launchRewardIntegrityAudit: (payload: Record<string, unknown>) =>
    request<unknown>("/reward-integrity-audits", { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardIntegrityAudit),
  rewardIntegrityAudit: (auditId: string) =>
    request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}`).then(normalizeRewardIntegrityAudit),
  rewardIntegrityAuditSamples: (auditId: string, params: { classification?: string; stratum?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.classification) {
      const outcomes: Record<string, string> = {
        optimizer_only_accept: "optimizer_only",
        sentinel_only_accept: "sentinel_only",
      };
      search.set("outcome", outcomes[params.classification] ?? params.classification);
    }
    if (params.stratum) search.set("population", params.stratum);
    if (params.q) search.set("q", params.q);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/samples${qs ? `?${qs}` : ""}`).then((value) => normalizePage(value, normalizeRewardIntegritySample));
  },
  rewardIntegrityAuditMetrics: (auditId: string) =>
    request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/metrics`).then((value) => normalizePage(value, normalizeRewardIntegrityMetric)),
  cancelRewardIntegrityAudit: (auditId: string) =>
    request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/cancel`, { method: "POST" }).then(normalizeRewardIntegrityAudit),
  retryRewardIntegrityAudit: (auditId: string, reason: string) =>
    request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/retry`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    }).then(normalizeRewardIntegrityAudit),
  verifyRewardIntegrityAudit: (auditId: string) =>
    request<{ audit_id: string; valid: boolean; errors?: string[]; checksums?: Record<string, string> }>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/verify`, { method: "POST" }),
  reviewRewardIntegrityAudit: (auditId: string, payload: { action: "continue" | "stop" | "fork"; reason: string }) =>
    request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/review`, { method: "POST", body: JSON.stringify(payload) }).then(normalizeRewardIntegrityReviewResult),
  rewardIntegrityForkContext: (auditId: string) =>
    request<unknown>(`/reward-integrity-audits/${encodeURIComponent(auditId)}/fork-context`).then(normalizeRewardIntegrityForkContext),
  compareRewardIntegrityAudits: (baseAuditId: string, candidateAuditId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams({ base_id: baseAuditId, candidate_id: candidateAuditId });
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    return request<unknown>(`/reward-integrity-audits/compare?${search.toString()}`).then(normalizeRewardIntegrityComparison);
  },
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
    request<TrainingLaunchResult>("/train/launch", {
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
  runLaunchConfig: (runId: string) =>
    request<{
      run_id: string;
      parent_run_id?: string | null;
      resolved_config: Record<string, unknown>;
      datasets: DatasetBinding[];
    }>(`/runs/${encodeURIComponent(runId)}/launch-config`),
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

  // ----- Dataset Lab ------------------------------------------------------
  listDatasets: async () => {
    const response = await request<{ items: DatasetRecordWire[] }>("/datasets");
    return { ...response, items: response.items.map(normalizeDatasetRecord) };
  },
  createDataset: async (payload: DatasetCreatePayload) =>
    normalizeDatasetRecord(await request<DatasetRecordWire>("/datasets", {
      method: "POST",
      body: JSON.stringify(payload),
    })),
  datasetDetail: async (datasetId: string) =>
    normalizeDatasetRecord(
      await request<DatasetRecordWire>(`/datasets/${encodeURIComponent(datasetId)}`),
    ),
  datasetPreview: (datasetId: string, params: { offset?: number; limit?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<DatasetPreview>(
      `/datasets/${encodeURIComponent(datasetId)}/preview${qs ? `?${qs}` : ""}`,
    );
  },
  datasetStatistics: (datasetId: string) =>
    request<Record<string, unknown>>(`/datasets/${encodeURIComponent(datasetId)}/statistics`),
  buildDataset: (datasetId: string, payload: DatasetBuildPayload) =>
    request<DatasetJobAccepted>(`/datasets/${encodeURIComponent(datasetId)}/build`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  datasetVersions: (datasetId: string) =>
    request<{ items: DatasetVersion[] }>(
      `/datasets/${encodeURIComponent(datasetId)}/versions`,
    ),
  datasetVersion: (versionId: string) =>
    request<DatasetVersion>(`/dataset-versions/${encodeURIComponent(versionId)}`),
  datasetVersionPreview: (
    versionId: string,
    params: { split?: string; offset?: number; limit?: number } = {},
  ) => {
    const search = new URLSearchParams();
    if (params.split) search.set("split", params.split);
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<DatasetPreview>(
      `/dataset-versions/${encodeURIComponent(versionId)}/preview${qs ? `?${qs}` : ""}`,
    );
  },
  datasetVersionStatistics: (versionId: string) =>
    request<Record<string, unknown>>(
      `/dataset-versions/${encodeURIComponent(versionId)}/statistics`,
    ),
  datasetVersionRuns: (versionId: string) =>
    request<{ items: RunListItem[] }>(
      `/dataset-versions/${encodeURIComponent(versionId)}/runs`,
    ),
  compareDatasetVersions: (versionId: string, otherVersionId: string) => {
    const search = new URLSearchParams({ other_version_id: otherVersionId });
    return request<DatasetVersionComparison>(
      `/dataset-versions/${encodeURIComponent(versionId)}/compare?${search.toString()}`,
    );
  },
  listTrainingArtifacts: (versionId: string) =>
    request<{ items: TrainingDatasetArtifact[] }>(
      `/dataset-versions/${encodeURIComponent(versionId)}/training-artifacts`,
    ),
  renderTrainingArtifact: (versionId: string, payload: TrainingArtifactCreatePayload) =>
    request<DatasetJobAccepted>(
      `/dataset-versions/${encodeURIComponent(versionId)}/training-artifacts`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  trainingArtifact: (artifactId: string) =>
    request<TrainingDatasetArtifact>(
      `/training-artifacts/${encodeURIComponent(artifactId)}`,
    ),
  exportDatasetVersion: (versionId: string, payload: DatasetExportPayload) =>
    request<Record<string, unknown>>(
      `/dataset-versions/${encodeURIComponent(versionId)}/export`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  materializeDatasetVersion: (versionId: string) =>
    request<DatasetJobAccepted>(
      `/dataset-versions/${encodeURIComponent(versionId)}/materialize`,
      { method: "POST", body: JSON.stringify({}) },
    ),
  cloneDatasetRecipe: (versionId: string) =>
    request<{ recipe: DatasetRecipe; parent_version_id: string }>(
      `/dataset-versions/${encodeURIComponent(versionId)}/clone-recipe`,
      { method: "POST" },
    ),
  listDatasetJobs: (params: { datasetId?: string; status?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.datasetId) search.set("dataset_id", params.datasetId);
    if (params.status) search.set("status", params.status);
    const qs = search.toString();
    return request<{ items: DatasetJob[] }>(`/dataset-jobs${qs ? `?${qs}` : ""}`);
  },
  datasetJob: (jobId: string) =>
    request<DatasetJob>(`/dataset-jobs/${encodeURIComponent(jobId)}`),
  cancelDatasetJob: (jobId: string) =>
    request<DatasetJob>(`/dataset-jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    }),
  retryDatasetJob: (jobId: string) =>
    request<DatasetJobAccepted>(`/dataset-jobs/${encodeURIComponent(jobId)}/retry`, {
      method: "POST",
    }),

  // ----- Guided own-data training ---------------------------------------
  interfaceCapabilities: () =>
    request<{ items: InterfaceCapabilityDescriptor[]; total?: number }>("/interface-capabilities"),
  trainingScenarios: (params: { includeUnavailable?: boolean; modality?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.includeUnavailable !== undefined) search.set("include_unavailable", String(params.includeUnavailable));
    if (params.modality) search.set("modality", params.modality);
    const qs = search.toString();
    return request<{ items: TrainingScenarioDescriptor[]; total?: number }>(`/training-scenarios${qs ? `?${qs}` : ""}`);
  },
  trainingScenario: (scenarioId: string) =>
    request<TrainingScenarioDescriptor>(`/training-scenarios/${encodeURIComponent(scenarioId)}`),
  adviseTrainingScenario: (payload: ScenarioAdviceRequest) =>
    request<ScenarioAdviceResult>("/training-scenarios/advise", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  guidedTrainingExamples: () =>
    request<{ items: GuidedExampleDescriptor[]; total?: number }>("/training-scenario-examples"),
  trainingScenarioExamples: (scenarioId: string) =>
    request<{ items: TrainingScenarioExample[]; total?: number }>(`/training-scenarios/${encodeURIComponent(scenarioId)}/examples`),
  trainingScenarioTemplate: (scenarioId: string) =>
    request<TrainingScenarioExample>(`/training-scenarios/${encodeURIComponent(scenarioId)}/template`),
  createDatasetImport: (payload: {
    source_kind: string;
    scenario_revision_id?: string;
    source_uri?: string;
    name?: string;
    config?: string;
    split?: string;
    revision?: string;
    example_id?: string;
    expected_size_bytes?: number;
    capacity_override_reason?: string;
  }) => request<DatasetImportSession>("/dataset-imports", { method: "POST", body: JSON.stringify(payload) }),
  datasetImport: (importId: string) =>
    request<DatasetImportSession>(`/dataset-imports/${encodeURIComponent(importId)}`),
  cancelDatasetImport: (importId: string) =>
    request<DomainWorkResult<DatasetImportSession>>(`/dataset-imports/${encodeURIComponent(importId)}/cancel`, { method: "POST" }),
  retryDatasetImport: (importId: string) =>
    request<DomainWorkResult<DatasetImportSession>>(`/dataset-imports/${encodeURIComponent(importId)}/retry`, { method: "POST" }),
  huggingFaceDatasetOptions: (repoId: string, revision: string) => {
    const search = new URLSearchParams({ repo_id: repoId, revision });
    return request<HuggingFaceDatasetOptions>(`/dataset-imports/huggingface/options?${search.toString()}`);
  },
  createDatasetImportFile: (importId: string, payload: { relative_path: string; size_bytes: number; content_type?: string; content_hash?: string; capacity_override_reason?: string }) =>
    request<DatasetImportFile>(`/dataset-imports/${encodeURIComponent(importId)}/files`, { method: "POST", body: JSON.stringify(payload) }),
  uploadDatasetImportFileChunk: (
    importId: string,
    fileId: string,
    content: ArrayBuffer,
    range: { start: number; end: number; total: number },
    contentHash?: string,
  ) => request<DatasetImportFile>(`/dataset-imports/${encodeURIComponent(importId)}/files/${encodeURIComponent(fileId)}/content`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Range": `bytes ${range.start}-${range.end}/${range.total}`,
      ...(contentHash ? { "X-Content-SHA256": contentHash } : {}),
    },
    body: content,
  }),
  inspectDatasetImport: (importId: string, payload: { scenario_revision_id?: string; force?: boolean } = {}) =>
    request<DomainWorkResult<DatasetSourceInspection>>(`/dataset-imports/${encodeURIComponent(importId)}/inspect`, { method: "POST", body: JSON.stringify(payload) }),
  datasetSourceInspection: (inspectionId: string) =>
    request<DatasetSourceInspection>(`/dataset-inspections/${encodeURIComponent(inspectionId)}`),
  cancelDatasetInspection: (inspectionId: string) =>
    request<DatasetSourceInspection>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/cancel`, { method: "POST" }),
  retryDatasetInspection: (inspectionId: string) =>
    request<DomainWorkResult<DatasetSourceInspection>>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/retry`, { method: "POST" }),
  previewDatasetMapping: (inspectionId: string, payload: { mapping_plan: FieldMappingPlan }) =>
    request<MappingPreview>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/mapping-preview`, { method: "POST", body: JSON.stringify(payload) }),
  previewDatasetSemantics: (inspectionId: string, payload: { mapping_plan: FieldMappingPlan }, limit = 50) =>
    request<SemanticPreviewResponse>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/semantic-preview?limit=${encodeURIComponent(String(limit))}`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  datasetInspectionReadiness: (inspectionId: string, payload: { preparation_plan: DatasetPreparationPlan }) =>
    request<DatasetReadiness>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/readiness`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  previewDatasetPreparation: (inspectionId: string, payload: { preparation_plan: DatasetPreparationPlan }) =>
    request<DatasetPreparationPlan>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/preparation-preview`, { method: "POST", body: JSON.stringify(payload) }),
  registerInspectedDataset: (inspectionId: string, payload: {
    name: string;
    description?: string;
    import_id?: string;
    scenario_revision_id: string;
    mapping_plan: FieldMappingPlan;
    preparation_plan: DatasetPreparationPlan;
    capacity_override_reason?: string;
  }) => request<DomainWorkResult<DatasetRecord>>(`/dataset-inspections/${encodeURIComponent(inspectionId)}/register`, { method: "POST", body: JSON.stringify(payload) }),
  refreshDatasetSource: (sourceId: string) =>
    request<DomainWorkResult<DatasetSource>>(`/dataset-sources/${encodeURIComponent(sourceId)}/refresh`, { method: "POST", body: JSON.stringify({}) }),
  datasetVersionReadiness: (versionId: string, trainerMode?: string, model?: string, verifierProfileRevisionId?: string) => {
    const search = new URLSearchParams();
    if (trainerMode) search.set("trainer_mode", trainerMode);
    if (model) search.set("model", model);
    if (verifierProfileRevisionId) search.set("verifier_profile_revision_id", verifierProfileRevisionId);
    const qs = search.toString() ? `?${search.toString()}` : "";
    return request<DatasetReadiness>(`/dataset-versions/${encodeURIComponent(versionId)}/readiness${qs}`);
  },
  launchDatasetProofRun: (versionId: string, payload: Record<string, unknown>) =>
    request<TrainingLaunchResult & { work_item_id?: string | null; proof_run?: boolean }>(`/dataset-versions/${encodeURIComponent(versionId)}/proof-run`, { method: "POST", body: JSON.stringify(payload) }),
  launchFullRunFromProof: (runId: string, payload: { reason?: string; assessment_id?: string; override_reason?: string } = {}) =>
    request<TrainingLaunchResult>(`/runs/${encodeURIComponent(runId)}/full-run`, { method: "POST", body: JSON.stringify(payload) }),
  labCapabilities: () => request<Record<string, unknown>>("/lab-capabilities"),
  listOutcomeAssessments: (params: { proofRunId?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.proofRunId) search.set("proof_run_id", params.proofRunId);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<TrainingOutcomeAssessment>>(`/outcome/assessments${qs ? `?${qs}` : ""}`);
  },
  assessTrainingOutcome: (proofRunId: string, payload: Record<string, unknown> = {}) =>
    request<TrainingOutcomeAssessment>(`/outcome/assessments/${encodeURIComponent(proofRunId)}`, { method: "POST", body: JSON.stringify(payload) }),
  prepareTrainingOutcome: (proofRunId: string, payload: Record<string, unknown> = {}) =>
    request<OutcomePreparation>(`/outcome/runs/${encodeURIComponent(proofRunId)}/prepare`, { method: "POST", body: JSON.stringify(payload) }),
  actionableGuidance: (contextKind: string, contextId: string) =>
    request<ActionableGuidance>(`/guidance/${encodeURIComponent(contextKind)}/${encodeURIComponent(contextId)}`),
  trainingOutcome: (assessmentId: string) =>
    request<TrainingOutcomeAssessment>(`/outcome/assessments/${encodeURIComponent(assessmentId)}`),
  outcomeFindings: (assessmentId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<Record<string, unknown>>>(`/outcome/assessments/${encodeURIComponent(assessmentId)}/findings${qs ? `?${qs}` : ""}`);
  },
  reviewTrainingOutcome: (proofRunId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/outcome/runs/${encodeURIComponent(proofRunId)}/review`, { method: "POST", body: JSON.stringify(payload) }),
  fullRunContext: (proofRunId: string, params: { assessmentId?: string; overrideReason?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.assessmentId) search.set("assessment_id", params.assessmentId);
    if (params.overrideReason) search.set("override_reason", params.overrideReason);
    const qs = search.toString();
    return request<Record<string, unknown>>(`/outcome/runs/${encodeURIComponent(proofRunId)}/full-run-context${qs ? `?${qs}` : ""}`);
  },
  documentExtractors: () =>
    request<{ items: DocumentExtractorDescriptor[]; total?: number }>("/document-extractors"),
  createDocumentExtraction: (payload: Record<string, unknown>) =>
    request<DomainWorkResult<DocumentExtraction>>("/document-extractions", { method: "POST", body: JSON.stringify(payload) }),
  documentExtraction: (extractionId: string) =>
    request<DocumentExtraction>(`/document-extractions/${encodeURIComponent(extractionId)}`),
  previewDocumentExtraction: (extractionId: string, payload: Record<string, unknown> = {}) =>
    request<DocumentExtractionPreview>(`/document-extractions/${encodeURIComponent(extractionId)}/preview`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  corpusProfile: (versionId: string) =>
    request<CorpusProfile>(`/dataset-versions/${encodeURIComponent(versionId)}/corpus-profile`),
  corpusPackingPlan: (versionId: string, payload: CorpusPackingRequest) =>
    request<CorpusPackingPlanResponse>(`/dataset-versions/${encodeURIComponent(versionId)}/packing-plan`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  cptPreflight: (payload: CorpusTrainingConfig) =>
    request<TrainingPreflight & { corpus_profile?: CorpusProfile; packing_plan?: CorpusPackingPlan; readiness?: DatasetReadiness }>("/cpt/preflight", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  launchCpt: (payload: CorpusTrainingConfig) =>
    request<TrainingLaunchResult>("/cpt/launch", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ----- benchmark suites and persistent evaluation --------------------
  listBenchmarkSuites: () =>
    request<{ items: BenchmarkSuite[] }>("/benchmark-suites"),
  createBenchmarkSuite: (payload: BenchmarkSuiteCreatePayload) =>
    request<BenchmarkSuite>("/benchmark-suites", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  benchmarkSuite: (suiteId: string) =>
    request<BenchmarkSuite>(`/benchmark-suites/${encodeURIComponent(suiteId)}`),
  createBenchmarkSuiteRevision: (
    suiteId: string,
    payload: Omit<BenchmarkSuiteCreatePayload, "name" | "description">,
  ) =>
    request<BenchmarkSuiteRevision>(
      `/benchmark-suites/${encodeURIComponent(suiteId)}/revisions`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  listEvaluations: (params: { runId?: string; status?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.runId) search.set("run_id", params.runId);
    if (params.status) search.set("status", params.status);
    const qs = search.toString();
    return request<{ items: Evaluation[] }>(`/evaluations${qs ? `?${qs}` : ""}`);
  },
  evaluationHistory: (params: { subjectRef?: string; suiteRevisionId?: string; limit?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.subjectRef) search.set("subject_ref", params.subjectRef);
    if (params.suiteRevisionId) search.set("suite_revision_id", params.suiteRevisionId);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<EvaluationHistoryResponse>(`/evaluations/history${qs ? `?${qs}` : ""}`);
  },
  evaluationDrift: (params: { baseId?: string; candidateId?: string; subjectRef?: string; suiteRevisionId?: string; practicalDelta?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.baseId) search.set("base_id", params.baseId);
    if (params.candidateId) search.set("candidate_id", params.candidateId);
    if (params.subjectRef) search.set("subject_ref", params.subjectRef);
    if (params.suiteRevisionId) search.set("suite_revision_id", params.suiteRevisionId);
    if (params.practicalDelta !== undefined) search.set("practical_delta", String(params.practicalDelta));
    const qs = search.toString();
    return request<EvaluationDrift>(`/evaluations/drift${qs ? `?${qs}` : ""}`);
  },
  launchEvaluation: (payload: EvaluationCreatePayload) =>
    request<Evaluation>("/evaluations", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  launchEvaluationBatch: (payload: {
    suite_revision_id: string;
    base: EvaluationSubject;
    candidates: EvaluationSubject[];
    adapter_id?: string;
    verifier_profile_revision_id?: string;
    request?: Record<string, unknown>;
    reuse_completed?: boolean;
    filters?: Record<string, unknown>;
  }) => request<EvaluationBatch>("/evaluation-batches", { method: "POST", body: JSON.stringify(payload) }),
  evaluation: (evaluationId: string) =>
    request<Evaluation>(`/evaluations/${encodeURIComponent(evaluationId)}`),
  evaluationSamples: (
    evaluationId: string,
    params: { offset?: number; limit?: number; classification?: string } = {},
  ) => {
    const search = new URLSearchParams();
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.classification) search.set("classification", params.classification);
    const qs = search.toString();
    return request<{ items: EvaluationSample[]; total: number; offset: number; limit: number }>(
      `/evaluations/${encodeURIComponent(evaluationId)}/samples${qs ? `?${qs}` : ""}`,
    );
  },
  listEvaluationJobs: () =>
    request<{ items: Evaluation[] }>("/evaluation-jobs"),
  cancelEvaluation: (evaluationId: string) =>
    request<Evaluation>(`/evaluations/${encodeURIComponent(evaluationId)}/cancel`, {
      method: "POST",
    }),
  retryEvaluation: (evaluationId: string) =>
    request<Evaluation>(`/evaluations/${encodeURIComponent(evaluationId)}/retry`, {
      method: "POST",
    }),
  compareEvaluations: (baseId: string, candidateId: string, offset = 0, limit = 100, filters: { classification?: string; q?: string } = {}) => {
    const search = new URLSearchParams({
      base_id: baseId,
      candidate_id: candidateId,
      offset: String(offset),
      limit: String(limit),
    });
    if (filters.classification) search.set("classification", filters.classification);
    if (filters.q) search.set("q", filters.q);
    return request<EvaluationComparison>(`/evaluations/compare?${search.toString()}`);
  },
  evaluationBatchComparisonSamples: (batchId: string, params: { candidateId?: string; classification?: string; q?: string; recordId?: string; offset?: number; limit?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.candidateId) search.set("candidate_id", params.candidateId);
    if (params.classification) search.set("classification", params.classification);
    if (params.q) search.set("q", params.q);
    if (params.recordId) search.set("record_id", params.recordId);
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<PaginatedResponse<EvaluationSampleDelta>>(`/evaluation-batches/${encodeURIComponent(batchId)}/comparison-samples${qs ? `?${qs}` : ""}`);
  },
  previewFailureMining: (payload: {
    base_id?: string;
    candidate_id: string;
    selector: FailureMiningSelector;
    excluded_record_ids?: string[];
  }) =>
    request<FailureMiningPreview>("/evaluation-mining/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  buildFailureMinedDataset: (payload: {
    dataset_id: string;
    parent_version_id: string;
    base_id?: string;
    candidate_id: string;
    selector: FailureMiningSelector;
    excluded_record_ids?: string[];
  }) =>
    request<DatasetJobAccepted>("/evaluation-mining/build", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ----- reproducible experiment operations -----------------------------
  listCheckpointPolicies: (params: { trainerMode?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.trainerMode) search.set("trainer_mode", params.trainerMode);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<CheckpointPolicyRevision>>(`/checkpoint-policies${qs ? `?${qs}` : ""}`);
  },
  checkpointPolicy: (policyRevisionId: string) =>
    request<CheckpointPolicyRevision>(`/checkpoint-policies/${encodeURIComponent(policyRevisionId)}`),
  createCheckpointPolicy: (payload: Omit<CheckpointPolicyRevision, "id" | "content_hash" | "created_at"> & { policy_id?: string; revision_number?: number }) =>
    request<CheckpointPolicyRevision>("/checkpoint-policies", { method: "POST", body: JSON.stringify(payload) }),
  resolveCheckpointPolicy: (payload: {
    policy_revision_id: string;
    trainer_mode: string;
    total_budget: number;
    budget_unit?: string;
    base_config?: Record<string, unknown>;
  }) => request<ResolvedCheckpointPlan>("/checkpoint-policies/resolve", { method: "POST", body: JSON.stringify(payload) }),
  listTrainerExecutionCapabilities: () =>
    request<{ items: TrainerExecutionCapability[] }>("/trainer-execution-capabilities"),
  listRunGroups: (params: { status?: string; kind?: string; limit?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.kind) search.set("kind", params.kind);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<{ items: RunGroup[] }>(`/run-groups${qs ? `?${qs}` : ""}`);
  },
  createRunGroup: (payload: RunGroupCreatePayload) =>
    request<RunGroup>("/run-groups", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runGroup: (groupId: string) =>
    request<RunGroup>(`/run-groups/${encodeURIComponent(groupId)}`),
  runGroupTrajectory: (groupId: string) =>
    request<RunGroupTrajectory>(`/run-groups/${encodeURIComponent(groupId)}/trajectory`),
  listRunGroupAnalyses: (groupId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<CohortAnalysisSnapshot>>(`/run-groups/${encodeURIComponent(groupId)}/analyses${qs ? `?${qs}` : ""}`);
  },
  createRunGroupAnalysis: (groupId: string, payload: Record<string, unknown> = {}) =>
    request<CohortAnalysisSnapshot>(`/run-groups/${encodeURIComponent(groupId)}/analyses`, { method: "POST", body: JSON.stringify(payload) }),
  cancelRunGroup: (groupId: string) =>
    request<RunGroup>(`/run-groups/${encodeURIComponent(groupId)}/cancel`, {
      method: "POST",
    }),
  resumeRunGroup: (groupId: string, reason?: string) =>
    request<RunGroup>(`/run-groups/${encodeURIComponent(groupId)}/resume`, {
      method: "POST",
      body: JSON.stringify(reason ? { reason } : {}),
    }),
  compareRunGroup: (groupId: string) =>
    request<Record<string, unknown>>(`/run-groups/${encodeURIComponent(groupId)}/compare`),
  forkBestRunGroup: (groupId: string) =>
    request<RunGroup>(
      `/run-groups/${encodeURIComponent(groupId)}/fork-best`,
      { method: "POST" },
    ),
  reviewGateDecision: (decisionId: string, payload: { action: "continue" | "stop"; reason: string }) =>
    request<CheckpointGateDecision>(`/gate-decisions/${encodeURIComponent(decisionId)}/review`, { method: "POST", body: JSON.stringify(payload) }),
  createResearchDecision: (payload: {
    analysis_snapshot_id: string;
    selected_subject: Record<string, unknown>;
    rejected_subjects?: Array<Record<string, unknown>>;
    exclusions?: Array<Record<string, unknown>>;
    rationale: string;
    override_reason?: string;
    fork_spec?: Record<string, unknown>;
  }) => request<ResearchDecisionRecord>("/research-decisions", { method: "POST", body: JSON.stringify(payload) }),
  researchDecision: (decisionId: string) =>
    request<ResearchDecisionRecord>(`/research-decisions/${encodeURIComponent(decisionId)}`),
  listResearchDecisions: (params: { runGroupId?: string; limit?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.runGroupId) search.set("run_group_id", params.runGroupId);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<PaginatedResponse<ResearchDecisionRecord>>(`/research-decisions${qs ? `?${qs}` : ""}`);
  },
  createEvidenceBundle: (payload: { analysis_snapshot_id: string; research_decision_id?: string; formats?: string[] }) =>
    request<EvidenceBundle>("/evidence-bundles", { method: "POST", body: JSON.stringify(payload) }),
  evidenceBundle: (bundleId: string) =>
    request<EvidenceBundle>(`/evidence-bundles/${encodeURIComponent(bundleId)}`),
  listWorkItems: (params: { status?: string; kind?: string; limit?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.kind) search.set("kind", params.kind);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    const qs = search.toString();
    return request<{ items: WorkItem[]; active_lease?: Record<string, unknown> | null }>(
      `/work-items${qs ? `?${qs}` : ""}`,
    );
  },
  workItem: (workItemId: string) =>
    request<WorkItem>(`/work-items/${encodeURIComponent(workItemId)}`),
  cancelWorkItem: (workItemId: string) =>
    request<WorkItem>(`/work-items/${encodeURIComponent(workItemId)}/cancel`, {
      method: "POST",
    }),
  retryWorkItem: (workItemId: string, reason?: string) =>
    request<WorkItem>(`/work-items/${encodeURIComponent(workItemId)}/retry`, {
      method: "POST",
      body: JSON.stringify(reason ? { reason } : {}),
    }),
  listModelArtifacts: (params: {
    runId?: string;
    groupId?: string;
    kind?: string;
    query?: string;
    limit?: number;
    offset?: number;
  } = {}) => {
    const search = new URLSearchParams();
    if (params.runId) search.set("run_id", params.runId);
    if (params.groupId) search.set("run_group_id", params.groupId);
    if (params.kind) search.set("kind", params.kind);
    if (params.query) search.set("query", params.query);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ModelArtifactOccurrence>>(
      `/model-artifacts${qs ? `?${qs}` : ""}`,
    );
  },
  modelArtifact: (artifactId: string) =>
    request<ModelArtifactOccurrence>(`/model-artifacts/${encodeURIComponent(artifactId)}`),
  artifactLineage: (artifactId: string) =>
    request<{ artifact: ModelArtifactOccurrence; parents: ModelArtifactOccurrence[]; children: ModelArtifactOccurrence[]; edges: ArtifactEdge[] }>(
      `/model-artifacts/${encodeURIComponent(artifactId)}/lineage`,
    ),
  importArtifact: (payload: { path: string; kind?: string; adopt?: boolean; notes?: string }) =>
    request<{ artifact?: ModelArtifactOccurrence; operation_id?: string; work_item_id?: string }>(
      "/model-artifacts/import",
      { method: "POST", body: JSON.stringify(payload) },
    ),
  verifyArtifact: (artifactId: string) =>
    request<{ operation_id?: string; work_item_id?: string; artifact?: ModelArtifactOccurrence }>(
      `/model-artifacts/${encodeURIComponent(artifactId)}/verify`,
      { method: "POST", body: JSON.stringify({}) },
    ),
  pinArtifact: (artifactId: string, pinned: boolean) =>
    request<ModelArtifactOccurrence>(`/model-artifacts/${encodeURIComponent(artifactId)}/pin`, {
      method: pinned ? "POST" : "DELETE",
      body: pinned ? JSON.stringify({}) : undefined,
    }),
  tagArtifact: (artifactId: string, tags: string[]) =>
    request<ModelArtifactOccurrence>(`/model-artifacts/${encodeURIComponent(artifactId)}/tags`, {
      method: "POST",
      body: JSON.stringify({ tags }),
    }),
  promoteArtifact: (artifactId: string, payload: { alias: "candidate" | "approved" | string; note?: string; override?: boolean }) =>
    request<ArtifactAlias>(`/model-artifacts/${encodeURIComponent(artifactId)}/promote`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ----- artifact operations and qualification --------------------------
  listArtifactOperations: (params: { status?: string; kind?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.kind) search.set("kind", params.kind);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ArtifactOperation>>(
      `/artifact-operations${qs ? `?${qs}` : ""}`,
    );
  },
  createArtifactOperation: (payload: {
    kind: string;
    input_artifact_ids: string[];
    config?: Record<string, unknown>;
  }) =>
    request<ArtifactOperation & { work_item_id?: string | null }>("/artifact-operations", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  artifactOperation: (operationId: string) =>
    request<ArtifactOperation>(`/artifact-operations/${encodeURIComponent(operationId)}`),
  listQualificationProfiles: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<QualificationProfileRevision>>(
      `/qualification-profiles${qs ? `?${qs}` : ""}`,
    );
  },
  createQualificationProfile: (payload: QualificationProfileCreatePayload) =>
    request<QualificationProfileRevision>("/qualification-profiles", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listQualifications: (params: { artifactId?: string; status?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.artifactId) search.set("artifact_id", params.artifactId);
    if (params.status) search.set("status", params.status);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ArtifactQualification>>(
      `/qualifications${qs ? `?${qs}` : ""}`,
    );
  },
  qualifyArtifact: (payload: { artifact_id: string; profile_revision_id: string; parent_artifact_id?: string }) =>
    request<ArtifactQualification & { work_item_id?: string | null }>("/qualifications", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  compareQualifications: (baseId: string, candidateId: string) => {
    const search = new URLSearchParams({ base_id: baseId, candidate_id: candidateId });
    return request<QualificationComparison>(`/qualifications/compare?${search.toString()}`);
  },

  // ----- persistent playground sessions --------------------------------
  listPlaygroundSessions: (params: { limit?: number; offset?: number; includeArchived?: boolean } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    if (params.includeArchived !== undefined) search.set("include_archived", String(params.includeArchived));
    const qs = search.toString();
    return request<PaginatedResponse<PlaygroundSession>>(`/playground/sessions${qs ? `?${qs}` : ""}`);
  },
  createPlaygroundSession: (payload: Omit<PlaygroundSession, "messages" | "created_at" | "updated_at" | "archived"> & { messages?: never }) =>
    request<PlaygroundSession>("/playground/sessions", { method: "POST", body: JSON.stringify(payload) }),
  playgroundSession: (sessionId: string) =>
    request<PlaygroundSession>(`/playground/sessions/${encodeURIComponent(sessionId)}`),
  updatePlaygroundSession: (sessionId: string, payload: Partial<Pick<PlaygroundSession, "name" | "artifact_id" | "compare_artifact_id" | "endpoint" | "seed" | "generation_settings" | "settings">>) =>
    request<PlaygroundSession>(`/playground/sessions/${encodeURIComponent(sessionId)}`, { method: "PATCH", body: JSON.stringify(payload) }),
  archivePlaygroundSession: (sessionId: string) =>
    request<PlaygroundSession>(`/playground/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),
  appendPlaygroundMessage: (sessionId: string, payload: PlaygroundSessionMessage) =>
    request<PlaygroundSession>(`/playground/sessions/${encodeURIComponent(sessionId)}/messages`, { method: "POST", body: JSON.stringify(payload) }),
  reviewPlaygroundSession: (sessionId: string, payload: {
    kind: "benchmark_suite" | "dataset_source" | "review_queue";
    message_ids?: string[];
    review_note: string;
    schema_revision_id?: string;
    name?: string;
    seed?: number;
    strategies?: Array<Record<string, unknown>>;
    policy?: ReviewPolicy;
    pairings?: PlaygroundReviewPairing[];
  }) =>
    request<PlaygroundReviewResult>(`/playground/sessions/${encodeURIComponent(sessionId)}/review`, { method: "POST", body: JSON.stringify(payload) }),

  // ----- Human Feedback and Active Data Studio -------------------------
  reviewCapabilities: () => request<ReviewCapabilities>("/review-capabilities"),
  listAnnotationSchemas: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AnnotationSchema>>(`/annotation-schemas${qs ? `?${qs}` : ""}`);
  },
  createAnnotationSchema: (payload: {
    schema_id?: string;
    name: string;
    description?: string;
    modality: AnnotationModality;
    task_type: AnnotationTaskType;
    definition: Record<string, unknown>;
  }) => request<{ schema: AnnotationSchema; revision: AnnotationSchemaRevision }>("/annotation-schemas", { method: "POST", body: JSON.stringify(payload) }),
  validateAnnotationSchema: (payload: {
    name?: string;
    modality: AnnotationModality;
    task_type: AnnotationTaskType;
    definition: Record<string, unknown>;
  }) => request<{ valid: boolean; schema?: Record<string, unknown>; errors: Array<{ field?: string; message: string }> }>("/annotation-schemas/validate", { method: "POST", body: JSON.stringify(payload) }),
  annotationSchema: (schemaId: string) =>
    request<AnnotationSchema & { revisions?: AnnotationSchemaRevision[]; revision_count?: number }>(`/annotation-schemas/${encodeURIComponent(schemaId)}`),
  listAnnotationSchemaRevisions: (schemaId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AnnotationSchemaRevision>>(`/annotation-schemas/${encodeURIComponent(schemaId)}/revisions${qs ? `?${qs}` : ""}`);
  },
  reviseAnnotationSchema: (schemaId: string, payload: { modality?: AnnotationModality; task_type?: AnnotationTaskType; definition: Record<string, unknown> }) =>
    request<AnnotationSchemaRevision>(`/annotation-schemas/${encodeURIComponent(schemaId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }),
  annotationSchemaRevision: (revisionId: string) =>
    request<AnnotationSchemaRevision>(`/annotation-schema-revisions/${encodeURIComponent(revisionId)}`),
  listAcquisitionBatches: (params: { status?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AcquisitionBatch>>(`/acquisition-batches${qs ? `?${qs}` : ""}`);
  },
  createAcquisitionBatch: (payload: AcquisitionRequest) => request<AcquisitionBatch>("/acquisition-batches", { method: "POST", body: JSON.stringify(payload) }),
  acquisitionBatch: (batchId: string) => request<AcquisitionBatch>(`/acquisition-batches/${encodeURIComponent(batchId)}`),
  cancelAcquisitionBatch: (batchId: string) => request<AcquisitionBatch>(`/acquisition-batches/${encodeURIComponent(batchId)}/cancel`, { method: "POST" }),
  retryAcquisitionBatch: (batchId: string) => request<AcquisitionBatch>(`/acquisition-batches/${encodeURIComponent(batchId)}/retry`, { method: "POST" }),
  acquisitionCandidates: (batchId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AcquisitionCandidate>>(`/acquisition-batches/${encodeURIComponent(batchId)}/candidates${qs ? `?${qs}` : ""}`);
  },
  listReviewQueues: (params: { status?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.q) search.set("q", params.q);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ReviewQueue>>(`/review-queues${qs ? `?${qs}` : ""}`);
  },
  reviewQueueSummaries: (params: { status?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.q) search.set("q", params.q);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ReviewQueueSummary>>(`/review-queues/summaries${qs ? `?${qs}` : ""}`);
  },
  createReviewQueue: (payload: { batch_id: string; schema_revision_id: string; name: string; policy: ReviewPolicy }) =>
    request<ReviewQueue>("/review-queues", { method: "POST", body: JSON.stringify(payload) }),
  cloneReviewQueue: (queueId: string, payload: { name?: string; policy?: ReviewPolicy } = {}) =>
    request<ReviewQueue>(`/review-queues/${encodeURIComponent(queueId)}/clone`, { method: "POST", body: JSON.stringify(payload) }),
  reviewQueue: (queueId: string) => request<ReviewQueue>(`/review-queues/${encodeURIComponent(queueId)}`),
  reviewQueueStatistics: (queueId: string) => request<ReviewQueueStatistics>(`/review-queues/${encodeURIComponent(queueId)}/statistics`),
  reviewQueueItems: (queueId: string, params: { status?: string; passNumber?: number; q?: string; cursor?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.passNumber !== undefined) search.set("pass_number", String(params.passNumber));
    if (params.q) search.set("q", params.q);
    if (params.cursor) search.set("cursor", params.cursor);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ReviewItem>>(`/review-queues/${encodeURIComponent(queueId)}/items${qs ? `?${qs}` : ""}`);
  },
  reviewItemNeighbors: (itemId: string, params: { status?: string; passNumber?: number; q?: string } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.passNumber !== undefined) search.set("pass_number", String(params.passNumber));
    if (params.q) search.set("q", params.q);
    const qs = search.toString();
    return request<ReviewItemNeighbors>(`/review-items/${encodeURIComponent(itemId)}/neighbors${qs ? `?${qs}` : ""}`);
  },
  updateReviewQueueState: (queueId: string, action: "pause" | "resume" | "archive" | "start-second-pass", reason?: string) =>
    request<ReviewQueue>(`/review-queues/${encodeURIComponent(queueId)}/${action}`, { method: "POST", body: JSON.stringify(reason ? { reason } : {}) }),
  reviewItem: (itemId: string) => request<ReviewItem>(`/review-items/${encodeURIComponent(itemId)}`),
  reviewItemEvents: (itemId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<ReviewEvent>>(`/review-items/${encodeURIComponent(itemId)}/events${qs ? `?${qs}` : ""}`);
  },
  submitReviewEvent: (itemId: string, payload: {
    event_type: string;
    pass_number: number;
    payload: Record<string, unknown>;
    idempotency_key: string;
    expected_active_event_id?: string | null;
    supersedes_event_id?: string | null;
    reason?: string;
  }) => request<ReviewEvent>(`/review-items/${encodeURIComponent(itemId)}/events`, { method: "POST", body: JSON.stringify(payload) }),
  submitReviewEventBatch: (queueId: string, events: Array<Record<string, unknown>>) =>
    request<{ items: ReviewEvent[]; count: number; queue_id: string }>(`/review-queues/${encodeURIComponent(queueId)}/event-batches`, { method: "POST", body: JSON.stringify({ events }) }),
  reviewItemSuggestions: (itemId: string) => request<PaginatedResponse<ReviewSuggestion>>(`/review-items/${encodeURIComponent(itemId)}/suggestions`),
  generateReviewSuggestion: (itemId: string, payload: Record<string, unknown>) =>
    request<ReviewSuggestion>(`/review-items/${encodeURIComponent(itemId)}/suggestions`, { method: "POST", body: JSON.stringify(payload) }),
  publishReviewLabelSet: (queueId: string, payload: { name?: string } = {}) =>
    request<LabelSetPublicationAccepted>(`/review-queues/${encodeURIComponent(queueId)}/label-set-revisions`, { method: "POST", body: JSON.stringify(payload) }),
  listLabelSets: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<LabelSet>>(`/label-sets${qs ? `?${qs}` : ""}`);
  },
  labelSet: (labelSetId: string) => request<LabelSet & { revisions?: LabelSetRevision[] }>(`/label-sets/${encodeURIComponent(labelSetId)}`),
  labelSetRevision: (revisionId: string) => request<LabelSetRevision>(`/label-set-revisions/${encodeURIComponent(revisionId)}`),
  labelSetItems: (revisionId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<LabelSetItem>>(`/label-set-revisions/${encodeURIComponent(revisionId)}/items${qs ? `?${qs}` : ""}`);
  },
  verifyLabelSetRevision: (revisionId: string) => request<{ revision_id: string; valid: boolean; checksums?: Record<string, string>; errors?: string[] }>(`/label-set-revisions/${encodeURIComponent(revisionId)}/verify`, { method: "POST" }),
  previewLabelSetDataset: (revisionId: string, payload: Record<string, unknown>) =>
    request<DatasetBuildPreview>(`/label-set-revisions/${encodeURIComponent(revisionId)}/dataset-preview`, { method: "POST", body: JSON.stringify(payload) }),
  buildLabelSetDataset: (revisionId: string, payload: Record<string, unknown>) =>
    request<DatasetJobAccepted>(`/label-set-revisions/${encodeURIComponent(revisionId)}/dataset-build`, { method: "POST", body: JSON.stringify(payload) }),
  listSpecDescriptors: (kind: string) => request<PaginatedResponse<SpecDescriptor>>(`/spec-descriptors/${encodeURIComponent(kind)}`),
  validateSpecDescriptor: (kind: string, descriptorId: string, value: Record<string, unknown>) =>
    request<SpecValidationResult>(`/spec-descriptors/${encodeURIComponent(kind)}/${encodeURIComponent(descriptorId)}/validate`, { method: "POST", body: JSON.stringify(value) }),

  // ----- Labs V12-V15 ---------------------------------------------------
  adaptationStudies: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AdaptationStudy>>(`/adaptation-studies${qs ? `?${qs}` : ""}`);
  },
  createAdaptationStudy: (payload: Record<string, unknown>) =>
    request<AdaptationStudy>("/adaptation-studies", { method: "POST", body: JSON.stringify(payload) }),
  createAdaptationStudyProtocol: (studyId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/adaptation-studies/${encodeURIComponent(studyId)}/protocols`, { method: "POST", body: JSON.stringify(payload) }),
  materializeAdaptationStudy: (revisionId: string) =>
    request<Record<string, unknown>>(`/adaptation-study-protocols/${encodeURIComponent(revisionId)}/materialize`, { method: "POST" }),
  adaptationStudyLaunchPlan: (revisionId: string) =>
    request<StudyLaunchPlan>(`/adaptation-study-protocols/${encodeURIComponent(revisionId)}/launch-plan`),
  launchAdaptationStudy: (revisionId: string, payload: Record<string, unknown> = {}) =>
    request<Record<string, unknown>>(`/adaptation-study-protocols/${encodeURIComponent(revisionId)}/launch`, { method: "POST", body: JSON.stringify(payload) }),
  analyzeAdaptationStudy: (revisionId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/adaptation-study-protocols/${encodeURIComponent(revisionId)}/analyses`, { method: "POST", body: JSON.stringify(payload) }),
  groundingProfiles: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<Record<string, unknown>>>(`/grounding/profiles${qs ? `?${qs}` : ""}`);
  },
  createGroundingProfile: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>("/grounding/profiles", { method: "POST", body: JSON.stringify(payload) }),
  createGroundingProfileRevision: (profileId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/grounding/profiles/${encodeURIComponent(profileId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }),
  groundedBatches: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<GroundedGenerationBatch>>(`/grounding/batches${qs ? `?${qs}` : ""}`);
  },
  groundedBatch: (batchId: string) =>
    request<GroundedGenerationBatch>(`/grounding/batches/${encodeURIComponent(batchId)}`),
  launchGroundedBatch: (revisionId: string, payload: Record<string, unknown>) =>
    request<GroundedGenerationBatch>(`/grounding/profile-revisions/${encodeURIComponent(revisionId)}/batches`, { method: "POST", body: JSON.stringify(payload) }),
  previewGroundedBatch: (revisionId: string, payload: Record<string, unknown>) =>
    request<GroundingGenerationPreview>(`/grounding/profile-revisions/${encodeURIComponent(revisionId)}/preview`, { method: "POST", body: JSON.stringify(payload) }),
  groundingCandidates: (batchId: string, params: { status?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<Record<string, unknown>>>(`/grounding/batches/${encodeURIComponent(batchId)}/candidates${qs ? `?${qs}` : ""}`);
  },
  createGroundingReviewProposal: (batchId: string) =>
    request<Record<string, unknown>>(`/grounding/batches/${encodeURIComponent(batchId)}/review-proposal`, { method: "POST" }),
  specializedTasks: () =>
    request<PaginatedResponse<SpecializedTaskDescriptor>>("/specialized-tasks"),
  specializedTaskReadiness: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>("/specialized-tasks/readiness", { method: "POST", body: JSON.stringify(payload) }),
  verifySpecializedTaskArtifact: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>("/specialized-task-artifacts/verify", { method: "POST", body: JSON.stringify(payload) }),
  agentEnvironments: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AgentEnvironment>>(`/environments${qs ? `?${qs}` : ""}`);
  },
  createAgentEnvironment: (payload: Record<string, unknown>) =>
    request<AgentEnvironment>("/environments", { method: "POST", body: JSON.stringify(payload) }),
  createAgentEnvironmentRevision: (environmentId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/environments/${encodeURIComponent(environmentId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }),
  createAgentEpisodeSuite: (environmentRevisionId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/environment-revisions/${encodeURIComponent(environmentRevisionId)}/suites`, { method: "POST", body: JSON.stringify(payload) }),
  createAgentEpisodeSuiteRevision: (suiteId: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/environment-suites/${encodeURIComponent(suiteId)}/revisions`, { method: "POST", body: JSON.stringify(payload) }),
  launchAgentEpisode: (suiteRevisionId: string, payload: Record<string, unknown>) =>
    request<AgentEpisode>(`/environment-suite-revisions/${encodeURIComponent(suiteRevisionId)}/episodes`, { method: "POST", body: JSON.stringify(payload) }),
  environmentPermissions: (environmentRevisionId: string) =>
    request<EnvironmentPermissionSummary>(`/environment-revisions/${encodeURIComponent(environmentRevisionId)}/permissions`),
  rerunAgentEpisode: (episodeId: string, payload: Record<string, unknown>) =>
    request<AgentEpisode>(`/environment-episodes/${encodeURIComponent(episodeId)}/rerun`, { method: "POST", body: JSON.stringify(payload) }),
  replayAgentEpisode: (episodeId: string) =>
    request<Record<string, unknown>>(`/environment-episodes/${encodeURIComponent(episodeId)}/replay`, { method: "POST" }),
  compareAgentEpisodes: (baseEpisodeId: string, candidateEpisodeId: string) =>
    request<Record<string, unknown>>("/environment-episodes/compare", { method: "POST", body: JSON.stringify({ base_episode_id: baseEpisodeId, candidate_episode_id: candidateEpisodeId }) }),
  publishAgentTrajectories: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>("/environment-trajectories", { method: "POST", body: JSON.stringify(payload) }),
  agentEpisodes: (params: { environmentRevisionId?: string; suiteRevisionId?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.environmentRevisionId) search.set("environment_revision_id", params.environmentRevisionId);
    if (params.suiteRevisionId) search.set("suite_revision_id", params.suiteRevisionId);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<AgentEpisode>>(`/environment-episodes${qs ? `?${qs}` : ""}`);
  },
  agentEpisode: (episodeId: string) =>
    request<AgentEpisode>(`/environment-episodes/${encodeURIComponent(episodeId)}`),
  agentEpisodeSteps: (episodeId: string, params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<Record<string, unknown>>>(`/environment-episodes/${encodeURIComponent(episodeId)}/steps${qs ? `?${qs}` : ""}`);
  },

  // ----- workstation activity and storage -------------------------------
  activity: (limit = 100) => loadActivitySnapshot(limit),
  activityEventsUrl: "/api/public/activity/events",
  listWorkers: () => request<PaginatedResponse<Worker>>("/workers"),
  storageInventory: () => request<StorageInventory>("/storage"),
  previewCleanup: (payload: { include_temporary?: boolean; include_trash?: boolean; older_than_days?: number } = {}) =>
    request<CleanupPlan>("/storage/cleanup", {
      method: "POST",
      body: JSON.stringify({ ...payload, preview: true }),
    }),
  executeCleanup: (planId: string, reviewNote: string) =>
    request<CleanupPlan & { work_item_id?: string | null }>("/storage/cleanup", {
      method: "POST",
      body: JSON.stringify({ plan_id: planId, approved: true, review_note: reviewNote }),
    }),

  // ----- guided non-destructive data repair and support -----------------
  datasetRepairs: (params: { limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<DatasetRepairSession>>(`/dataset-repairs${qs ? `?${qs}` : ""}`);
  },
  createDatasetRepair: (payload: Record<string, unknown>) =>
    request<DatasetRepairSession>("/dataset-repairs", { method: "POST", body: JSON.stringify(payload) }),
  datasetRepair: (sessionId: string) =>
    request<DatasetRepairSession>(`/dataset-repairs/${encodeURIComponent(sessionId)}`),
  datasetRepairIssues: (sessionId: string, params: { category?: string; severity?: string; limit?: number; offset?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.category) search.set("category", params.category);
    if (params.severity) search.set("severity", params.severity);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    const qs = search.toString();
    return request<PaginatedResponse<DatasetIssue>>(`/dataset-repairs/${encodeURIComponent(sessionId)}/issues${qs ? `?${qs}` : ""}`);
  },
  createDatasetRepairPlan: (sessionId: string, actions: DatasetRepairAction[]) =>
    request<DatasetRepairPlanRevision>(`/dataset-repairs/${encodeURIComponent(sessionId)}/plans`, { method: "POST", body: JSON.stringify({ actions }) }),
  datasetRepairPlan: (revisionId: string) =>
    request<DatasetRepairPlanRevision>(`/dataset-repair-plans/${encodeURIComponent(revisionId)}`),
  createDatasetRepairPreview: (sessionId: string, planRevisionId: string) =>
    request<DatasetRepairPreview>(`/dataset-repairs/${encodeURIComponent(sessionId)}/previews`, { method: "POST", body: JSON.stringify({ plan_revision_id: planRevisionId }) }),
  datasetRepairPreview: (previewId: string) =>
    request<DatasetRepairPreview>(`/dataset-repair-previews/${encodeURIComponent(previewId)}`),
  publishDatasetRepair: (previewId: string) =>
    request<Record<string, unknown>>(`/dataset-repair-previews/${encodeURIComponent(previewId)}/publish`, { method: "POST" }),
  datasetRepairRevision: (revisionId: string) =>
    request<Record<string, unknown>>(`/dataset-repair-revisions/${encodeURIComponent(revisionId)}`),
  rebaseDatasetRepair: (sessionId: string) =>
    request<DatasetRepairSession>(`/dataset-repairs/${encodeURIComponent(sessionId)}/rebase`, { method: "POST" }),
  cancelDatasetRepair: (sessionId: string) =>
    request<DatasetRepairSession>(`/dataset-repairs/${encodeURIComponent(sessionId)}/cancel`, { method: "POST" }),
  supportBundlePreview: (categories?: string[]) =>
    request<SupportBundlePreview>("/support-bundles/preview", { method: "POST", body: JSON.stringify({ categories }) }),
  createSupportBundle: (categories?: string[]) =>
    request<SupportBundle>("/support-bundles", { method: "POST", body: JSON.stringify({ categories }) }),
  supportBundle: (bundleId: string) => request<SupportBundle>(`/support-bundles/${encodeURIComponent(bundleId)}`),
  verifySupportBundle: (bundleId: string) =>
    request<{ id: string; valid: boolean; errors?: string[] }>(`/support-bundles/${encodeURIComponent(bundleId)}/verify`, { method: "POST" }),

  // ----- resumable workspaces and global lookup -------------------------
  workspaceDraft: <T = Record<string, unknown>>(surface: string, key: string) =>
    request<WorkspaceDraft<T>>(`/workspace-drafts/${encodeURIComponent(surface)}/${encodeURIComponent(key)}`),
  saveWorkspaceDraft: <T = Record<string, unknown>>(surface: string, key: string, payload: { name: string; content: T }) =>
    request<WorkspaceDraft<T>>(`/workspace-drafts/${encodeURIComponent(surface)}/${encodeURIComponent(key)}`, { method: "PUT", body: JSON.stringify({ kind: surface, ...payload }) }),
  deleteWorkspaceDraft: (surface: string, key: string) =>
    request<void>(`/workspace-drafts/${encodeURIComponent(surface)}/${encodeURIComponent(key)}`, { method: "DELETE" }),
  globalSearch: (query: string, params: { types?: string[]; limit?: number } = {}) => {
    const search = new URLSearchParams({ q: query });
    if (params.types?.length) search.set("types", params.types.join(","));
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    return request<PaginatedResponse<GlobalSearchResult>>(`/search?${search.toString()}`);
  },

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
