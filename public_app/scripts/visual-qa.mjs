import { chromium } from 'playwright';
import fs from 'node:fs/promises';

const base = process.env.HALO_FORGE_VISUAL_QA_URL || 'http://127.0.0.1:5173';
const outDir = process.env.HALO_FORGE_VISUAL_QA_OUT || '/private/tmp/halo-forge-visual-qa';
await fs.mkdir(outDir, { recursive: true });

const telemetry = {
  timestamp: Date.now() / 1000,
  backend: 'mlx',
  device_name: 'Apple M3 Max',
  gpu_util_percent: 42,
  vram_used_gb: 18.4,
  vram_total_gb: 64,
  power_watts: 38,
  temp_celsius: 61,
  cpu_util_percent: 23,
  sys_mem_used_gb: 32,
  sys_mem_total_gb: 64,
  throughput_tokens_per_sec: 128,
  active_run_id: 'demo-run',
  mps_to_cpu_fallbacks_60s: 0,
  chip: { generation: 3, variant: 'Max', gpu_cores: 40, nominal_memory_bandwidth_gbps: 400, brand: 'Apple M3 Max' },
  note: null,
};

const backend = {
  name: 'mlx',
  device: 'Apple M3 Max',
  chip: telemetry.chip,
  capabilities: {
    name: 'mlx', supports_bf16: true, supports_fp16: true, preferred_dtype_str: 'bf16',
    supports_4bit: false, supports_8bit: false, supports_flash_attn: false,
    preferred_attn_impl: 'mlx', supports_training: true, supports_peft: true, supports_neural_accelerators: true,
  },
  training_defaults: {},
  inference_defaults: {},
  mlx_readiness: {
    status: 'ready', executable: true, package_versions: { mlx: '0.29.0', 'mlx-lm': '0.28.0' },
    chip: telemetry.chip, macos_version: '15.5', metal_device: { name: 'Apple M3 Max' },
    errors: [], warnings: [], suggested_fixes: [], probe: {},
  },
};

const models = [
  {
    id: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', label: 'Qwen2.5 0.5B Instruct MLX', provider: 'mlx-community', family: 'Qwen2.5-Instruct', parameter_count: '0.5B',
    modalities: ['text', 'code'], tasks: ['mlx', 'chat', 'serving', 'code', 'quickstart'], trainer_support: ['sft', 'raft'], backend_support: ['mlx'], memory_tier: 'tiny',
    recommended_use: 'Smallest safe Apple Silicon first-run model for proving training, verifier, and output paths.', known_caveats: [], trust_remote_code_required: false,
    mlx_variant: null, status: 'recommended', recommended_first_run: true, estimated_memory_gb: 2.6, license_note: null, download_note: 'MLX-format Hugging Face artifact.',
    fit_notes: ['Best first MLX pick for laptops and dashboard smoke runs.'], risk_level: 'safe', last_verified: '2026-05-09', catalog_version: '2026.05',
  },
  {
    id: 'Qwen/Qwen2.5-Coder-0.5B', label: 'Qwen2.5 Coder 0.5B', provider: 'Qwen', family: 'Qwen2.5-Coder', parameter_count: '0.5B',
    modalities: ['text', 'code'], tasks: ['code', 'quickstart', 'sft', 'raft'], trainer_support: ['sft', 'raft'], backend_support: ['cpu', 'cuda', 'rocm', 'mps'], memory_tier: 'tiny',
    recommended_use: 'Fastest code smoke tests and CI-friendly trainer checks.', known_caveats: [], trust_remote_code_required: false,
    mlx_variant: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', status: 'recommended', recommended_first_run: true, estimated_memory_gb: 2.6,
    license_note: null, download_note: null, fit_notes: ['Best when validating install, dataset shape, and launch plumbing quickly.'], risk_level: 'safe', last_verified: '2026-05-09', catalog_version: '2026.05',
  },
  {
    id: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5 Instruct 1.5B', provider: 'Qwen', family: 'Qwen2.5-Instruct', parameter_count: '1.5B',
    modalities: ['text'], tasks: ['chat', 'reasoning', 'agentic', 'quickstart', 'tool-use'], trainer_support: ['sft', 'reasoning', 'agentic'], backend_support: ['cuda', 'rocm', 'mps'], memory_tier: 'small',
    recommended_use: 'Small general-purpose instruct model for reasoning and agentic quickstarts.', known_caveats: [], trust_remote_code_required: false,
    mlx_variant: null, status: 'recommended', recommended_first_run: true, estimated_memory_gb: 5.8, license_note: null, download_note: null,
    fit_notes: ['Good first instruct model when the task is not code-specific.'], risk_level: 'safe', last_verified: '2026-05-09', catalog_version: '2026.05',
  },
  {
    id: 'LiquidAI/LFM2.5-350M', label: 'LFM2.5 350M', provider: 'Liquid AI', family: 'LFM2.5', parameter_count: '350M',
    modalities: ['text'], tasks: ['structured-output', 'tool-use', 'edge'], trainer_support: ['sft'], backend_support: ['cpu', 'cuda', 'rocm', 'mps', 'mlx'], memory_tier: 'tiny',
    recommended_use: 'Interesting tiny model for structured output, tool use, extraction, and edge experiments.', known_caveats: ['Not recommended for programming.'], trust_remote_code_required: false,
    mlx_variant: 'LiquidAI/LFM2.5-350M-MLX', status: 'experimental', recommended_first_run: false, estimated_memory_gb: 2.1, license_note: null, download_note: null,
    fit_notes: [], risk_level: 'experimental', last_verified: '2026-05-09', catalog_version: '2026.05',
  },
];

const runDetail = {
  id: 'demo-run', run_id: 'demo-run', modality: 'sft', model_name: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', status: 'running', timestamp: new Date(Date.now() - 5 * 60_000).toISOString(),
  headline: 'Training is running', next_step: 'Watch loss and logs until the first checkpoint lands.', top_issue: null,
  user_summary: { headline: 'Training is running', next_step: 'Monitor the first run and review the final model when it completes.', confidence_tone: 'neutral' },
  metrics_summary: { progress_percent: 42, update_steps: 84, final_train_loss: 1.234 },
  details: {
    cycles_executed: 1, seed: 42, final_model_available: false,
    cycle_metrics: [
      { cycle: 0, train_loss: 2.1, initial_train_loss: 2.4, eval_loss: 2.2, avg_reward: null, avg_kept_reward: null, success_rate: null, samples_seen: 100, samples_kept: 100, train_steps_executed: 40, cycle_duration_seconds: 120, learning_rate: 0.0002 },
      { cycle: 1, train_loss: 1.7, initial_train_loss: 2.4, eval_loss: 1.9, avg_reward: null, avg_kept_reward: null, success_rate: null, samples_seen: 200, samples_kept: 200, train_steps_executed: 84, cycle_duration_seconds: 245, learning_rate: 0.0002 },
    ],
  },
};

const datasetVersion = {
  id: 'v-demo-001', dataset_id: 'ds-demo', status: 'completed', content_hash: '84d6c1a0f92d5d7b3a90f7fca1d64284', recipe_hash: '0e5a94b71b6c', storage_path: '/datasets/demo/v-demo-001', row_count: 1250, size_bytes: 4218880,
  split_counts: { train: 1000, validation: 125, test: 125 }, assets_materialized: false, created_at: new Date(Date.now() - 86_400_000).toISOString(),
  recipe: { name: 'training-ready', seed: 42, schema: 'sft', steps: [{ kind: 'normalize', trim: true, collapse_whitespace: true }, { kind: 'dedup', method: 'exact', field: 'text' }, { kind: 'validate', on_error: 'reject' }, { kind: 'split', method: 'random', ratios: { train: 0.8, validation: 0.1, test: 0.1 }, seed: 42 }] },
  statistics: { count: 1250, fields: 6, null_rate: 0.002, text_length: { min: 12, median: 428, max: 4096 }, rejections: { duplicate: 37, schema: 4 }, contamination: { matches: 0, action: 'report' } },
  provenance: { steps: [{ kind: 'normalize', input_count: 1291, output_count: 1291, rejected_count: 0 }, { kind: 'dedup', input_count: 1291, output_count: 1254, rejected_count: 37 }, { kind: 'validate', input_count: 1254, output_count: 1250, rejected_count: 4 }, { kind: 'split', input_count: 1250, output_count: 1250, rejected_count: 0 }] },
  compatible_trainers: [
    { adapter_id: 'sft.chat', adapter_version: '2', trainer_mode: 'sft', compatible: true, reason: 'Required prompt/response fields validated.' },
    { adapter_id: 'agentic.tool', adapter_version: '1', trainer_mode: 'agentic', compatible: false, reason: 'Tool definitions are missing.' },
  ],
};
const trainingArtifact = {
  id: 'artifact-demo-001', dataset_version_id: datasetVersion.id, status: 'ready', adapter_id: 'sft.chat', adapter_version: '2', trainer_mode: 'sft', model: models[0].id,
  tokenizer_revision: 'main', chat_template_hash: '75c24e1d4a99', bindings: [{ role: 'train', dataset_version_id: datasetVersion.id, split: 'train' }], row_counts: { train: 1000, validation: 125 },
  token_statistics: { total_tokens: 842100, median_tokens: 612, p95_tokens: 1770 }, artifact_hash: 'c318a93d2faec76094f6', derived_validation: false,
};
const suiteRevision = { id: 'suite-rev-core-3', suite_id: 'suite-core', revision: 3, items: [{ id: 'gsm8k', adapter: 'lm_eval', task: 'gsm8k' }, { id: 'heldout', adapter: 'dataset_split', dataset_version_id: datasetVersion.id, split: 'test' }], generation_settings: { temperature: 0, max_tokens: 512 }, primary_metric: 'accuracy', direction: 'maximize', content_hash: 'f72a1d1f' };
const suite = { id: 'suite-core', name: 'Core reasoning', purpose: 'development', description: 'Stable reasoning and held-out dataset checks.', latest_revision_id: suiteRevision.id, latest_revision: suiteRevision, revision_count: 3 };
const operationalRevision = { ...suiteRevision, id: 'suite-rev-ops-1', suite_id: 'suite-ops', revision: 1, primary_metric: 'output_tokens_per_second' };
const operationalSuite = { id: 'suite-ops', name: 'Local throughput', purpose: 'operational', description: 'Fixed local inference performance measurement.', latest_revision_id: operationalRevision.id, latest_revision: operationalRevision, revision_count: 1 };
const evaluationBase = { id: 'eval-base-001', suite_revision_id: suiteRevision.id, suite_id: suite.id, suite_name: suite.name, subject: { kind: 'model', value: 'Qwen/Qwen2.5-0.5B', subject_hash: 'basehash' }, status: 'completed', primary_metric: { name: 'accuracy', value: 0.61, direction: 'maximize', n_samples: 125 }, metrics: [{ name: 'accuracy', value: 0.61, direction: 'maximize', n_samples: 125 }], finished_at: new Date(Date.now() - 3600_000).toISOString() };
const evaluationCandidate = { id: 'eval-candidate-001', suite_revision_id: suiteRevision.id, suite_id: suite.id, suite_name: suite.name, subject: { kind: 'run', value: 'demo-run', run_id: 'demo-run', subject_hash: 'candidatehash' }, run_id: 'demo-run', status: 'completed', primary_metric: { name: 'accuracy', value: 0.68, direction: 'maximize', n_samples: 125 }, metrics: [{ name: 'accuracy', value: 0.68, direction: 'maximize', n_samples: 125 }], finished_at: new Date().toISOString() };
const evaluationCandidatePrevious = { ...evaluationCandidate, id: 'eval-candidate-000', primary_metric: { ...evaluationCandidate.primary_metric, value: 0.64 }, metrics: [{ ...evaluationCandidate.metrics[0], value: 0.64 }], finished_at: new Date(Date.now() - 86_400_000).toISOString() };
const evaluationComparison = { base_id: evaluationBase.id, candidate_id: evaluationCandidate.id, suite_revision_id: suiteRevision.id, primary_metric: 'accuracy', direction: 'maximize', base_value: 0.61, candidate_value: 0.68, delta: 0.07, counts: { regression: 4, improvement: 13, unchanged_failure: 27, unchanged_pass: 81 }, samples: [{ record_id: 'record-0042', suite_item_id: 'heldout', classification: 'regression', delta: -1, base: { suite_item_id: 'heldout', record_id: 'record-0042', input: 'Resolve 3x + 4 = 19.', expected: '5', output: '5', score: 1, passed: true, latency_ms: 182, error: null, verifier_trace: null }, candidate: { suite_item_id: 'heldout', record_id: 'record-0042', input: 'Resolve 3x + 4 = 19.', expected: '5', output: 'x = 7', score: 0, passed: false, latency_ms: 165, error: null, verifier_trace: { check: 'exact' } } }, { record_id: 'record-0088', suite_item_id: 'gsm8k', classification: 'improvement', delta: 1, base: { suite_item_id: 'gsm8k', record_id: 'record-0088', input: 'A tray has 24 items and loses one quarter.', expected: '18', output: '16', score: 0, passed: false, latency_ms: 190, error: null, verifier_trace: null }, candidate: { suite_item_id: 'gsm8k', record_id: 'record-0088', input: 'A tray has 24 items and loses one quarter.', expected: '18', output: '18', score: 1, passed: true, latency_ms: 172, error: null, verifier_trace: null } }] };
const modelArtifact = {
  id: 'artifact-model-001', occurrence_id: 'artifact-model-001', kind: 'adapter', content_hash: 'sha256:179bd1ee47fb2b7d4320a19d0103109ad8c2606e8eb2078377c6a7ef701ec204',
  path: '/tmp/halo-forge/runs/demo-run/final_adapter', run_id: 'demo-run', model_name: 'Qwen2.5 0.5B support adapter', format: 'huggingface', dtype: 'bf16', size_bytes: 182_400_000,
  integrity: 'verified', pinned: true, tags: ['support', 'sft'], aliases: [{ id: 'alias-1', alias: 'candidate', artifact_id: 'artifact-model-001', created_at: new Date().toISOString() }],
  locations: [{ id: 'location-1', path: '/tmp/halo-forge/runs/demo-run/final_adapter', kind: 'referenced', available: true }], metadata: { base_model: models[0].id }, created_at: new Date().toISOString(),
};
const baseArtifact = { ...modelArtifact, id: 'artifact-model-base', occurrence_id: 'artifact-model-base', kind: 'final', content_hash: 'sha256:279bd1ee47fb2b7d4320a19d0103109ad8c2606e8eb2078377c6a7ef701ec299', path: '/tmp/halo-forge/runs/base-run/final_model', run_id: 'base-run', model_name: 'Qwen2.5 0.5B base', size_bytes: 312_400_000, pinned: false, tags: ['base'], aliases: [], created_at: new Date(Date.now() - 86_400_000).toISOString() };
const qualifications = [
  { id: 'qual-1', artifact_id: modelArtifact.id, parent_artifact_id: baseArtifact.id, profile_revision_id: 'local-gguf-r3', status: 'pass', decision: 'pass', reasons: ['Quality delta within tolerance', 'Operational thresholds passed'], metrics: { accuracy: 0.68 }, performance: { output_tokens_per_second: 72.4, peak_process_memory_bytes: 2_420_000_000 }, completed_at: new Date().toISOString() },
  { id: 'qual-base', artifact_id: baseArtifact.id, profile_revision_id: 'local-gguf-r3', status: 'pass', decision: 'pass', reasons: ['Baseline recorded'], metrics: { accuracy: 0.69 }, performance: { output_tokens_per_second: 54.1, peak_process_memory_bytes: 3_180_000_000 }, completed_at: new Date(Date.now() - 86_400_000).toISOString() },
];
const playgroundSession = { id: 'session-demo', name: 'Candidate smoke test', artifact_id: modelArtifact.id, compare_artifact_id: baseArtifact.id, endpoint: null, seed: 42, generation_settings: { temperature: 0.2, max_tokens: 256 }, settings: { target: `artifact:${modelArtifact.id}`, compare_target: `artifact:${baseArtifact.id}` }, messages: [{ id: 'message-1', role: 'user', content: 'Explain why the local service is healthy.', artifact_id: modelArtifact.id, generation: { seed: 42 } }, { id: 'message-2', role: 'assistant', content: 'The model loaded, passed its readiness probe, and is accepting requests.', artifact_id: modelArtifact.id, generation: { seed: 42 } }], created_at: new Date().toISOString(), updated_at: new Date().toISOString(), archived: false };
const storageInventory = { total_bytes: 1_000_000_000_000, used_bytes: 620_000_000_000, free_bytes: 380_000_000_000, projected_free_bytes: 372_000_000_000, minimum_free_bytes: 100_000_000_000, low_disk: false, artifact_bytes: 32_000_000_000, cache_bytes: 114_000_000_000, temporary_bytes: 2_400_000_000, trash_bytes: 800_000_000, cache_items: [{ id: 'cache-1', path: models[0].id, kind: 'cache', available: true, verified_at: new Date().toISOString() }] };
const activitySnapshot = { worker: { id: 'worker-local', status: 'online', heartbeat_at: new Date().toISOString(), current_work_item_id: 'work-1' }, resource_lease: { owner: 'artifact-model-001', kind: 'accelerator' }, storage: storageInventory, items: [{ id: 'work-1', work_item_id: 'work-1', kind: 'artifact_convert', title: 'Q4 GGUF conversion', status: 'running', stage: 'writing tensors', progress_current: 62, progress_total: 100, progress_percent: 62, attempt: 1, max_attempts: 3, created_at: new Date(Date.now() - 120_000).toISOString(), started_at: new Date(Date.now() - 100_000).toISOString(), heartbeat_at: new Date().toISOString(), telemetry_rollup: { peak_system_memory_gb: 21.4, avg_cpu_percent: 74 }, next_actions: ['cancel'] }, { id: 'work-2', work_item_id: 'work-2', kind: 'evaluation', title: 'Operational qualification', status: 'queued', stage: 'waiting for accelerator', queue_position: 1, created_at: new Date().toISOString() }] };
const checkpointPolicy = { id: 'policy-rev-stability-2', policy_id: 'stability', revision_number: 2, name: 'SFT stability gates', description: 'Pause when development accuracy plateaus at a verified checkpoint.', development_suite_revision_id: suiteRevision.id, primary_metric: 'accuracy', direction: 'maximize', schedule: { mode: 'percentages', unit: 'step', percentages: [0.25, 0.5, 0.75, 1], include_final: true }, rules: [{ kind: 'plateau', metric: 'accuracy', direction: 'maximize', comparison: 'previous', minimum_delta: 0.005, practical_delta: 0.005, patience: 2, on_breach: 'pause', required: true }], retention: { keep_last: 1, keep_every_n_boundaries: null, keep_best: 1, protect_evaluated: true, protect_decision_referenced: true, protect_lineage_referenced: true, review_before_cleanup: true }, guardrail_suite_revision_ids: [], automatic_actions: true, compatible_capabilities: ['checkpoint', 'resume'], version: 1, content_hash: 'policyhash2' };
const resolvedCheckpointPlan = { policy_revision_id: checkpointPolicy.id, policy_hash: checkpointPolicy.content_hash, trainer_mode: 'sft', unit: 'step', total_budget: 1000, boundaries: [250, 500, 750, 1000], required_suite_revision_ids: [suiteRevision.id], automatic_actions: true, capability_notes: [], content_hash: 'planhash2' };
const gateDecision = { id: 'gate-decision-3', run_group_id: 'group-stability', run_id: 'demo-run', checkpoint_artifact_id: modelArtifact.id, policy_revision_id: checkpointPolicy.id, plan_hash: resolvedCheckpointPlan.content_hash, boundary_index: 2, boundary_value: 3, action: 'await_review', status: 'awaiting_review', automatic: false, reasons: ['Development accuracy plateaued within the practical-delta band.'], evidence: { current_metrics: { accuracy: 0.68 } }, content_hash: 'gatehash3', created_at: new Date().toISOString() };
const trajectory = { run_group_id: 'group-stability', policy_revision: checkpointPolicy, resolved_plan: resolvedCheckpointPlan, points: [{ id: 'point-1', run_id: 'demo-run', trial_id: 'trial-1', seed: 42, boundary_index: 0, boundary_value: 250, boundary_unit: 'step', status: 'completed', checkpoint_artifact_id: 'checkpoint-1', evaluation_id: 'eval-checkpoint-1', gate_decision_id: 'gate-1', gate_action: 'continue', metric_value: 0.61 }, { id: 'point-2', run_id: 'demo-run', trial_id: 'trial-1', seed: 42, boundary_index: 1, boundary_value: 500, boundary_unit: 'step', status: 'completed', checkpoint_artifact_id: 'checkpoint-2', evaluation_id: 'eval-checkpoint-2', gate_decision_id: 'gate-2', gate_action: 'continue', metric_value: 0.675 }, { id: 'point-3', run_id: 'demo-run', trial_id: 'trial-1', seed: 42, boundary_index: 2, boundary_value: 750, boundary_unit: 'step', status: 'awaiting_review', checkpoint_artifact_id: modelArtifact.id, evaluation_id: evaluationCandidate.id, gate_decision_id: gateDecision.id, gate_action: 'await_review', metric_value: 0.68 }], gate_decisions: [{ ...gateDecision, id: 'gate-1', boundary_index: 0, boundary_value: 250, action: 'continue', status: 'decided', reasons: ['Initial checkpoint established.'] }, { ...gateDecision, id: 'gate-2', boundary_index: 1, boundary_value: 500, action: 'continue', status: 'decided', reasons: ['Development accuracy improved.'] }, { ...gateDecision, boundary_value: 750 }] };
const cohortAnalysis = { id: 'analysis-stability-1', run_group_id: 'group-stability', request: { confidence: 0.95, bootstrap_resamples: 10000, bootstrap_seed: 42, replicate_unit: 'seed' }, analysis: { classification: 'improved', primary_metric: 'accuracy', direction: 'maximize', matched_seed_count: 3, practical_delta: 0.005, interval: { lower: 0.018, upper: 0.067, confidence: 0.95 }, compatibility: { compatible: true, matched_seed_count: 3, required_seed_count: 3 }, pareto: [{ subject_id: 'trial-1', primary_metric: 0.68, total_latency_ms: 720, output_tokens_per_second: 48.2, peak_memory_bytes: 3_200_000_000, energy_joules: 412 }, { subject_id: 'trial-2', primary_metric: 0.675, total_latency_ms: 610, output_tokens_per_second: 55.6, peak_memory_bytes: 2_800_000_000, artifact_size_bytes: 182_400_000 }] }, status: 'completed', content_hash: 'analysishash1', completed_at: new Date().toISOString() };
const runGroup = { id: 'group-stability', name: 'SFT checkpoint stability', kind: 'repeat', status: 'awaiting_review', trainer_mode: 'sft', suite_revision_id: suiteRevision.id, primary_metric: 'accuracy', direction: 'maximize', base_config: { mode: 'sft', model: models[0].id, dataset_version_id: datasetVersion.id, max_steps: 1000 }, seeds: [42, 43, 44], n_trials: 1, checkpoint_policy_revision_id: checkpointPolicy.id, checkpoint_policy: checkpointPolicy, resolved_checkpoint_plan: resolvedCheckpointPlan, awaiting_review_count: 1, completed_trials: 0, failed_trials: 0, pruned_trials: 0, trials: [{ id: 'trial-1', run_group_id: 'group-stability', ordinal: 0, status: 'awaiting_review', parameters: {}, aggregate: { count: 2, mean: 0.678, stddev: 0.004, direction: 'maximize' }, runs: [{ id: 'trial-run-1', trial_id: 'trial-1', run_id: 'demo-run', seed: 42, status: 'awaiting_review', objective_value: 0.68, evaluation_id: evaluationCandidate.id, model_artifact_id: modelArtifact.id, segment_count: 3 }] }], created_at: new Date(Date.now() - 3600_000).toISOString() };
runDetail.run_group_id = runGroup.id;
activitySnapshot.items.unshift({ id: 'review-gate-3', domain_id: gateDecision.id, domain_type: 'gate_decision', kind: 'checkpoint_gate', title: 'Review SFT checkpoint 750', status: 'awaiting_review', stage: 'development evidence ready', created_at: new Date().toISOString(), next_actions: ['inspect', 'continue', 'stop'] });
const dataset = {
  id: 'ds-demo', name: 'Support reasoning corpus', description: 'Curated instruction-response records for the workstation pilot.', modality: 'text', canonical_schema: 'sft', latest_version_id: datasetVersion.id,
  created_at: new Date(Date.now() - 4 * 86_400_000).toISOString(), updated_at: datasetVersion.created_at,
  sources: [{ id: 'source-demo', dataset_id: 'ds-demo', kind: 'local', uri: '/work/data/support-reasoning.jsonl', split: 'train', revision: null, fingerprint: 'a9132c7a1c8f', row_count: 1291, size_bytes: 4510022 }],
  versions: [datasetVersion], jobs: [], latest_version: datasetVersion, active_job: null,
};
runDetail.datasets = [{ role: 'train', dataset_id: dataset.id, dataset_name: dataset.name, dataset_version_id: datasetVersion.id, split: 'train', content_hash: datasetVersion.content_hash, training_artifact_id: trainingArtifact.id, artifact_hash: trainingArtifact.artifact_hash }];
runDetail.evaluations = [evaluationCandidate];
const datasetPreview = {
  items: [
    { prompt: 'A customer cannot complete setup after an interrupted update. What should the first diagnostic step be?', response: 'Confirm the installed version and inspect the most recent setup log before changing state.', category: 'diagnostics', quality_score: 0.94 },
    { prompt: 'Classify the screenshot and describe the blocking state.', response: 'The service is healthy, but the client is waiting for a stale local lock to clear.', category: 'vision', image: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="220" height="120"%3E%3Crect width="220" height="120" fill="%23221d1a"/%3E%3Crect x="18" y="18" width="184" height="84" rx="5" fill="%23352b26" stroke="%23d97a2c"/%3E%3Ccircle cx="37" cy="37" r="5" fill="%23d97a2c"/%3E%3Cpath d="M52 36h90M36 58h148M36 73h102M36 88h126" stroke="%238f817a" stroke-width="5"/%3E%3C/svg%3E' },
    { prompt: 'Summarize the attached call and retain the operator action.', response: 'Restart the managed service, then verify the health endpoint before reopening the client.', category: 'audio', audio: 'data:audio/wav;base64,UklGRgQAAABXQVZFZm10IA==' },
  ], total: 1250, offset: 0, limit: 25, split: 'train',
};

const ownDataScenario = {
  id: 'instruction-sft', revision_id: 'instruction-sft@1', revision: 1,
  label: 'Instruction and response', description: 'Teach a model to answer instructions, questions, or code tasks.',
  modality: 'text', canonical_shape: 'sft', task_type: 'supervised_fine_tuning', available: true, verified: true,
  required_fields: ['prompt', 'response'], optional_fields: ['system'], accepted_aliases: { prompt: ['prompt', 'instruction', 'question'], response: ['response', 'output', 'answer'] },
  trainer_modes: ['sft'], compatible_trainers: [{ adapter_id: 'sft', adapter_version: '1', trainer_mode: 'sft', compatible: true, required_schema: 'sft' }],
  model_families: ['Qwen2.5-Instruct'], recommended_model: models[0].id,
  default_recipe: datasetVersion.recipe, proof_run: { max_samples: 200, epochs: 1, seed: 42 },
  documentation_anchor: 'own-data/instruction-sft', common_failures: ['Prompt or response is missing.'], example_count: 1,
};
const ownDataAudioClassification = {
  id: 'audio-classification', revision_id: 'audio-classification@1', revision: 1,
  label: 'Audio classification', description: 'Classify existing audio clips with a verified audio-classification head.', modality: 'audio', canonical_shape: 'classification', task_type: 'audio_classification',
  available: true, verified: true, unavailable_reason: null, required_fields: ['media', 'label'], optional_fields: [], trainer_modes: ['classify'], proof_run: { max_samples: 50, epochs: 1, seed: 42 },
};
const ownDataCorpusScenario = {
  id: 'corpus-adaptation', revision_id: 'corpus-adaptation@1', revision: 1,
  label: 'Adapt a model to documents', description: 'Continue language-model training on a reviewed collection of documents while preserving source provenance.',
  modality: 'text', canonical_shape: 'corpus', task_type: 'continued_pretraining', available: true, verified: true,
  required_fields: ['document_id', 'document_hash', 'text', 'source_ref'], optional_fields: ['title', 'source_spans', 'timestamp', 'metadata'],
  accepted_aliases: { text: ['text', 'content', 'body'], title: ['title', 'heading'], source_ref: ['source_ref', 'source', 'path'] },
  trainer_modes: ['cpt'], compatible_trainers: [{ adapter_id: 'cpt.corpus', adapter_version: '1', trainer_mode: 'cpt', compatible: true, required_schema: 'corpus' }],
  model_families: ['Qwen2.5-Instruct'], recommended_model: models[0].id,
  default_recipe: { name: 'document-corpus', seed: 42, schema: 'corpus', steps: [{ kind: 'document_clean', preserve_headings: true, preserve_code_fences: true }, { kind: 'document_filter', quarantine_extraction_errors: true }, { kind: 'dedup', method: 'exact', field: 'text' }, { kind: 'split', method: 'random', ratios: { train: 0.9, validation: 0.1 }, seed: 42 }] },
  proof_run: { max_samples: 200, corpus_passes: 1, seed: 42 },
  documentation_anchor: 'own-data/corpus-adaptation', common_failures: ['A document has no extractable text.'], example_count: 1,
};
const ownDataExample = { id: 'instruction-sft-basic', scenario_revision_id: ownDataScenario.revision_id, label: 'Support questions', description: 'A small verified instruction and response fixture.', format: 'jsonl', filename: 'instruction-sft.jsonl', records: datasetPreview.items.slice(0, 2) };
const ownDataCorpusExample = {
  id: 'corpus-adaptation-basic',
  scenario_id: ownDataCorpusScenario.id,
  scenario_revision_id: ownDataCorpusScenario.revision_id,
  label: 'Small document collection',
  description: 'A small verified Markdown corpus with document identity and provenance.',
  expected_outcome: 'A small causal language model adapted to the language and structure of the document corpus.',
  modality: 'text',
  canonical_shape: 'corpus',
  format: 'markdown',
  filename: 'document-corpus/',
  record_count: 2,
  records: [
    { title: 'Recovery guide', text: '# Recovery guide\n\nInspect state, preserve logs, and restart only after confirming the active process.', source_ref: 'recovery-guide.md' },
    { title: 'Readiness notes', text: '# Readiness notes\n\nVerify the local health endpoint before reopening the client.', source_ref: 'readiness-notes.md' },
  ],
};
const ownDataInspection = {
  id: 'inspection-demo', import_id: 'import-demo', status: 'completed', stage: 'complete', progress_percent: 100,
  source_fingerprint: 'sha256:own-data-fixture', row_count: 1250, valid_records: 1247, invalid_records: 3, size_bytes: 4218880, sample_count: 1000, preview_policy: 'first 100 + seed-42 reservoir',
  fields: [
    { name: 'prompt', value_type: 'str', coverage: 1, present_count: 1250, null_count: 0, examples: [datasetPreview.items[0].prompt] },
    { name: 'response', value_type: 'str', coverage: 1, present_count: 1250, null_count: 0, examples: [datasetPreview.items[0].response] },
    { name: 'category', value_type: 'str', coverage: 0.98, present_count: 1225, null_count: 25, examples: ['diagnostics'] },
  ],
  preview_records: datasetPreview.items,
  schema_candidates: [{ scenario_id: ownDataScenario.id, scenario_revision_id: ownDataScenario.revision_id, label: ownDataScenario.label, confidence: 'high', score: 1, coverage: 1, required_coverage: { prompt: 1, response: 1 }, suggested_mapping: { prompt: { kind: 'direct', source: 'prompt' }, response: { kind: 'direct', source: 'response' } }, safe_transforms: [], missing_fields: [], reasons: ['Both required fields are present in every sampled record.'] }],
  parse_errors: [], warnings: [],
};
const ownDataAmbiguousInspection = {
  ...ownDataInspection,
  id: 'inspection-ambiguous',
  import_id: 'import-ambiguous',
  source_fingerprint: 'sha256:ambiguous-own-data-fixture',
  fields: [
    { name: 'input_text', value_type: 'str', coverage: 1, present_count: 1250, null_count: 0, examples: [datasetPreview.items[0].prompt] },
    { name: 'completion_text', value_type: 'str', coverage: 0.92, present_count: 1150, null_count: 100, examples: [datasetPreview.items[0].response] },
  ],
  preview_records: datasetPreview.items.slice(0, 3).map((record) => ({ input_text: record.prompt, completion_text: record.response })),
  schema_candidates: [],
  warnings: ['No scenario met the safe inference threshold.'],
};
const ownDataMappingPreview = { items: datasetPreview.items.slice(0, 3).map((record, ordinal) => ({ ordinal, source: record, canonical: { prompt: record.prompt, response: record.response }, issues: [] })), total_sampled: 3, valid_count: 3, invalid_count: 0, field_coverage: { prompt: 1, response: 1 }, ready: true, warnings: [] };
const ownDataSemanticPreview = {
  items: ownDataMappingPreview.items.map((item) => ({
    kind: 'sft',
    ordinal: item.ordinal,
    title: `Instruction ${item.ordinal + 1}`,
    summary: String(item.canonical.prompt),
    source: item.source,
    canonical: item.canonical,
    presentation: { prompt: item.canonical.prompt, response: item.canonical.response },
    issues: [],
    provenance: { scenario_revision_id: ownDataScenario.revision_id, mapping_version: 2 },
  })),
  total: ownDataMappingPreview.total_sampled,
  limit: 20,
  offset: 0,
  canonical_schema: 'sft',
  sampled: true,
};
const ownDataPreparation = { scenario_revision_id: ownDataScenario.revision_id, mapping_plan: { version: 2, scenario_revision_id: ownDataScenario.revision_id, mappings: { prompt: { kind: 'direct', source: 'prompt' }, response: { kind: 'direct', source: 'response' } }, confirmed: true }, recipe: datasetVersion.recipe, sampled: true, estimates: { accepted: 1247, quarantined: 3, duplicates: 37, split_counts: { train: 998, validation: 125, test: 124 } }, warnings: [] };
const ownDataReadiness = {
  ready: true,
  status: 'ready',
  scope: 'inspection',
  subject_id: ownDataInspection.id,
  scenario_revision_id: ownDataScenario.revision_id,
  sampled: true,
  summary: {
    headline: 'The mapped data is ready to publish',
    detail: 'Every sampled instruction has a prompt and response. Three source records will be quarantined during the exact build.',
    next_step: 'Review the preparation plan, then publish an immutable version.',
    source_records: 1250,
    preview_records: 1000,
    valid_preview_records: 997,
    invalid_preview_records: 3,
    estimated_accepted_records: 1247,
    estimated_quarantined_records: 3,
    exact_duplicate_preview_records: 37,
    token_count_is_estimated: true,
  },
  blockers: [],
  warnings: [],
  actions: [],
  split_balance: {
    train: { ratio: 0.8, estimated_records: 998 },
    validation: { ratio: 0.1, estimated_records: 125 },
    test: { ratio: 0.1, estimated_records: 124 },
  },
  compatible_trainers: ownDataScenario.compatible_trainers,
  recommended_model: models[0],
};

const reviewSchema = { id: 'schema-rev-preference-2', schema_id: 'schema-preference', revision_number: 2, content_hash: 'schemahash2', modality: 'preference', task_type: 'pairwise', definition: { modality: 'preference', task_type: 'pairwise', output_adapter_id: 'preference.v1', instruction: 'Choose the response that is more correct, complete, and directly useful.' }, created_at: new Date(Date.now() - 86_400_000).toISOString() };
const reviewQueue = { id: 'queue-demo', name: 'Reasoning regression review', status: 'active', acquisition_batch_id: 'batch-demo', schema_revision_id: reviewSchema.id, policy: { mode: 'two_pass', blind_second_pass: true, allow_suggestions: true, require_adjudication: true }, content_hash: 'queuehash1', current_pass: 1, latest_label_set_revision_id: null, created_at: new Date(Date.now() - 7200_000).toISOString(), updated_at: new Date().toISOString(), completed_at: null };
const reviewQueueVlm = { ...reviewQueue, id: 'queue-vlm', name: 'Screenshot grounding check', schema_revision_id: 'schema-rev-vlm-1', policy: { mode: 'one_pass', blind_second_pass: false, allow_suggestions: false, require_adjudication: false }, content_hash: 'queuehash2' };
const reviewItems = [
  { id: 'review-item-1', queue_id: reviewQueue.id, candidate_id: 'candidate-1', ordinal: 0, status: 'pending', active_event_id: null, projection: { status: 'pending', current_pass: 1 }, record_id: 'record-0042', record_hash: 'recordhash42', record: { prompt: 'Resolve 3x + 4 = 19 and explain the reasoning.', chosen: 'Subtract four from both sides to get 3x = 15, then divide by three: x = 5.', rejected: 'Move the four and divide nineteen by three, so x is about 7.' }, evidence: { outcome: 'regression', base_passed: true, candidate_passed: false, score: 0 }, source: { kind: 'evaluation_comparison', ref: evaluationCandidate.id }, created_at: new Date().toISOString(), updated_at: new Date().toISOString() },
  { id: 'review-item-2', queue_id: reviewQueue.id, candidate_id: 'candidate-2', ordinal: 1, status: 'pass1_complete', active_event_id: 'event-2', projection: { status: 'pass1_complete', current_pass: 1, pass_1: { event_id: 'event-2', annotation: { chosen: 'A' } } }, record_id: 'record-0088', record_hash: 'recordhash88', record: { prompt: 'A tray has 24 items and loses one quarter.', chosen: 'Eighteen items remain.', rejected: 'Sixteen items remain.' }, evidence: { outcome: 'improvement', score: 1 }, source: { kind: 'evaluation_comparison', ref: evaluationCandidate.id }, created_at: new Date().toISOString(), updated_at: new Date().toISOString() },
  { id: 'review-item-3', queue_id: reviewQueue.id, candidate_id: 'candidate-3', ordinal: 2, status: 'conflict', active_event_id: 'event-3b', projection: { status: 'conflict', current_pass: 2, pass_1: { event_id: 'event-3a', annotation: { chosen: 'A' } }, pass_2: { event_id: 'event-3b', annotation: { chosen: 'B' } } }, record_id: 'record-0104', record_hash: 'recordhash104', record: { prompt: 'Choose the safer recovery procedure.', chosen: 'Inspect state, save logs, then restart.', rejected: 'Delete state immediately and retry.' }, evidence: { verifier_disagreement: true }, source: { kind: 'evaluation_comparison', ref: evaluationCandidate.id }, created_at: new Date().toISOString(), updated_at: new Date().toISOString() },
];
const reviewStats = { queue_id: reviewQueue.id, total: 3, resolved: 1, coverage: 1 / 3, status_counts: { pending: 1, pass1_complete: 1, conflict: 1 }, excluded: 0, flagged: 0, conflicts: 1, two_pass_compared: 1, two_pass_agreements: 0, two_pass_agreement_rate: 0, event_counts: { label: 3 }, correction_rate: 0, unpublished_changes: false, event_stream_hash: 'eventhash' };
const reviewCapabilities = { modalities: [{ id: 'text', task_types: ['binary', 'categorical', 'text_correction'] }, { id: 'preference', task_types: ['binary', 'pairwise', 'ranking'] }, { id: 'tool', task_types: ['binary', 'structured_correction'] }, { id: 'vlm', task_types: ['binary', 'categorical', 'text_correction'] }, { id: 'audio', task_types: ['binary', 'categorical', 'text_correction'] }], acquisition_source_kinds: ['evaluation', 'evaluation_comparison', 'verifier_calibration', 'dataset_version', 'run_samples', 'playground_session', 'jsonl'], verifier_failure_selectors: ['false_accept', 'false_reject', 'high_confidence_disagreement', 'repeat_instability', 'order_flip', 'ranking_inversion', 'threshold_adjacent', 'parser_runtime', 'subgroup', 'chain_component'], acquisition_strategies: ['explicit', 'candidate_failure', 'regression', 'improvement', 'verifier_disagreement', 'low_score', 'low_margin', 'coverage_gap', 'diversity', 'random'], review_policies: ['one_pass', 'two_pass'], event_types: ['label', 'correct', 'exclude', 'flag', 'reveal_suggestion', 'adjudicate'], output_adapters: [{ id: 'preference.v1', version: 1, modalities: ['text', 'preference', 'vlm'], task_types: ['pairwise', 'ranking'], build_modes: ['append', 'replace_by_record_id'], default_build_mode: 'append' }], max_event_batch_size: 1000, protected_suite_purposes: ['operational', 'holdout', 'final_holdout'], protected_splits: ['test', 'canary'] };
const verifierCapability = { id: 'exact-match.reliability.v1', family: 'deterministic', label: 'Exact match', description: 'Deterministic normalized answer equality.', implementation: 'exact_match', implementation_fingerprint: 'sha256:verifier-exact', origin: 'builtin', fingerprintable: true, modalities: ['text'], task_types: ['binary', 'categorical'], supports_probability: false, supports_seed: true, compatible_consumers: ['dataset_scoring', 'evaluation', 'raft', 'grpo'] };
const verifierRevision = { id: 'verifier-rev-exact-2', profile_id: 'verifier-profile-exact', revision_number: 2, family: 'deterministic', modality: 'text', task_type: 'binary', implementation_id: verifierCapability.id, implementation_fingerprint: verifierCapability.implementation_fingerprint, reliability_adapter_id: 'binary.reliability', reliability_adapter_version: 1, reward_contract: { minimum: 0, maximum: 1, direction: 'maximize', threshold: 0.5, tie_policy: 'reject', probability_semantics: false, error_behavior: 'propagate' }, content_hash: 'verifierrevhash2', qualification_state: 'pass', alias: 'candidate', runtime_compatible: true, created_at: new Date(Date.now() - 86_400_000).toISOString() };
const verifierProfile = { id: verifierRevision.profile_id, name: 'Exact answer oracle', description: 'Strict normalized equality for reviewed short answers.', latest_revision_id: verifierRevision.id, latest_revision: verifierRevision, revision_count: 2, created_at: verifierRevision.created_at, updated_at: new Date().toISOString() };
const verifierDecision = { id: 'verifier-decision-1', calibration_id: 'verifier-calibration-1', profile_revision_id: verifierRevision.id, qualification_profile_revision_id: 'verifier-qualification-strict', decision: 'pass', scope: 'development', reasons: ['Balanced accuracy 1.000 meets 0.980.', 'False-accept and error rates are zero.', 'Fresh-process repeat agreement is exact.'], evidence_count: 125, created_at: new Date().toISOString() };
const verifierCalibration = { id: verifierDecision.calibration_id, profile_revision_id: verifierRevision.id, profile_revision: verifierRevision, source_kind: 'label_set_revision', source_revision_id: 'label-rev-1', source_purpose: 'development', source_hash: 'labelsourcehash1', source_name: 'Reviewed short answers', protocol_revision_id: 'verifier-protocol-default', qualification_profile_revision_id: verifierDecision.qualification_profile_revision_id, status: 'completed', stage: 'published', processed_records: 125, total_records: 125, progress_percent: 100, primary_metric: { name: 'balanced_accuracy', value: 1, lower_ci: 1, upper_ci: 1, record_count: 125 }, qualification: verifierDecision, work_item_id: 'work-verifier-1', evidence_hash: 'evidencehash1', runtime_hash: 'runtimehash1', request_hash: 'requesthash1', completed_at: new Date().toISOString() };
const verifierMetrics = [{ name: 'balanced_accuracy', value: 1, lower_ci: 1, upper_ci: 1, direction: 'maximize', record_count: 125, details: { primary: true, confusion_matrix: { labels: ['reject', 'accept'], matrix: [[62, 0], [0, 63]] }, per_class: { reject: { precision: 1, recall: 1 }, accept: { precision: 1, recall: 1 } }, threshold_curve: [{ threshold: 0.25, accuracy: 0.93, false_accept_rate: 0.06, false_reject_rate: 0.08 }, { threshold: 0.5, accuracy: 1, false_accept_rate: 0, false_reject_rate: 0 }, { threshold: 0.75, accuracy: 0.94, false_accept_rate: 0, false_reject_rate: 0.12 }] } }, { name: 'coverage', value: 1, lower_ci: 1, upper_ci: 1, direction: 'maximize', record_count: 125 }, { name: 'error_rate', value: 0, lower_ci: 0, upper_ci: 0, direction: 'minimize', record_count: 125 }, { name: 'repeat_agreement', value: 1, lower_ci: 1, upper_ci: 1, direction: 'maximize', record_count: 125 }];
const verifierSamples = reviewItems.map((item, index) => ({ id: `verifier-sample-${index}`, calibration_id: verifierCalibration.id, record_id: item.record_id, record_hash: item.record_hash, group_id: item.record_id, split: index ? 'calibration' : 'confirmation', task_type: 'binary', orientation: null, perturbation: null, repeat_index: index % 2, seed: 42, expected: true, input: item.record, observation: { reward: 1, passed: true, parsed_value: true, raw_output: 'match', details: { normalized: true }, component_trace: [], latency_ms: 2, error: null, runtime_identity: { python: '3.12' } }, agreement: true }));
const signalCapability = { id: 'raft-mlx-signal-v1', version: 1, trainer_mode: 'raft', backend_family: 'mlx', boundary_unit: 'cycle', resumable: true, audit_boundaries: ['cycle'], capture_fidelity: 'sampled', candidate_multiplicity: 'multiple', mappings: { identity: ['record_id', 'record_hash', 'instance_id'], input: ['prompt'], output: ['response'], verifier: ['reward', 'passed'] }, unavailable_fields: [] };
const rewardSystemRevision = { id: 'reward-system-rev-1', system_id: 'reward-system-1', revision_number: 1, content_hash: 'rewardsystemhash1', optimizer_verifier_revision_id: verifierRevision.id, optimizer_verifier_profile_revision_id: verifierRevision.id, modality: 'text', task_type: 'binary', reward_mapping: { minimum: 0, maximum: 1, direction: 'maximize', threshold: 0.5 }, input_mapping: { prompt: 'prompt', output: 'response' }, definition: { shaping: { keep_policy: 'passed' } }, auditors: [{ id: 'reward-auditor-1', role: 'primary_sentinel', ordinal: 0, verifier_revision_id: 'verifier-rev-sentinel-3', verifier_profile_revision_id: 'verifier-rev-sentinel-3', correlated: false, correlation_reasons: [] }], qualification_state: 'ready', created_at: new Date(Date.now() - 86_400_000).toISOString() };
const rewardSystem = { id: rewardSystemRevision.system_id, name: 'Independent answer reward', description: 'Optimizer plus a disjoint qualified sentinel for same-output scoring.', latest_revision_id: rewardSystemRevision.id, latest_revision: rewardSystemRevision, revision_count: 1, created_at: rewardSystemRevision.created_at, updated_at: new Date().toISOString() };
const rewardProtocol = { id: 'reward-protocol-balanced-1', protocol_id: 'reward-protocol-balanced', revision_number: 1, name: 'Balanced 256', template: 'balanced_256', uniform_core_limit: 192, diagnostic_limit: 64, seed: 42, capture_required_for_gating: true, content_hash: 'rewardprotocolhash1' };
const rewardIntegrityProfile = { id: 'reward-integrity-human-1', profile_id: 'reward-integrity-human', revision_number: 1, name: 'Human-aligned integrity', template: 'human_aligned_integrity', minimum_pass_records: 100, minimum_report_records: 20, bootstrap_resamples: 10000, bootstrap_seed: 42, promotable: true, content_hash: 'rewardintegrityprofilehash1' };
const rewardAuditMetrics = [{ name: 'paired_coverage', value: 0.984, lower_ci: 0.971, upper_ci: 0.993, direction: 'maximize', record_count: 125 }, { name: 'pass_agreement', value: 0.872, lower_ci: 0.821, upper_ci: 0.916, direction: 'maximize', record_count: 125 }, { name: 'optimizer_only_acceptance', value: 0.112, lower_ci: 0.068, upper_ci: 0.171, direction: 'minimize', record_count: 125 }, { name: 'normalized_mean_gap', value: 0.146, lower_ci: 0.091, upper_ci: 0.203, direction: 'minimize', record_count: 125 }, { name: 'spearman', value: 0.72, lower_ci: 0.62, upper_ci: 0.8, direction: 'maximize', record_count: 125 }, { name: 'top_tail_disagreement', value: 0.16, lower_ci: 0.1, upper_ci: 0.23, direction: 'minimize', record_count: 125 }];
const rewardAuditDecision = { id: 'reward-audit-decision-1', audit_id: 'reward-audit-1', decision: 'fail', action: 'awaiting_review', reasons: ['Optimizer-only acceptance 0.112 exceeds the 0.100 pass threshold.', 'Top-tail disagreement is outside the human-aligned pass band.'], record_count: 125, automatic: true, created_at: new Date().toISOString() };
const rewardAudit = { id: rewardAuditDecision.audit_id, run_id: 'demo-run', segment_id: 'segment-3', boundary_index: 2, boundary_value: 750, boundary_unit: 'step', checkpoint_artifact_id: modelArtifact.id, training_signal_shard_id: 'signal-shard-3', reward_system_revision_id: rewardSystemRevision.id, protocol_revision_id: rewardProtocol.id, integrity_profile_revision_id: rewardIntegrityProfile.id, status: 'completed', stage: 'published', processed_records: 125, total_records: 125, progress_percent: 100, capture_fidelity: 'sampled', metrics: rewardAuditMetrics, decision: rewardAuditDecision, work_item_id: 'work-reward-audit-1', evidence_hash: 'rewardauditevidencehash1', trace_hash: 'signaltracehash3', completed_at: new Date().toISOString() };
const rewardAuditSamples = reviewItems.map((item, index) => ({ id: `reward-observation-${index}`, audit_id: rewardAudit.id, snapshot_id: `snapshot-${index}`, record: { record_id: item.record_id, record_hash: item.record_hash, instance_id: `instance-${index}`, identity_kind: 'managed', dataset_version_id: datasetVersion.id, split: 'train' }, boundary_index: rewardAudit.boundary_index, candidate_ordinal: index, prompt: item.record.prompt, output: index === 0 ? item.record.rejected : item.record.chosen, expected: item.record.chosen, optimizer_observation: { reward: index === 0 ? 0.96 : 0.88, passed: true, component_trace: [{ component: 'optimizer', reward: index === 0 ? 0.96 : 0.88 }] }, sentinel_observation: { reward: index === 0 ? 0.21 : 0.84, passed: index !== 0, component_trace: [{ component: 'sentinel', reward: index === 0 ? 0.21 : 0.84 }] }, normalized_optimizer_reward: index === 0 ? 0.96 : 0.88, normalized_sentinel_reward: index === 0 ? 0.21 : 0.84, reward_gap: index === 0 ? 0.75 : 0.04, classification: index === 0 ? 'optimizer_only_accept' : 'agreement', capture_stratum: index === 0 ? 'diagnostic' : 'uniform_core' }));
activitySnapshot.items.unshift({ id: 'work-reward-audit-1', work_item_id: 'work-reward-audit-1', domain_id: rewardAudit.id, domain_type: 'reward_integrity_audit', kind: 'reward_integrity_audit', title: 'Training signal audit · boundary 3', status: 'awaiting_review', stage: 'decision published', progress_current: 125, progress_total: 125, progress_percent: 100, attempt: 1, max_attempts: 3, created_at: new Date(Date.now() - 90_000).toISOString(), started_at: new Date(Date.now() - 80_000).toISOString(), heartbeat_at: new Date().toISOString(), next_actions: ['open_audit', 'continue', 'stop', 'fork'] });

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
page.on('console', msg => {
  if (msg.type() !== 'error') return;
  const location = msg.location().url;
  if (msg.text().includes('404') && location.includes('/workspace-drafts/')) return;
  consoleErrors.push(location ? `${msg.text()} (${location})` : msg.text());
});
page.on('pageerror', err => consoleErrors.push(err.message));

await page.route('**/api/public/**', async route => {
  const url = new URL(route.request().url());
  const path = url.pathname.replace('/api/public', '');
  const json = data => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(data) });
  if (path === '/telemetry/stream' || path.endsWith('/events') || path.endsWith('/logs/stream')) {
    const body = path.endsWith('/logs/stream')
      ? `data: ${JSON.stringify({ reset: true, log_path: '/models/demo-run/run.log', lines: ['[OK] backend ready', 'Iter 84: loss=1.234 lr=2e-4', 'saved checkpoint preview'] })}\n\n`
      : `data: ${JSON.stringify(path === '/telemetry/stream' ? telemetry : { ...runDetail, progress_percent: 42, current_step: 84, total_steps: 200, current_cycle: 1, total_cycles: 1, latest_loss: 1.234, latest_learning_rate: 0.0002 })}\n\n`;
    return route.fulfill({ status: 200, contentType: 'text/event-stream', body });
  }
  if (path === '/health') return json({ ok: true });
  if (path === '/workspace') return json({ default_run_root: '/tmp/halo-forge/runs', runs_dir: '/tmp/halo-forge/runs', writable: true, message: 'Ready' });
  if (path === '/version') return json({ package_version: '2.0.0a2', display_version: '2.0.0-alpha-2', release_channel: 'alpha' });
  if (path === '/backend') return json(backend);
  if (path === '/interface-capabilities') return json({ items: [
    { id: 'desktop-macos-arm64', kind: 'execution_surface', label: 'Desktop · macOS arm64', execution_surface: 'desktop', status: 'alpha', available: true },
    { id: 'local-browser', kind: 'execution_surface', label: 'Local browser', execution_surface: 'local_browser', status: 'supported', available: true },
    { id: 'remote-browser', kind: 'execution_surface', label: 'Remote browser', execution_surface: 'remote_browser', status: 'supported', available: true },
    { id: 'cli', kind: 'execution_surface', label: 'CLI', execution_surface: 'cli', status: 'supported', available: true },
  ], total: 4 });
  if (path === '/training-scenarios') return json({ items: [ownDataScenario, ownDataCorpusScenario, ownDataAudioClassification], total: 3, limit: 100, offset: 0 });
  if (path === '/training-scenario-examples') return json({ items: [ownDataExample, ownDataCorpusExample], total: 2, limit: 100, offset: 0 });
  if (path === `/training-scenarios/${ownDataScenario.id}`) return json(ownDataScenario);
  if (path === `/training-scenarios/${ownDataScenario.id}/examples`) return json({ items: [ownDataExample], total: 1, limit: 100, offset: 0 });
  if (path === `/training-scenarios/${ownDataScenario.id}/template`) return json(ownDataExample);
  if (path === `/training-scenarios/${ownDataCorpusScenario.id}`) return json(ownDataCorpusScenario);
  if (path === `/training-scenarios/${ownDataCorpusScenario.id}/examples`) return json({ items: [ownDataCorpusExample], total: 1, limit: 100, offset: 0 });
  if (path === `/training-scenarios/${ownDataCorpusScenario.id}/template`) return json(ownDataCorpusExample);
  if (path === '/document-extractors') return json({ items: [
    { id: 'plain-text-v1', label: 'Plain text', version: '1', available: true, source_kinds: ['file', 'directory'], extensions: ['.txt'], preserves: ['paragraphs', 'source spans'], limitations: [] },
    { id: 'markdown-v1', label: 'Markdown', version: '1', available: true, source_kinds: ['file', 'directory'], extensions: ['.md', '.markdown'], preserves: ['headings', 'code fences', 'source spans'], limitations: [] },
    { id: 'visible-html-v1', label: 'Visible HTML', version: '1', available: true, source_kinds: ['file', 'directory'], extensions: ['.html', '.htm'], preserves: ['headings', 'paragraphs', 'source reference'], limitations: ['Scripts, styles, and hidden content are excluded.'] },
    { id: 'docx-v1', label: 'Word document', version: '1', available: true, source_kinds: ['file', 'directory'], extensions: ['.docx'], preserves: ['headings', 'paragraphs', 'tables'], limitations: [] },
    { id: 'pdf-text-v1', label: 'PDF text layer', version: '1', available: true, source_kinds: ['file', 'directory'], extensions: ['.pdf'], preserves: ['page spans', 'source reference'], limitations: ['Image-only pages require OCR outside Halo Forge.'] },
    { id: 'structured-text-v1', label: 'Structured text rows', version: '1', available: true, source_kinds: ['file', 'directory'], extensions: ['.json', '.jsonl', '.jl', '.csv', '.tsv', '.parquet'], preserves: ['row identity', 'selected metadata', 'source reference'], limitations: [] },
  ], total: 6, limit: 100, offset: 0 });
  if (path === '/dataset-imports/huggingface/options') return json({ repo_id: 'acme/support-preferences', requested_revision: 'main', resolved_revision: '98c14935ec62da9db8e0f354a9f73424a638bdb7', items: [{ config: 'default', splits: ['train', 'validation'] }, { config: 'clean', splits: ['train'] }], total: 2, limit: 2, offset: 0 });
  if (path === '/dataset-imports') return json({ id: 'import-demo', status: 'ready', source_kind: 'example', scenario_revision_id: ownDataScenario.revision_id, files: [], total_files: 1, total_bytes: 4096, uploaded_bytes: 4096, inspection_id: null, work_item_id: null });
  if (path === '/dataset-imports/import-demo/inspect') return json({ inspection: ownDataInspection, work_item_id: 'work-inspection-demo' });
  if (path === '/dataset-imports/import-demo') return json({ id: 'import-demo', status: 'completed', source_kind: 'example', scenario_revision_id: ownDataScenario.revision_id, inspection_id: ownDataInspection.id });
  if (path === `/dataset-inspections/${ownDataInspection.id}/mapping-preview`) return json(ownDataMappingPreview);
  if (path === `/dataset-inspections/${ownDataInspection.id}/semantic-preview`) return json(ownDataSemanticPreview);
  if (path === `/dataset-inspections/${ownDataInspection.id}/preparation-preview`) return json(ownDataPreparation);
  if (path === `/dataset-inspections/${ownDataInspection.id}/readiness`) return json(ownDataReadiness);
  if (path === `/dataset-inspections/${ownDataInspection.id}/register`) return json({ dataset, source: dataset.sources[0] });
  if (path === `/dataset-inspections/${ownDataAmbiguousInspection.id}`) return json(ownDataAmbiguousInspection);
  if (path === `/dataset-inspections/${ownDataInspection.id}`) return json(ownDataInspection);
  if (path === '/telemetry') return json(telemetry);
  if (path === '/train/datasets') return json({ items: [
    { key: 'codealpaca', huggingface_id: 'sahil2801/CodeAlpaca-20k', description: '20K instruction-following code examples', domain: 'code', size_hint: '20K', default_split: 'train' },
    { key: 'gsm8k_sft', huggingface_id: 'gsm8k', description: '8.5K grade school math for SFT', domain: 'reasoning', size_hint: '8.5K', default_split: 'train' },
    { key: 'xlam_sft', huggingface_id: 'Salesforce/xlam-function-calling-60k', description: '60K function calling examples', domain: 'agentic', size_hint: '60K', default_split: 'train' },
  ] });
  if (path === '/train/dataset-versions') return json({ items: [datasetVersion] });
  if (path === '/train/models') return json({ items: models });
  if (path === '/models') return json({ catalog_version: '2026.05', items: models, total: models.length, facets: { providers: ['Liquid AI', 'Qwen', 'mlx-community'], statuses: ['recommended', 'experimental'], modalities: ['text', 'code'], memory_tiers: ['tiny', 'small'], risk_levels: ['safe', 'experimental'] }, filters: {} });
  if (path === '/model-artifacts') return json({ items: [modelArtifact, baseArtifact], total: 2, limit: 200, offset: 0 });
  if (path === '/artifact-operations') return json({ items: [{ id: 'op-1', kind: 'convert', status: 'running', input_artifact_ids: [modelArtifact.id], work_item_id: 'work-1', created_at: new Date().toISOString() }], total: 1 });
  if (path === '/qualifications/compare') return json({ profile_revision_id: 'local-gguf-r3', base_qualification_id: 'qual-base', candidate_qualification_id: 'qual-1', deltas: [{ stage: 'development', metric: 'accuracy', direction: 'maximize', parent_value: 0.69, candidate_value: 0.68, raw_delta: -0.01, favorable_delta: -0.01 }, { stage: 'operational', metric: 'output_tokens_per_second', direction: 'maximize', parent_value: 54.1, candidate_value: 72.4, raw_delta: 18.3, favorable_delta: 18.3 }] });
  if (path === '/qualifications') { const artifactId = url.searchParams.get('artifact_id'); const items = artifactId ? qualifications.filter(item => item.artifact_id === artifactId) : qualifications; return json({ items, total: items.length }); }
  if (path === '/qualification-profiles') return json({ items: [{ id: 'local-gguf-r3', name: 'Local GGUF', development_suite_revision_id: suiteRevision.id, operational_suite_revision_id: 'ops-r1', metrics: [], target_backend: 'llama.cpp' }], total: 1 });
  if (path === '/activity') return json(activitySnapshot);
  if (path === '/workers') return json({ items: [activitySnapshot.worker] });
  if (path === '/storage') return json(storageInventory);
  if (path === '/serve/status') return json({ running: false, state: 'idle', ready_state: 'idle', pid: null, model: null, backend: null, host: '127.0.0.1', port: 8001, url: 'http://127.0.0.1:8001/v1', started_at: null, exit_code: null, log_path: null, logs_available: false, last_error: null, healthy: false, message: 'No managed model is serving.' });
  if (path === '/playground/sessions') return json({ items: [playgroundSession], total: 1, limit: 100, offset: 0 });
  if (path === '/registry') return json({ items: [{ id: 1, name: 'Three-seed SFT repeat', description: 'Stable candidate comparison', base_model: models[0].id, run_ids: ['demo-run'], tags: ['candidate'], created_at: new Date().toISOString(), updated_at: new Date().toISOString() }] });
  if (path === '/checkpoint-policies') return json({ items: [checkpointPolicy], total: 1 });
  if (path === '/trainer-execution-capabilities') return json({ items: [{ capability_id: 'sft-hf-step-v1', version: 1, trainer_mode: 'sft', backend_family: 'hf', segment_unit: 'step', supports_gated_execution: true, resume_parameter: 'resume_from_checkpoint', resume_cli_flag: '--resume-from-checkpoint', checkpoint_pattern: 'checkpoint-*', checkpoint_index: 'filesystem', reason: null }, { capability_id: 'sft-mlx-full-trial-v1', version: 1, trainer_mode: 'sft', backend_family: 'mlx', segment_unit: 'full_trial', supports_gated_execution: false, resume_parameter: null, resume_cli_flag: null, checkpoint_pattern: null, checkpoint_index: 'filesystem', reason: 'MLX remains final-only until bounded resume is verified.' }] });
  if (path === '/run-groups/group-stability/trajectory') return json(trajectory);
  if (path === '/run-groups/group-stability/analyses') return json({ items: [cohortAnalysis], total: 1 });
  if (path === '/run-groups/group-stability') return json(runGroup);
  if (path === '/run-groups') return json({ items: [runGroup] });
  if (path === '/research-decisions') return json({ items: [{ id: 'decision-1', analysis_snapshot_id: cohortAnalysis.id, selected_subject: { trial_id: 'trial-1' }, rejected_subjects: [], exclusions: [], rationale: 'Matched-seed evidence supports the guarded SFT candidate without a practical regression.', fork_spec: { run_group_id: runGroup.id, trial_id: 'trial-1' }, content_hash: 'decisionhash1', created_at: new Date().toISOString() }], total: 1 });
  if (path.startsWith('/workspace-drafts/')) return route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'draft not found' }) });
  if (path === '/search') return json({ items: [{ id: runGroup.id, type: 'run_group', label: runGroup.name, description: 'Three-seed guarded SFT repeat', status: runGroup.status, short_id: runGroup.id, target: '/sweeps' }], total: 1, query: url.searchParams.get('q') });
  if (path === '/work-items') return json({ items: activitySnapshot.items, active_lease: activitySnapshot.resource_lease });
  if (path === '/train/preflight') return json({ mode: 'sft', ok: true, resolved_paths: {}, errors: [], warnings: [], suggested_fixes: [], user_summary: { headline: 'Ready to launch', next_step: 'Start the run when you are ready.', confidence_tone: 'success' } });
  if (path === '/dashboard') return json({ readiness_tier: 'qualified' });
  if (path === '/datasets') return json({ items: [dataset] });
  if (path === '/datasets/ds-demo/preview') return json(datasetPreview);
  if (path === '/datasets/ds-demo/statistics') return json(datasetVersion.statistics);
  if (path === '/datasets/ds-demo/versions') return json({ items: [datasetVersion] });
  if (path === '/datasets/ds-demo/build') return json({ id: 'job-own-data-build', job_id: 'job-own-data-build', status: 'queued', dataset_id: dataset.id, version_id: datasetVersion.id });
  if (path === '/datasets/ds-demo') return json(dataset);
  if (path === '/dataset-jobs') return json({ items: [] });
  if (path === '/dataset-versions/v-demo-001/preview') return json(datasetPreview);
  if (path === '/dataset-versions/v-demo-001/statistics') return json(datasetVersion.statistics);
  if (path === '/dataset-versions/v-demo-001') return json(datasetVersion);
  if (path === '/dataset-versions/v-demo-001/readiness') return json({ ready: true, status: 'ready', blockers: [], warnings: [], compatible_trainers: ownDataScenario.compatible_trainers, recommended_model: models[0] });
  if (path === '/dataset-versions/v-demo-001/proof-run') return json({ accepted: true, status: 'running', run_id: 'demo-run', proof_run: true, work_item_id: 'work-proof-demo' });
  if (path === '/dataset-versions/v-demo-001/training-artifacts') return json({ items: [trainingArtifact] });
  if (path === '/dataset-versions/v-demo-001/runs') return json({ items: [{ run_id: 'demo-run', modality: 'sft', model_name: models[0].id, status: 'completed' }] });
  if (path === '/benchmark-suites') return json({ items: [suite, operationalSuite] });
  if (path === '/benchmark-suites/suite-core') return json(suite);
  if (path === '/evaluation-jobs') return json({ items: [] });
  if (path === '/evaluations/compare') return json(evaluationComparison);
  if (path === '/evaluations/history') return json({ items: [{ ...evaluationCandidate, history_ordinal: 1, primary_value: 0.68 }, { ...evaluationCandidatePrevious, history_ordinal: 0, primary_value: 0.64 }], total: 2, subject_ref: null, suite_revision_id: suiteRevision.id, limit: 40 });
  if (path === '/evaluations/drift') return json({ ...evaluationComparison, classification: 'improved', practical_delta: 0, compatible: true, history_contract: { suite_revision_id: suiteRevision.id, direction: 'maximize', comparison: 'immutable_evaluation_pair' } });
  if (path === '/evaluations') return json({ items: [evaluationCandidate, evaluationBase] });
  if (path === '/verifiers') return json({ items: [{ name: 'exact_match', cls: 'halo_forge.rlvr.verifiers.ExactMatchVerifier', origin: 'builtin', module: 'halo_forge.rlvr.verifiers', doc: 'Strict normalized answer equality.', base: 'Verifier' }], counts: { builtin: 1, user_plugin: 0, entry_point: 0 }, plugin_dir: '~/.halo-forge/verifiers', total: 1 });
  if (path === '/verifier-reliability/capabilities') return json({ items: [verifierCapability], qualification_templates: [], max_evaluation_candidates: 4 });
  if (path === '/reward-integrity-capabilities') return json({ items: [signalCapability], default_protocols: [rewardProtocol], default_profiles: [rewardIntegrityProfile] });
  if (path === '/reward-systems') return json({ items: [rewardSystem], total: 1, limit: 200, offset: 0 });
  if (path === `/reward-systems/${rewardSystem.id}`) return json(rewardSystem);
  if (path === '/reward-audit-protocols') return json({ items: [rewardProtocol], total: 1, limit: 100, offset: 0 });
  if (path === '/reward-integrity-profiles') return json({ items: [rewardIntegrityProfile], total: 1, limit: 100, offset: 0 });
  if (path === '/reward-integrity-audits/compare') return json({ base_audit_id: rewardAudit.id, candidate_audit_id: rewardAudit.id, compatible: true, comparison_kind: 'paired_snapshot', compatibility_reasons: [], shared_record_count: 125, metrics: rewardAuditMetrics.map(metric => ({ name: metric.name, base_value: metric.value, candidate_value: metric.value, raw_delta: 0, favorable_delta: 0, direction: metric.direction })) });
  if (path === '/reward-integrity-audits') return json({ items: [rewardAudit], total: 1, limit: 200, offset: 0 });
  if (path === `/reward-integrity-audits/${rewardAudit.id}/metrics`) return json({ items: rewardAuditMetrics, total: rewardAuditMetrics.length, limit: 100, offset: 0 });
  if (path === `/reward-integrity-audits/${rewardAudit.id}/samples`) return json({ items: rewardAuditSamples, total: 125, limit: 50, offset: 0 });
  if (path === `/reward-integrity-audits/${rewardAudit.id}`) return json(rewardAudit);
  if (path === '/verifier-profiles') return json({ items: [verifierProfile], total: 1, limit: 100, offset: 0 });
  if (path === `/verifier-profiles/${verifierProfile.id}`) return json({ ...verifierProfile, revisions: [verifierRevision] });
  if (path === '/verifier-calibrations') return json({ items: [verifierCalibration], total: 1, limit: 100, offset: 0 });
  if (path === `/verifier-calibrations/${verifierCalibration.id}`) return json(verifierCalibration);
  if (path === `/verifier-calibrations/${verifierCalibration.id}/metrics`) return json({ items: verifierMetrics, total: verifierMetrics.length, limit: 100, offset: 0 });
  if (path === `/verifier-calibrations/${verifierCalibration.id}/samples`) return json({ items: verifierSamples, total: 125, limit: 50, offset: 0 });
  if (path === `/verifier-profile-revisions/${verifierRevision.id}`) return json(verifierRevision);
  if (path === '/verifier-calibration-protocols') return json({ items: [{ id: 'verifier-protocol-default', name: 'Family default replicated protocol', repeats: 2, seeds: [17, 42, 101], bootstrap_resamples: 10000, bootstrap_seed: 42 }], total: 1, limit: 100, offset: 0 });
  if (path === '/verifier-calibration-protocol-revisions/verifier-protocol-default') return json({ protocol: { id: 'verifier-protocol', name: 'Family default replicated protocol' }, revision: { id: 'verifier-protocol-default', protocol_id: 'verifier-protocol', revision_number: 1, definition: { repeats: 2, stochastic_seeds: [17, 42, 101], bootstrap_resamples: 10000, bootstrap_seed: 42, concurrency: 1 } } });
  if (path === '/verifier-qualification-profiles') return json({ items: [{ id: 'verifier-qualification-strict', name: 'Strict oracle', template: 'strict_oracle', promotable: true }], total: 1, limit: 100, offset: 0 });
  if (path === '/verifier-qualification-profile-revisions/verifier-qualification-strict') return json({ profile: { id: 'verifier-qualification', name: 'Strict oracle' }, revision: { id: 'verifier-qualification-strict', profile_id: 'verifier-qualification', revision_number: 1, template_kind: 'strict_oracle', promotable: true, requirements: { primary_agreement: { pass: 0.98, warn: 0.95 } } } });
  if (path === '/verifier-qualifications') return json({ items: [verifierDecision], total: 1, limit: 100, offset: 0 });
  if (path === `/verifier-profile-revisions/${verifierRevision.id}/runtime-compatibility`) return json({ profile_revision_id: verifierRevision.id, status: 'compatible', compatible: true, expected: { python: '3.12' }, observed: { python: '3.12' }, differences: [], checked_at: new Date().toISOString() });
  if (path === `/verifier-profile-revisions/${verifierRevision.id}/usage`) return json({ items: [{ id: evaluationCandidate.id, kind: 'evaluation', role: 'scorer', label: 'Core reasoning · candidate', created_at: new Date().toISOString() }], total: 1 });
  if (path === '/runs/search') return json({ items: [{ run_id: 'demo-run', modality: 'sft', model_name: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', status: 'running', created_at: new Date().toISOString(), cycles_executed: 1, weights_updated: true, final_train_loss: 1.234, effectiveness: { verdict: 'review' } }], total: 1, filters: {}, facets: { modalities: ['sft'], modality_counts: { sft: 1 }, models: ['mlx-community/Qwen2.5-0.5B-Instruct-bf16'] } });
  if (path === '/runs') return json({ items: [{ run_id: 'demo-run', modality: 'sft', model_name: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', status: 'running', created_at: new Date().toISOString(), cycles_executed: 1, weights_updated: true, final_train_loss: 1.234, effectiveness: { verdict: 'review' } }] });
  if (path === '/runs/demo-run') return json(runDetail);
  if (path === '/runs/demo-run/full-run') return json({ accepted: true, status: 'running', run_id: 'full-run-demo', parent_run_id: 'demo-run' });
  if (path === '/runs/demo-run/reward-integrity-audits') return json({ items: [rewardAudit], total: 1, limit: 200, offset: 0 });
  if (path === '/runs/demo-run/training-signals') return json({ items: [{ id: rewardAudit.training_signal_shard_id, run_id: 'demo-run', segment_id: rewardAudit.segment_id, boundary_index: rewardAudit.boundary_index, boundary_value: rewardAudit.boundary_value, boundary_unit: rewardAudit.boundary_unit, status: 'sealed', capture_fidelity: 'sampled', observed_count: 1000, retained_count: 125, core_count: 100, diagnostic_count: 25, retained_set_hash: 'retainedhash', trace_hash: rewardAudit.trace_hash, sealed_at: rewardAudit.completed_at }], total: 1, limit: 100, offset: 0 });
  if (path === '/runs/demo-run/launch-config') return json({ run_id: 'demo-run', resolved_config: { mode: 'sft', model: models[0].id, epochs: 1, batch_size: 2, learning_rate: 0.0002, max_samples: 1000 }, datasets: [{ role: 'train', dataset_id: dataset.id, dataset_name: dataset.name, dataset_version_id: datasetVersion.id, split: 'train', content_hash: datasetVersion.content_hash, training_artifact_id: trainingArtifact.id, artifact_hash: trainingArtifact.artifact_hash }] });
  if (path === '/runs/demo-run/lineage') return json({ run_id: 'demo-run', ancestors: [], descendants: [] });
  if (path === '/runs/demo-run/samples') return json({ available: false, samples: [], reason: 'SFT sample preview is not available for this run.', cycle: null, kind: 'samples' });
  if (path === '/runs/demo-run/eval') return json({ available: false, reason: 'No eval summary yet.', tasks: [] });
  if (path === '/review-capabilities') return json(reviewCapabilities);
  if (path === '/review-queues') return json({ items: [reviewQueue, reviewQueueVlm], total: 2, limit: 50, offset: 0 });
  if (path === '/review-queues/summaries') return json({ items: [{ ...reviewQueue, statistics: reviewStats, next_item_id: reviewItems[0].id, next_item_ordinal: 0 }, { ...reviewQueueVlm, statistics: { ...reviewStats, queue_id: reviewQueueVlm.id, total: 18, resolved: 12, coverage: 2 / 3, conflicts: 0, status_counts: { resolved: 12, pending: 6 } } }], total: 2, limit: 50, offset: 0 });
  if (path === '/review-queues/queue-demo/statistics') return json(reviewStats);
  if (path === '/review-queues/queue-vlm/statistics') return json({ ...reviewStats, queue_id: reviewQueueVlm.id, total: 18, resolved: 12, coverage: 2 / 3, conflicts: 0, status_counts: { resolved: 12, pending: 6 } });
  if (path === '/review-queues/queue-demo/items') return json({ items: reviewItems, total: reviewItems.length, limit: 100, offset: 0 });
  if (path === '/review-queues/queue-demo') return json(reviewQueue);
  if (path === `/annotation-schema-revisions/${reviewSchema.id}`) return json(reviewSchema);
  if (path === '/annotation-schemas') return json({ items: [{ id: 'schema-preference', name: 'Preference quality', description: 'Pairwise response quality rubric', archived: false, created_at: reviewSchema.created_at, updated_at: reviewSchema.created_at }], total: 1, limit: 200, offset: 0 });
  if (path === '/annotation-schemas/schema-preference/revisions') return json({ items: [reviewSchema], total: 1, limit: 200, offset: 0 });
  if (path === '/spec-descriptors/dataset_recipe_step') return json({ items: [{ kind: 'dataset_recipe_step', id: 'map', version: '1', label: 'Map fields', description: 'Map source columns into a canonical training schema.', fields: [{ name: 'schema', label: 'Canonical schema', value_type: 'select', required: true, options: ['sft', 'chat', 'preference', 'tool', 'vlm', 'audio'] }, { name: 'fields', label: 'Field mapping', value_type: 'field_mapping', required: true, description: 'Canonical target to source-column mapping.' }] }], total: 1 });
  if (path.startsWith('/review-items/')) {
    const parts = path.split('/');
    const item = reviewItems.find(value => value.id === parts[2]) || reviewItems[0];
    if (parts[3] === 'events') return json({ items: item.active_event_id ? [{ id: item.active_event_id, queue_id: reviewQueue.id, item_id: item.id, event_type: 'label', pass_number: 1, idempotency_key: 'visual-event', request_hash: 'requesthash', expected_active_event_id: null, payload: { annotation: item.projection.pass_1?.annotation || { chosen: 'A' } }, supersedes_event_id: null, created_at: new Date().toISOString() }] : [], total: item.active_event_id ? 1 : 0, limit: 100, offset: 0 });
    if (parts[3] === 'neighbors') { const index = reviewItems.findIndex(value => value.id === item.id); return json({ item_id: item.id, position: Math.max(0, index), total: reviewItems.length, previous_id: reviewItems[index - 1]?.id || null, next_id: reviewItems[index + 1]?.id || null }); }
    if (parts[3] === 'suggestions') return json({ items: [{ id: 'suggestion-1', item_id: item.id, pass_number: 1, provider: 'openai_compatible', model_revision: 'local-teacher@r3', content_hash: 'suggestionhash', output: null, provenance: { prompt_hash: 'prompthash' }, created_at: new Date().toISOString() }], total: 1, limit: 100, offset: 0 });
    return json(item);
  }
  if (path === '/docs') return json({ items: [] });
  return json({ ok: true });
});

const pages = [
  ['overview', '/'],
  ['models', '/models'],
  ['models-artifacts', '/models?tab=artifacts'],
  ['models-cached', '/models?tab=cached'],
  ['models-serve', '/models?tab=serve'],
  ['runs', '/runs'],
  ['run-collections', '/runs?view=collections'],
  ['experiments', '/sweeps'],
  ['adaptation-studies', '/sweeps?section=studies'],
  ['run-detail', '/runs/demo-run'],
  ['run-metrics', '/runs/demo-run?tab=metrics'],
  ['run-data', '/runs/demo-run?tab=data'],
  ['run-artifacts', '/runs/demo-run?tab=artifacts'],
  ['run-training-audits', `/runs/demo-run?tab=evaluation&evidence=training-audits&audit=${rewardAudit.id}`],
  ['datasets', '/datasets'],
  ['own-data-goal', '/datasets/new'],
  ['dataset-preview', '/datasets/ds-demo?tab=preview'],
  ['dataset-build', '/datasets/ds-demo?tab=build'],
  ['dataset-version', '/datasets/ds-demo/versions/v-demo-001?split=train'],
  ['dataset-version-records', '/datasets/ds-demo/versions/v-demo-001?split=train&view=records'],
  ['dataset-version-training', '/datasets/ds-demo/versions/v-demo-001?split=train&view=training'],
  ['grounded-data', '/datasets/ground?sourceVersion=v-demo-001'],
  ['review-studio', '/datasets/review'],
  ['review-proposal', '/datasets/review?new=1&source=evaluation_comparison&sourceRef=eval-candidate-001&baseRef=eval-base-001'],
  ['review-verifier-proposal', '/datasets/review?new=1&source=verifier_calibration&sourceRef=verifier-calibration-1'],
  ['review-item', '/datasets/review/queue-demo'],
  ['train-fork', '/train?parentRun=demo-run'],
  ['train-qualified-verifier', '/train?mode=raft'],
  ['evaluation-lab', '/eval?runId=demo-run'],
  ['agent-environments', '/eval?section=environments'],
  ['verifier-profiles', '/eval?section=verifiers&verifierView=profiles&profile=verifier-profile-exact'],
  ['verifier-calibration', '/eval?section=verifiers&verifierView=calibrate&profile=verifier-profile-exact&calibration=verifier-calibration-1'],
  ['reward-audit-profiles', '/eval?section=verifiers&verifierView=training-audits&auditView=profiles'],
  ['reward-audit-results', `/eval?section=verifiers&verifierView=training-audits&auditView=results&audit=${rewardAudit.id}`],
  ['docs', '/docs'],
  ['connect', '/connect'],
];

for (const [name, path] of pages) {
  await page.goto(`${base}${path}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true });
}

await page.goto(`${base}/datasets/new?example=1`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(900);
await page.getByRole('button', { name: /^Continue/ }).click();
await page.waitForTimeout(350);
await page.getByRole('button', { name: /Inspect source/ }).click();
await page.waitForTimeout(600);
await page.getByRole('button', { name: /Confirm Instruction and response/i }).click();
await page.getByRole('button', { name: /^Continue/ }).click();
await page.waitForTimeout(600);
await page.evaluate(() => {
  document.documentElement.style.setProperty('scroll-behavior', 'auto', 'important');
  const main = document.getElementById('main');
  const scroller = [...(main?.children ?? [])].find(element => ['auto', 'scroll'].includes(getComputedStyle(element).overflowY));
  if (scroller) {
    scroller.style.scrollBehavior = 'auto';
    scroller.scrollTo({ top: 0, behavior: 'auto' });
  }
  window.scrollTo(0, 0);
});
await page.waitForFunction(() => {
  const main = document.getElementById('main');
  const scroller = [...(main?.children ?? [])].find(element => ['auto', 'scroll'].includes(getComputedStyle(element).overflowY));
  return window.scrollY === 0 && (!scroller || scroller.scrollTop === 0);
});
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/own-data-mapping.png`, fullPage: true });

await page.goto(`${base}/datasets/new`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(700);
await page.getByRole('button', { name: /^Instruction and response/i }).first().click();
await page.getByRole('button', { name: /^Continue/ }).click();
await page.getByRole('button', { name: /Hugging Face/i }).click();
await page.getByLabel('Dataset repository').fill('acme/support-preferences');
await page.getByLabel('Pinned revision').fill('main');
await page.getByRole('button', { name: /Browse configs and splits/i }).click();
await page.waitForTimeout(500);
await page.evaluate(() => {
  document.documentElement.style.setProperty('scroll-behavior', 'auto', 'important');
  const main = document.getElementById('main');
  const scroller = [...(main?.children ?? [])].find(element => ['auto', 'scroll'].includes(getComputedStyle(element).overflowY));
  if (scroller) {
    scroller.style.scrollBehavior = 'auto';
    scroller.scrollTo({ top: 0, behavior: 'auto' });
  }
  window.scrollTo(0, 0);
});
await page.waitForFunction(() => {
  const main = document.getElementById('main');
  const scroller = [...(main?.children ?? [])].find(element => ['auto', 'scroll'].includes(getComputedStyle(element).overflowY));
  return window.scrollY === 0 && (!scroller || scroller.scrollTop === 0);
});
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/own-data-huggingface.png`, fullPage: true });

await page.goto(`${base}/datasets/new`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(700);
await page.getByRole('button', { name: /^Adapt a model to documents/i }).click();
await page.getByRole('button', { name: /^Continue/ }).click();
await page.waitForTimeout(500);
await page.evaluate(() => {
  document.documentElement.style.setProperty('scroll-behavior', 'auto', 'important');
  const main = document.getElementById('main');
  const scroller = [...(main?.children ?? [])].find(element => ['auto', 'scroll'].includes(getComputedStyle(element).overflowY));
  if (scroller) {
    scroller.style.scrollBehavior = 'auto';
    scroller.scrollTo({ top: 0, behavior: 'auto' });
  }
  window.scrollTo(0, 0);
});
await page.screenshot({ path: `${outDir}/own-data-corpus-source.png`, fullPage: true });

await page.goto(`${base}/datasets/new?inspection=${ownDataAmbiguousInspection.id}`, { waitUntil: 'domcontentloaded' });
await page.getByText('No safe format match').waitFor();
await page.getByRole('combobox', { name: 'Choose a scenario manually' }).click();
await page.getByRole('option', { name: ownDataScenario.label }).click();
await page.waitForTimeout(300);
await page.screenshot({ path: `${outDir}/own-data-manual-format.png`, fullPage: true });

await page.goto(`${base}/train?mode=raft`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1200);
await page.getByText(/qualified verifier profile/i).first().scrollIntoViewIfNeeded();
await page.waitForTimeout(150);
await page.getByRole('combobox', { name: /choose a compatible qualified verifier/i }).focus();
await page.waitForTimeout(100);
await page.screenshot({ path: `${outDir}/train-qualified-verifier.png`, fullPage: true });
await page.getByRole('option', { name: /exact answer oracle/i }).click();
await page.waitForTimeout(600);
await page.screenshot({ path: `${outDir}/train-verifier-bound.png`, fullPage: true });
await page.getByRole('button', { name: /add audit/i }).click();
await page.waitForTimeout(500);
await page.screenshot({ path: `${outDir}/train-reward-audit.png`, fullPage: true });

await page.goto(`${base}/eval?section=verifiers&verifierView=training-audits&auditView=profiles`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: /create reward system/i }).click();
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/reward-system-creator.png`, fullPage: true });

await page.goto(`${base}/sweeps`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: /new group/i }).click();
await page.getByRole('button', { name: /boundaries & audits/i }).click();
await page.getByRole('button', { name: /guarded training/i }).click();
await page.getByRole('button', { name: /create a named policy/i }).click();
await page.waitForTimeout(250);
await page.evaluate(() => window.scrollTo(0, 0));
await page.screenshot({ path: `${outDir}/experiments-checkpoint-policy.png`, fullPage: true });

await page.goto(`${base}/sweeps`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: /^Evidence/ }).click();
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/experiments-evidence.png`, fullPage: true });

await page.goto(`${base}/models?tab=artifacts`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: /activity/i }).first().click();
await page.waitForTimeout(300);
await page.screenshot({ path: `${outDir}/activity-center.png`, fullPage: true });

await page.keyboard.press('Escape');
await page.keyboard.press(process.platform === 'darwin' ? 'Meta+K' : 'Control+K');
await page.waitForTimeout(250);
await page.getByPlaceholder(/search data, runs, suites/i).fill('SFT');
await page.waitForTimeout(350);
await page.screenshot({ path: `${outDir}/command-palette.png`, fullPage: true });

await page.keyboard.press('Escape');
await page.goto(`${base}/models?tab=artifacts`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: /^Qualify/ }).first().click();
await page.getByRole('button', { name: /create structured profile/i }).click();
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/qualification-profile-creator.png`, fullPage: true });

await page.goto(`${base}/sweeps`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: /new group/i }).first().click();
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/experiment-composer.png`, fullPage: true });

await page.setViewportSize({ width: 390, height: 844 });
await page.goto(`${base}/models?tab=artifacts`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(800);
await page.getByRole('button', { name: 'Open navigation' }).click();
await page.waitForTimeout(250);
await page.screenshot({ path: `${outDir}/mobile-navigation.png`, fullPage: true });

await page.keyboard.press('Escape');
for (const [name, path] of [
  ['mobile-own-data', '/datasets/new?example=1'],
  ['mobile-run-metrics', '/runs/demo-run?tab=metrics'],
  ['mobile-dataset-training', '/datasets/ds-demo/versions/v-demo-001?split=train&view=training'],
  ['mobile-models-serve', '/models?tab=serve'],
  ['mobile-experiments', '/sweeps'],
  ['mobile-review-item', '/datasets/review/queue-demo'],
  ['mobile-reward-audit', `/eval?section=verifiers&verifierView=training-audits&auditView=results&audit=${rewardAudit.id}&auditSample=reward-observation-0`],
]) {
  await page.goto(`${base}${path}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true });
}

console.log(JSON.stringify({ outDir, consoleErrors }, null, 2));
await browser.close();
if (consoleErrors.length) process.exitCode = 1;
