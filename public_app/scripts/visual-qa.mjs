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

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
page.on('console', msg => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
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
  if (path === '/backend') return json(backend);
  if (path === '/telemetry') return json(telemetry);
  if (path === '/train/datasets') return json({ items: [
    { key: 'codealpaca', huggingface_id: 'sahil2801/CodeAlpaca-20k', description: '20K instruction-following code examples', domain: 'code', size_hint: '20K', default_split: 'train' },
    { key: 'gsm8k_sft', huggingface_id: 'gsm8k', description: '8.5K grade school math for SFT', domain: 'reasoning', size_hint: '8.5K', default_split: 'train' },
    { key: 'xlam_sft', huggingface_id: 'Salesforce/xlam-function-calling-60k', description: '60K function calling examples', domain: 'agentic', size_hint: '60K', default_split: 'train' },
  ] });
  if (path === '/train/models') return json({ items: models });
  if (path === '/models') return json({ catalog_version: '2026.05', items: models, total: models.length, facets: { providers: ['Liquid AI', 'Qwen', 'mlx-community'], statuses: ['recommended', 'experimental'], modalities: ['text', 'code'], memory_tiers: ['tiny', 'small'], risk_levels: ['safe', 'experimental'] }, filters: {} });
  if (path === '/train/preflight') return json({ mode: 'sft', ok: true, resolved_paths: {}, errors: [], warnings: [], suggested_fixes: [], user_summary: { headline: 'Ready to launch', next_step: 'Start the run when you are ready.', confidence_tone: 'success' } });
  if (path === '/dashboard') return json({ readiness_tier: 'qualified' });
  if (path === '/runs/search') return json({ items: [{ run_id: 'demo-run', modality: 'sft', model_name: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', status: 'running', created_at: new Date().toISOString(), cycles_executed: 1, weights_updated: true, final_train_loss: 1.234, effectiveness: { verdict: 'review' } }], total: 1, filters: {}, facets: { modalities: ['sft'], modality_counts: { sft: 1 }, models: ['mlx-community/Qwen2.5-0.5B-Instruct-bf16'] } });
  if (path === '/runs') return json({ items: [{ run_id: 'demo-run', modality: 'sft', model_name: 'mlx-community/Qwen2.5-0.5B-Instruct-bf16', status: 'running', created_at: new Date().toISOString(), cycles_executed: 1, weights_updated: true, final_train_loss: 1.234, effectiveness: { verdict: 'review' } }] });
  if (path === '/runs/demo-run') return json(runDetail);
  if (path === '/runs/demo-run/lineage') return json({ run_id: 'demo-run', ancestors: [], descendants: [] });
  if (path === '/runs/demo-run/samples') return json({ available: false, samples: [], reason: 'SFT sample preview is not available for this run.', cycle: null, kind: 'samples' });
  if (path === '/runs/demo-run/eval') return json({ available: false, reason: 'No eval summary yet.', tasks: [] });
  if (path === '/docs') return json({ items: [] });
  return json({ ok: true });
});

const pages = [
  ['start', '/start'],
  ['models', '/models'],
  ['runs', '/runs'],
  ['run-detail', '/runs/demo-run'],
  ['docs', '/docs'],
  ['connect', '/connect'],
];

for (const [name, path] of pages) {
  await page.goto(`${base}${path}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: true });
}

console.log(JSON.stringify({ outDir, consoleErrors }, null, 2));
await browser.close();
