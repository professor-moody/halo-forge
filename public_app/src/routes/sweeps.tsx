import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Ban,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  Copy,
  FileClock,
  FlaskConical,
  Gauge,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Trophy,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  api,
  type BenchmarkSuite,
  type DatasetVersion,
  type ExperimentTrial,
  type CheckpointPolicyRevision,
  type ModelCatalogEntry,
  type RunGroup,
  type RunGroupCreatePayload,
  type TrainingMode,
  type TrainerExecutionCapability,
  type WorkItem,
  type AdaptationStudy,
} from "@/lib/api";
import { useModelCatalog } from "@/lib/hooks";
import { useWorkspaceDraft } from "@/lib/workspace-draft";
import { Topbar } from "@/components/shell";
import { AdaptiveExperimentWorkspace } from "@/components/research/adaptive-workspace";
import {
  EMPTY_REWARD_AUDIT_BINDING,
  RewardAuditBindingEditor,
  type RewardAuditBindingValue,
} from "@/components/research/reward-audit-binding";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SearchPicker } from "@/components/ui/search-picker";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/sweeps")({
  component: SweepsRoute,
  validateSearch: (search): { section?: "groups" | "studies"; group?: string; new?: "1"; policy?: string; datasetVersion?: string; trainerMode?: TrainingMode; kind?: "repeat" | "sweep" } => ({
    section: search.section === "studies" ? "studies" : search.section === "groups" ? "groups" : undefined,
    group: typeof search.group === "string" ? search.group : undefined,
    new: search.new === "1" ? "1" : undefined,
    policy: typeof search.policy === "string" ? search.policy : undefined,
    datasetVersion: typeof search.datasetVersion === "string" ? search.datasetVersion : undefined,
    trainerMode: typeof search.trainerMode === "string" && TRAINERS.includes(search.trainerMode as TrainingMode) ? search.trainerMode as TrainingMode : undefined,
    kind: search.kind === "sweep" ? "sweep" : search.kind === "repeat" ? "repeat" : undefined,
  }),
});

const TRAINERS: TrainingMode[] = [
  "sft",
  "dpo",
  "orpo",
  "rm",
  "grpo",
  "raft",
  "vlm",
  "audio",
  "reasoning",
  "agentic",
  "classify",
  "embed",
  "rerank",
];

type ComposerState = {
  name: string;
  kind: "repeat" | "sweep";
  trainerMode: TrainingMode;
  suiteRevisionId: string;
  model: string;
  datasetVersionId: string;
  epochs: number;
  adaptiveBudget: number;
  seeds: string;
  nTrials: number;
  learningRateLow: string;
  learningRateHigh: string;
  batchSizes: string;
  loraRanks: string;
  pruning: boolean;
  checkpointMode: "final_only" | "periodic" | "guarded";
  checkpointPolicyRevisionId: string;
  rewardAudit: RewardAuditBindingValue;
};

const INITIAL_COMPOSER: ComposerState = {
  name: "",
  kind: "repeat",
  trainerMode: "sft",
  suiteRevisionId: "",
  model: "",
  datasetVersionId: "",
  epochs: 1,
  adaptiveBudget: 1000,
  seeds: "42, 43, 44",
  nTrials: 9,
  learningRateLow: "1e-5",
  learningRateHigh: "3e-4",
  batchSizes: "1, 2",
  loraRanks: "8, 16, 32",
  pruning: false,
  checkpointMode: "final_only",
  checkpointPolicyRevisionId: "",
  rewardAudit: EMPTY_REWARD_AUDIT_BINDING,
};

function SweepsRoute() {
  const search = Route.useSearch();
  const activeSection = search.section ?? "groups";
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState("");
  const [composerOpen, setComposerOpen] = useState(false);
  const [composer, setComposer] = useState<ComposerState>(INITIAL_COMPOSER);
  const draft = useWorkspaceDraft({
    surface: "experiment",
    draftKey: "new-run-group",
    name: composer.name || "New experiment",
    value: composer,
    enabled: true,
    onRestore: (value) => { setComposer(value); setComposerOpen(true); },
  });

  const groups = useQuery({
    queryKey: ["run-groups"],
    queryFn: () => api.listRunGroups({ limit: 200 }),
    refetchInterval: 5_000,
    retry: false,
  });
  const queue = useQuery({
    queryKey: ["work-items"],
    queryFn: () => api.listWorkItems({ limit: 200 }),
    refetchInterval: 2_000,
    retry: false,
  });
  const suites = useQuery({
    queryKey: ["benchmark-suites"],
    queryFn: api.listBenchmarkSuites,
    retry: false,
  });
  const catalog = useModelCatalog();
  const policies = useQuery({
    queryKey: ["checkpoint-policies", composer.trainerMode],
    queryFn: () => api.listCheckpointPolicies({ trainerMode: composer.trainerMode, limit: 100 }),
    retry: false,
  });
  const requestedPolicy = useQuery({
    queryKey: ["checkpoint-policies", "requested", search.policy],
    queryFn: () => api.checkpointPolicy(search.policy!),
    enabled: Boolean(search.policy),
    retry: false,
  });
  const executionCapabilities = useQuery({
    queryKey: ["trainer-execution-capabilities"],
    queryFn: api.listTrainerExecutionCapabilities,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const datasetVersions = useQuery({
    queryKey: ["training", "dataset-versions", composer.trainerMode],
    queryFn: () => api.trainingDatasetVersions(composer.trainerMode),
    retry: false,
  });
  const detail = useQuery({
    queryKey: ["run-groups", selectedId],
    queryFn: () => api.runGroup(selectedId),
    enabled: Boolean(selectedId),
    refetchInterval: (query) =>
      query.state.data && terminalStatus(query.state.data.status) ? false : 3_000,
    retry: false,
  });

  useEffect(() => {
    if (selectedId || !groups.data?.items.length) return;
    setSelectedId(groups.data.items[0].id);
  }, [groups.data?.items, selectedId]);

  useEffect(() => {
    if (search.group) setSelectedId(search.group);
    if (search.new === "1" || search.policy || search.datasetVersion) setComposerOpen(true);
    if (search.datasetVersion || search.trainerMode || search.kind) {
      setComposer((value) => ({
        ...value,
        datasetVersionId: search.datasetVersion || value.datasetVersionId,
        trainerMode: search.trainerMode || value.trainerMode,
        kind: search.kind || value.kind,
        name: value.name || (search.kind === "sweep" ? "Dataset search" : "Dataset repeat"),
      }));
    }
  }, [search.datasetVersion, search.group, search.kind, search.new, search.policy, search.trainerMode]);

  useEffect(() => {
    const policy = requestedPolicy.data;
    if (!policy || composer.checkpointPolicyRevisionId === policyIdentity(policy)) return;
    setComposer((value) => ({ ...value, checkpointMode: policy.rules.length ? "guarded" : "periodic", checkpointPolicyRevisionId: policyIdentity(policy), suiteRevisionId: policy.development_suite_revision_id || value.suiteRevisionId }));
  }, [composer.checkpointPolicyRevisionId, requestedPolicy.data]);

  useEffect(() => {
    if (composer.suiteRevisionId || !suites.data?.items.length) return;
    const latest = latestRevisionId(suites.data.items[0]);
    if (latest) setComposer((value) => ({ ...value, suiteRevisionId: latest }));
  }, [composer.suiteRevisionId, suites.data?.items]);

  const create = useMutation({
    mutationFn: async () => {
      const payload = buildGroupPayload(composer);
      if (composer.checkpointPolicyRevisionId) {
        const unit = checkpointUnit(composer.trainerMode);
        payload.resolved_checkpoint_plan = await api.resolveCheckpointPolicy({
          policy_revision_id: composer.checkpointPolicyRevisionId,
          trainer_mode: composer.trainerMode,
          total_budget: composer.adaptiveBudget,
          budget_unit: unit,
          base_config: payload.base_config,
        });
      }
      return api.createRunGroup(payload);
    },
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["run-groups"] });
      queryClient.invalidateQueries({ queryKey: ["work-items"] });
      setSelectedId(created.id);
      setComposerOpen(false);
      setComposer((value) => ({ ...INITIAL_COMPOSER, suiteRevisionId: value.suiteRevisionId }));
      void draft.clear();
    },
  });
  const cancel = useMutation({
    mutationFn: (id: string) => api.cancelRunGroup(id),
    onSuccess: refresh,
  });
  const resume = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) => api.resumeRunGroup(id, reason),
    onSuccess: refresh,
  });
  const retryWork = useMutation({
    mutationFn: (id: string) => api.retryWorkItem(id),
    onSuccess: refresh,
  });
  const forkBest = useMutation({
    mutationFn: (id: string) => api.forkBestRunGroup(id),
    onSuccess: (created) => {
      refresh();
      setSelectedId(created.id);
    },
  });

  function refresh() {
    queryClient.invalidateQueries({ queryKey: ["run-groups"] });
    queryClient.invalidateQueries({ queryKey: ["work-items"] });
    if (selectedId) queryClient.invalidateQueries({ queryKey: ["run-groups", selectedId] });
  }

  const selected = detail.data ?? groups.data?.items.find((group) => group.id === selectedId) ?? null;
  const activeItems = (queue.data?.items ?? []).filter((item) => ["queued", "running"].includes(item.status));
  const running = activeItems.filter((item) => item.status === "running").length;
  const queued = activeItems.filter((item) => item.status === "queued").length;

  return (
    <>
      <Topbar
        eyebrow="Research operations"
        title="Experiments"
        subtitle={activeSection === "studies" ? "Ask a controlled adaptation question, pin the evidence, and compare matched arms." : "Repeat, compare, and sweep one pinned training and evaluation contract at a time."}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={refresh} disabled={groups.isFetching || queue.isFetching}>
              <RefreshCw className={cn((groups.isFetching || queue.isFetching) && "animate-spin")} />
              Refresh
            </Button>
            {activeSection === "groups" ? <Button size="sm" onClick={() => setComposerOpen((value) => !value)}>
              <Plus /> New group
            </Button> : null}
          </>
        }
        statusBar={
          <>
            <Readout label="GROUPS" value={String(groups.data?.items.length ?? 0)} />
            <span className="text-fg-disabled">·</span>
            <Readout label="RUNNING" value={String(running)} />
            <span className="text-fg-disabled">·</span>
            <Readout label="QUEUED" value={String(queued)} />
            <span className="text-fg-disabled">·</span>
            <Readout label="HEAVY SLOTS" value={running ? "1 / 1" : "0 / 1"} />
          </>
        }
      />

      <nav className="flex h-10 items-end gap-1 border-b border-border bg-bg px-5" aria-label="Experiment sections">
        <Link to="/sweeps" search={{ ...search, section: "groups" }} className={cn("relative h-10 px-3 text-[10.5px] leading-10", activeSection === "groups" ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>Repeats & searches{activeSection === "groups" ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}</Link>
        <Link to="/sweeps" search={{ ...search, section: "studies", group: undefined, new: undefined }} className={cn("relative h-10 px-3 text-[10.5px] leading-10", activeSection === "studies" ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>Studies{activeSection === "studies" ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}</Link>
      </nav>

      {activeSection === "groups" && composerOpen ? (
        <GroupComposer
          value={composer}
          suites={suites.data?.items ?? []}
          models={catalog.data?.items ?? []}
          datasetVersions={datasetVersions.data?.items ?? []}
          policies={policies.data?.items ?? []}
          executionCapabilities={executionCapabilities.data?.items ?? []}
          draftCandidate={draft.candidate}
          draftSaving={draft.isSaving}
          draftSavedAt={draft.savedAt}
          onRestoreDraft={draft.restore}
          onDiscardDraft={draft.discard}
          onChange={setComposer}
          onCancel={() => setComposerOpen(false)}
          onCreate={() => create.mutate()}
          pending={create.isPending}
          error={create.error instanceof Error ? create.error.message : null}
        />
      ) : null}

      {activeSection === "studies" ? <StudiesWorkspace /> : <div className="grid min-h-[calc(100vh-152px)] xl:grid-cols-[260px_minmax(0,1fr)_280px]">
        <GroupRail
          groups={groups.data?.items ?? []}
          selectedId={selectedId}
          loading={groups.isLoading}
          onSelect={setSelectedId}
        />
        <main className="min-w-0 border-b border-border-subtle xl:border-b-0 xl:border-r">
          {selected ? (
            <GroupWorkspace
              group={selected}
              onCancel={() => cancel.mutate(selected.id)}
              onResume={(reason) => resume.mutate({ id: selected.id, reason })}
              onForkBest={() => forkBest.mutate(selected.id)}
              actionPending={cancel.isPending || resume.isPending || forkBest.isPending}
            />
          ) : (
            <EmptyWorkspace loading={groups.isLoading} onCreate={() => setComposerOpen(true)} />
          )}
        </main>
        <QueueRail
          items={queue.data?.items ?? []}
          activeLease={queue.data?.active_lease ?? null}
          loading={queue.isLoading}
          onRetry={(id) => retryWork.mutate(id)}
        />
      </div>}
    </>
  );
}

function GroupComposer({
  value,
  suites,
  models,
  datasetVersions,
  policies,
  executionCapabilities,
  draftCandidate,
  draftSaving,
  draftSavedAt,
  onRestoreDraft,
  onDiscardDraft,
  onChange,
  onCancel,
  onCreate,
  pending,
  error,
}: {
  value: ComposerState;
  suites: BenchmarkSuite[];
  models: ModelCatalogEntry[];
  datasetVersions: DatasetVersion[];
  policies: CheckpointPolicyRevision[];
  executionCapabilities: TrainerExecutionCapability[];
  draftCandidate: import("@/lib/api").WorkspaceDraft<ComposerState> | null;
  draftSaving: boolean;
  draftSavedAt: string | null;
  onRestoreDraft: () => void;
  onDiscardDraft: () => void;
  onChange: (value: ComposerState) => void;
  onCancel: () => void;
  onCreate: () => void;
  pending: boolean;
  error: string | null;
}) {
  const [step, setStep] = useState(0);
  const steps = ["Configuration", "Search / repeats", "Boundaries & audits", "Objective / budget", "Review"];
  const seeds = parseSeeds(value.seeds);
  const compatibleModels = models.filter((model) => (model.trainer_support ?? []).includes(value.trainerMode));
  const modelOptions = compatibleModels.length ? compatibleModels : models;
  const compatibleVersions = datasetVersions.filter((version) => datasetFitsTrainer(version, value.trainerMode));
  const selectedModel = models.find((model) => model.id === value.model);
  const backendFamily = selectedModel?.backend_support?.length === 1 && selectedModel.backend_support[0] === "mlx" || /(^|\/)mlx-|mlx-community/i.test(value.model) ? "mlx" : "hf";
  const executionCapability = executionCapabilities.find((item) => item.trainer_mode === value.trainerMode && item.backend_family === backendFamily) ?? executionCapabilities.find((item) => item.trainer_mode === value.trainerMode && item.backend_family === "*");
  const adaptiveAvailable = executionCapability ? executionCapability.supports_gated_execution : true;
  const adaptiveUnit = executionCapability?.segment_unit === "cycle" ? "cycle" : "step";
  const trialCount = value.kind === "repeat" ? seeds.length : Math.max(1, value.nTrials) * seeds.length;
  const auditReady = !value.rewardAudit.enabled || rewardAuditBindingReady(value.rewardAudit);
  const ready = Boolean(value.name.trim() && value.model.trim() && value.suiteRevisionId && parseSeeds(value.seeds).length && auditReady && (value.checkpointMode === "final_only" ? value.epochs > 0 : adaptiveAvailable && value.checkpointPolicyRevisionId && value.adaptiveBudget > 0));
  const stepReady = step === 0 ? Boolean(value.name.trim() && value.model.trim()) : step === 1 ? Boolean(seeds.length) : step === 2 ? auditReady && (value.checkpointMode === "final_only" || adaptiveAvailable && Boolean(value.checkpointPolicyRevisionId)) : step === 3 ? Boolean(value.suiteRevisionId && (value.checkpointMode === "final_only" ? value.epochs > 0 : value.adaptiveBudget > 0)) : ready;
  const selectedPolicy = policies.find((policy) => policyIdentity(policy) === value.checkpointPolicyRevisionId);
  return (
    <section className="border-b border-border bg-bg-subtle/35 px-5 py-4">
      <div className="mx-auto max-w-[1180px]">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-accent">New run group</div>
            <h2 className="mt-1 text-[18px] font-medium text-fg">Pin the training and evaluation contract</h2>
            <p className="mt-1 text-[12px] text-fg-muted">A repeat changes only the seed. A sweep changes only the selected search fields.</p>
          </div>
          <div className="flex items-center gap-2"><span className="font-mono text-[9.5px] text-fg-disabled">{draftSaving ? "Saving draft…" : draftSavedAt ? "Draft saved" : "Autosave on"}</span><Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button></div>
        </div>

        {draftCandidate ? <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-l-2 border-accent bg-accent-bg/30 px-3 py-2"><div><div className="text-[11px] font-medium text-fg">Resume the saved experiment draft?</div><div className="mt-0.5 text-[9.5px] text-fg-subtle">Saved {draftCandidate.updated_at ? new Date(draftCandidate.updated_at).toLocaleString() : "recently"} · expires {draftCandidate.expires_at ? new Date(draftCandidate.expires_at).toLocaleDateString() : "after 30 inactive days"}</div></div><div className="flex gap-2"><Button size="sm" onClick={onRestoreDraft}><FileClock /> Restore</Button><Button size="sm" variant="ghost" onClick={onDiscardDraft}>Discard</Button></div></div> : null}

        <div className="mb-5 flex overflow-x-auto border-b border-border-subtle">
          {steps.map((label, index) => (
            <button key={label} type="button" onClick={() => setStep(index)} className={cn("relative flex h-9 shrink-0 items-center gap-2 px-3 text-[10.5px] transition-colors", index === step ? "text-fg" : index < step ? "text-fg-muted" : "text-fg-disabled")}>
              <span className={cn("grid h-4 w-4 place-items-center rounded-full border font-mono text-[8.5px]", index === step ? "border-accent bg-accent-bg text-accent" : index < step ? "border-success/40 text-success" : "border-border")}>{index < step ? "✓" : index + 1}</span>
              {label}
              {index === step ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}
            </button>
          ))}
        </div>

        {step === 0 ? (
          <div className="grid gap-x-5 gap-y-4 md:grid-cols-2">
            <Field label="Experiment name"><Input value={value.name} onChange={(event) => onChange({ ...value, name: event.target.value })} placeholder="SFT stability check" /></Field>
            <Field label="Operation"><NativeSelect value={value.kind} onChange={(kind) => onChange({ ...value, kind: kind as ComposerState["kind"] })}><option value="repeat">Repeat across seeds</option><option value="sweep">Search training settings</option></NativeSelect></Field>
            <Field label="Trainer"><NativeSelect value={value.trainerMode} onChange={(trainerMode) => onChange({ ...value, trainerMode: trainerMode as TrainingMode, model: "", datasetVersionId: "", checkpointPolicyRevisionId: "", rewardAudit: EMPTY_REWARD_AUDIT_BINDING, adaptiveBudget: checkpointUnit(trainerMode as TrainingMode) === "cycle" ? 3 : 1000 })}>{TRAINERS.map((trainer) => <option key={trainer} value={trainer}>{trainer.toUpperCase()}</option>)}</NativeSelect></Field>
            <Field label="Compatible base model"><SearchPicker value={value.model} onChange={(model) => onChange({ ...value, model })} placeholder="Search compatible models" options={modelOptions.map((model) => ({ value: model.id, label: model.label || model.id, description: `${model.memory_tier || "memory unknown"} · ${model.id}`, keywords: `${model.id} ${model.trainer_support?.join(" ") ?? ""}` }))} />{!compatibleModels.length && models.length ? <p className="mt-1 text-[10px] text-warning">No catalog model explicitly advertises {value.trainerMode.toUpperCase()}; showing the full catalog for review.</p> : null}</Field>
            <Field label="Compatible dataset version (optional)"><SearchPicker value={value.datasetVersionId} onChange={(datasetVersionId) => onChange({ ...value, datasetVersionId })} placeholder="Search ready dataset versions" allowEmpty options={compatibleVersions.map((version) => ({ value: version.id, label: version.label || `Version ${version.version ?? version.id}`, description: `${version.row_count ?? "?"} rows · ${version.canonical_schema || "schema unknown"}`, status: version.status }))} />{datasetVersions.length > 0 && !compatibleVersions.length ? <p className="mt-1 text-[10px] text-warning">No ready dataset version reports compatibility with this trainer.</p> : null}</Field>
            <details className="md:col-span-2"><summary className="cursor-pointer text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled hover:text-fg">Advanced · use an unlisted identifier</summary><div className="mt-3 grid gap-3 md:grid-cols-2"><Field label="Model repository or local path"><Input value={value.model} onChange={(event) => onChange({ ...value, model: event.target.value })} placeholder="Qwen/Qwen2.5-3B-Instruct" mono /></Field><Field label="Dataset version ID"><Input value={value.datasetVersionId} onChange={(event) => onChange({ ...value, datasetVersionId: event.target.value })} placeholder="version id" mono /></Field></div></details>
          </div>
        ) : null}

        {step === 1 ? (
          <div className="grid gap-x-5 gap-y-4 md:grid-cols-2 xl:grid-cols-4">
            <Field label="Deterministic seeds"><Input value={value.seeds} onChange={(event) => onChange({ ...value, seeds: event.target.value })} placeholder="42, 43, 44" mono /><p className="mt-1 text-[10px] text-fg-disabled">{seeds.length} valid unique seed{seeds.length === 1 ? "" : "s"}</p></Field>
            {value.kind === "sweep" ? <><Field label="Candidate settings"><Input type="number" min={1} value={value.nTrials} onChange={(event) => onChange({ ...value, nTrials: Math.max(1, Number(event.target.value)) })} /></Field><Field label="Learning rate range"><div className="grid grid-cols-2 gap-2"><Input value={value.learningRateLow} onChange={(event) => onChange({ ...value, learningRateLow: event.target.value })} aria-label="Minimum learning rate" mono /><Input value={value.learningRateHigh} onChange={(event) => onChange({ ...value, learningRateHigh: event.target.value })} aria-label="Maximum learning rate" mono /></div></Field><Field label="Batch sizes"><Input value={value.batchSizes} onChange={(event) => onChange({ ...value, batchSizes: event.target.value })} placeholder="1, 2, 4" mono /></Field><Field label="LoRA ranks"><Input value={value.loraRanks} onChange={(event) => onChange({ ...value, loraRanks: event.target.value })} placeholder="8, 16, 32" mono /></Field></> : <div className="rounded-md border border-border-subtle bg-surface/45 px-4 py-3 md:col-span-1"><div className="text-[10px] uppercase tracking-wider text-fg-disabled">Repeat contract</div><p className="mt-2 text-[11.5px] leading-relaxed text-fg-muted">Only the seed changes. Model, data, training budget, and evaluation revision remain pinned.</p></div>}
          </div>
        ) : null}

        {step === 2 ? (
          <div className="space-y-4">
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div>
              <div className="grid gap-px border border-border-subtle bg-border-subtle sm:grid-cols-3">
                <CheckpointModeButton active={value.checkpointMode === "final_only"} icon={CheckCircle2} label="Final only" description="Train once, then evaluate the final artifact." onClick={() => onChange({ ...value, checkpointMode: "final_only", checkpointPolicyRevisionId: "" })} />
                <CheckpointModeButton active={value.checkpointMode === "periodic"} disabled={!adaptiveAvailable} icon={Gauge} label="Periodic observation" description={adaptiveAvailable ? "Measure verified checkpoints and continue automatically when evidence is complete." : "This trainer/backend can evaluate only after the full trial."} onClick={() => { const match = policies.find((policy) => policy.rules.length === 0 && policy.automatic_actions && (!policy.schedule.unit || policy.schedule.unit === adaptiveUnit)); onChange({ ...value, checkpointMode: "periodic", checkpointPolicyRevisionId: match ? policyIdentity(match) : "" }); }} />
                <CheckpointModeButton active={value.checkpointMode === "guarded"} disabled={!adaptiveAvailable} icon={ShieldCheck} label="Guarded training" description={adaptiveAvailable ? "Pause or stop only at predeclared verified boundaries." : "Bounded resume is unavailable for this trainer/backend."} onClick={() => { const match = policies.find((policy) => policy.rules.length > 0 && (!policy.schedule.unit || policy.schedule.unit === adaptiveUnit)); onChange({ ...value, checkpointMode: "guarded", checkpointPolicyRevisionId: match ? policyIdentity(match) : "" }); }} />
              </div>
              {value.checkpointMode !== "final_only" ? <div className="mt-4"><Field label="Compatible checkpoint policy"><SearchPicker disabled={!adaptiveAvailable} value={value.checkpointPolicyRevisionId} onChange={(checkpointPolicyRevisionId) => onChange({ ...value, checkpointPolicyRevisionId })} placeholder="Search immutable policies" options={policies.filter((policy) => (!policy.schedule.unit || policy.schedule.unit === adaptiveUnit) && (value.checkpointMode === "periodic" ? policy.rules.length === 0 && policy.automatic_actions : policy.rules.length > 0)).map((policy) => ({ value: policyIdentity(policy), label: policy.name, description: `revision ${policy.revision_number} · ${policy.schedule.mode || policy.schedule.kind || policy.schedule.unit || "schedule"}`, status: policy.rules.length ? (policy.automatic_actions ? "guarded" : "manual review") : "periodic", keywords: `${policy.primary_metric} ${policy.compatible_capabilities?.join(" ") ?? ""}` }))} emptyLabel={`No ${adaptiveUnit}-compatible policy revisions`} /></Field></div> : null}
              {value.checkpointMode !== "final_only" && !policies.length ? <div className="mt-3 border-l-2 border-warning pl-3 text-[10.5px] text-warning">No checkpoint policy advertises compatibility with this trainer. Final-only remains available.</div> : null}
              {value.checkpointMode !== "final_only" && adaptiveAvailable ? <PolicyCreator mode={value.checkpointMode} trainerMode={value.trainerMode} unit={adaptiveUnit} suiteRevisionId={value.suiteRevisionId} defaultMetric={suites.find((suite) => latestRevisionId(suite) === value.suiteRevisionId)?.latest_revision?.primary_metric ?? "score"} defaultDirection={suites.find((suite) => latestRevisionId(suite) === value.suiteRevisionId)?.latest_revision?.direction ?? "maximize"} onCreated={(checkpointPolicyRevisionId) => onChange({ ...value, checkpointPolicyRevisionId })} /> : null}
            </div>
            <aside className="border-l border-border-subtle pl-4">
              <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">Resolved intent</div>
              {selectedPolicy ? <div className="mt-3"><div className="text-[13px] font-medium text-fg">{selectedPolicy.name}</div><p className="mt-1 text-[10.5px] leading-relaxed text-fg-subtle">{selectedPolicy.description || "Immutable checkpoint schedule and evidence gates."}</p><dl className="mt-3 divide-y divide-border-subtle"><PolicyValue label="Primary metric" value={`${selectedPolicy.primary_metric} · ${selectedPolicy.direction}`} /><PolicyValue label="Schedule" value={formatSchedule(selectedPolicy)} /><PolicyValue label="Rules" value={String(selectedPolicy.rules.length)} /><PolicyValue label="Actions" value={policyActionLabel(selectedPolicy)} /><PolicyValue label="Retention" value={formatRetention(selectedPolicy)} /><PolicyValue label="Execution" value={executionCapability ? `${executionCapability.backend_family} · ${executionCapability.segment_unit}` : `${backendFamily} · capability pending`} /></dl></div> : <p className="mt-3 text-[10.5px] leading-relaxed text-fg-subtle">{value.checkpointMode === "final_only" ? "No intermediate checkpoint can change training. This preserves existing behavior." : !adaptiveAvailable ? executionCapability?.reason || "This trainer/backend is final-only." : "Choose a policy to expose exact boundaries, evidence, and trainer capability notes."}</p>}
            </aside>
          </div>
          <RewardAuditBindingEditor trainerMode={value.trainerMode} backendFamily={backendFamily} value={value.rewardAudit} onChange={(rewardAudit) => onChange({ ...value, rewardAudit })} totalBudget={value.checkpointMode === "final_only" ? value.epochs : value.adaptiveBudget} budgetUnit={value.checkpointMode === "final_only" ? "epoch" : adaptiveUnit} compact />
          </div>
        ) : null}

        {step === 3 ? (
          <div className="grid gap-x-5 gap-y-4 md:grid-cols-2 xl:grid-cols-4">
            <Field label="Development suite"><SearchPicker value={value.suiteRevisionId} onChange={(suiteRevisionId) => onChange({ ...value, suiteRevisionId })} placeholder="Search development suites" options={suites.flatMap((suite) => { const revisionId = latestRevisionId(suite); return revisionId ? [{ value: revisionId, label: suite.name, description: `immutable revision · ${revisionId}`, keywords: suite.description ?? "" }] : []; })} /></Field>
            {value.checkpointMode === "final_only" ? <Field label="Full training budget"><div className="flex items-center gap-2"><Input type="number" min={1} value={value.epochs} onChange={(event) => onChange({ ...value, epochs: Math.max(1, Number(event.target.value)) })} /><span className="text-[11px] text-fg-subtle">epochs</span></div></Field> : <Field label={adaptiveUnit === "step" ? "Adaptive step budget" : "Adaptive cycle budget"}><div className="flex items-center gap-2"><Input type="number" min={1} value={value.adaptiveBudget} onChange={(event) => onChange({ ...value, adaptiveBudget: Math.max(1, Number(event.target.value)) })} /><span className="text-[11px] text-fg-subtle">{adaptiveUnit}s</span></div><p className="mt-1 text-[10px] text-fg-disabled">Training can pause only at resolved {adaptiveUnit} boundaries.</p></Field>}
            {value.kind === "sweep" ? <label className="flex items-start gap-2 border-l-2 border-border-strong px-3 py-2 text-[11px] leading-relaxed text-fg-muted md:col-span-2"><input className="mt-0.5" type="checkbox" checked={value.pruning} onChange={(event) => onChange({ ...value, pruning: event.target.checked })} /><span><b className="font-medium text-fg">Successive halving</b><br />Advance only after complete required seed coverage at synchronized boundaries.</span></label> : null}
            <div className="border-l-2 border-accent bg-accent-bg/20 px-4 py-3 md:col-span-2 xl:col-span-4"><div className="text-[9.5px] uppercase tracking-wider text-accent">Expected workstation load</div><div className="mt-1 font-mono text-[16px] text-fg">{trialCount} training run{trialCount === 1 ? "" : "s"}</div><p className="mt-1 text-[10.5px] text-fg-subtle">{seeds.length} seed{seeds.length === 1 ? "" : "s"} × {value.kind === "sweep" ? `${value.nTrials} candidate settings` : "one pinned configuration"}{selectedPolicy ? ` · ${formatSchedule(selectedPolicy)} · ${value.adaptiveBudget} ${adaptiveUnit}s max` : " · final evaluation only"}. Preflight resolves exact checkpoint, evaluation, and storage estimates.</p></div>
          </div>
        ) : null}

        {step === 4 ? (
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
            <div className="divide-y divide-border-subtle border-y border-border-subtle bg-surface/20"><ReviewRow label="Experiment" value={`${value.name || "Unnamed"} · ${value.kind}`} /><ReviewRow label="Training" value={`${value.trainerMode.toUpperCase()} · ${value.model || "model not selected"}`} mono /><ReviewRow label="Data" value={value.datasetVersionId || "No managed dataset bound"} mono /><ReviewRow label="Expansion" value={`${trialCount} runs · seeds ${seeds.join(", ") || "none"}`} /><ReviewRow label="Boundaries" value={selectedPolicy ? `${selectedPolicy.name} · revision ${selectedPolicy.revision_number}` : "Final only"} /><ReviewRow label="Training audit" value={value.rewardAudit.enabled ? `${value.rewardAudit.auditBoundaries || "resolved boundaries"} · fail pauses` : "Not enabled"} /><ReviewRow label="Evaluation" value={value.suiteRevisionId || "suite not selected"} mono /><ReviewRow label="Budget" value={value.checkpointMode === "final_only" ? `${value.epochs} epoch${value.epochs === 1 ? "" : "s"}${value.pruning ? " · successive halving enabled" : ""}` : `${value.adaptiveBudget} ${adaptiveUnit}s · bounded resumable segments${value.pruning ? " · successive halving enabled" : ""}`} /></div>
            <div className="rounded-md border border-border-subtle px-4 py-4"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Before queueing</div><ul className="mt-3 space-y-2 text-[10.5px] leading-relaxed text-fg-subtle"><li>• Every trial receives an isolated attempt directory.</li><li>• The suite revision and resolved inputs remain immutable.</li><li>• Low disk or RAM preflight can block the group before heavy work starts.</li></ul></div>
          </div>
        ) : null}

        <div className="mt-5 flex items-center justify-between border-t border-border-subtle pt-3">
          <Button variant="ghost" size="sm" onClick={() => step === 0 ? onCancel() : setStep((value) => Math.max(0, value - 1))}>{step === 0 ? "Cancel" : "Back"}</Button>
          <div className="flex items-center gap-3"><span className="text-[10px] text-fg-disabled">Step {step + 1} of {steps.length}</span>{step < steps.length - 1 ? <Button size="sm" onClick={() => setStep((value) => Math.min(steps.length - 1, value + 1))} disabled={!stepReady}>Continue <ChevronRight /></Button> : <Button size="sm" onClick={onCreate} disabled={!ready || pending}>{pending ? <Loader2 className="animate-spin" /> : <Play />} Queue {trialCount} run{trialCount === 1 ? "" : "s"}</Button>}</div>
        </div>
        {error ? <div className="mt-3 text-[12px] text-danger">{error}</div> : null}
      </div>
    </section>
  );
}

function GroupRail({ groups, selectedId, loading, onSelect }: { groups: RunGroup[]; selectedId: string; loading: boolean; onSelect: (id: string) => void }) {
  return (
    <aside className="border-b border-border-subtle bg-bg-subtle/25 xl:border-b-0 xl:border-r">
      <div className="border-b border-border-subtle px-4 py-3">
        <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Run groups</div>
      </div>
      <div className="divide-y divide-border-subtle">
        {groups.map((group) => (
          <button
            key={group.id}
            type="button"
            onClick={() => onSelect(group.id)}
            className={cn(
              "group relative w-full px-4 py-3 text-left transition-colors hover:bg-surface/60",
              selectedId === group.id && "bg-accent-bg/65",
            )}
          >
            {selectedId === group.id ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-[12.5px] font-medium text-fg">{group.name}</div>
                <div className="mt-1 font-mono text-[10px] uppercase tracking-wide text-fg-disabled">{group.kind} · {group.trainer_mode}</div>
              </div>
              <StatusDot status={group.status} />
            </div>
            <div className="mt-2 flex items-center gap-2 text-[10.5px] text-fg-muted">
              <span>{group.completed_trials ?? 0}/{group.n_trials} trials</span>
              {group.best_value != null ? <span className="font-mono text-accent">best {formatMetric(group.best_value)}</span> : null}
            </div>
          </button>
        ))}
        {loading ? <RailMessage><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading groups</RailMessage> : null}
        {!loading && !groups.length ? <RailMessage><CircleDashed className="h-3.5 w-3.5" /> No groups yet</RailMessage> : null}
      </div>
    </aside>
  );
}

function GroupWorkspace({ group, onCancel, onResume, onForkBest, actionPending }: { group: RunGroup; onCancel: () => void; onResume: (reason: string) => void; onForkBest: () => void; actionPending: boolean }) {
  const [resumeOpen, setResumeOpen] = useState(false);
  const [resumeReason, setResumeReason] = useState("");
  const trials = [...(group.trials ?? [])].sort((a, b) => a.ordinal - b.ordinal);
  const completed = group.completed_trials ?? trials.filter((trial) => trial.status === "completed").length;
  const progress = group.n_trials ? Math.min(100, (completed / group.n_trials) * 100) : 0;
  const canCancel = ["queued", "running", "paused", "awaiting_review"].includes(group.status);
  const canResume = ["cancelled", "interrupted", "failed", "paused", "stopped"].includes(group.status);
  return (
    <div>
      <header className="border-b border-border-subtle px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-[18px] font-medium text-fg">{group.name}</h1>
              <StatusBadge status={group.status} />
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10.5px] text-fg-disabled">
              <span>{group.id}</span>
              <span>{group.trainer_mode}</span>
              <span>{group.primary_metric} · {group.direction}</span>
              <span>{group.seeds.length} seed{group.seeds.length === 1 ? "" : "s"}</span>
            </div>
          </div>
          <div className="flex gap-2">
            {canResume ? <Button size="sm" variant="secondary" onClick={() => setResumeOpen((open) => !open)} disabled={actionPending}><RotateCcw /> Resume</Button> : null}
            {canCancel ? <Button size="sm" variant="ghost" onClick={onCancel} disabled={actionPending}><Ban /> Cancel</Button> : null}
            {group.best_trial_id ? (
              <Button size="sm" onClick={onForkBest} disabled={actionPending}>
                <Copy /> Fork best
              </Button>
            ) : null}
          </div>
        </div>
        {resumeOpen ? <div className="mt-3 flex flex-wrap items-center gap-2 border-l-2 border-warning pl-3"><Input className="h-8 min-w-[240px] flex-1 text-[11px]" value={resumeReason} onChange={(event) => setResumeReason(event.target.value)} placeholder="Required reason for resuming this stopped group" /><Button size="sm" onClick={() => { onResume(resumeReason.trim()); setResumeOpen(false); setResumeReason(""); }} disabled={!resumeReason.trim() || actionPending}>Resume group</Button></div> : null}
        <div className="mt-4 h-1 overflow-hidden rounded-full bg-surface">
          <div className="h-full bg-accent transition-[width] duration-500" style={{ width: `${progress}%` }} />
        </div>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-fg-muted">
          <span><b className="font-medium text-fg">{completed}</b> complete</span>
          <span><b className="font-medium text-fg">{group.failed_trials ?? 0}</b> failed</span>
          <span><b className="font-medium text-fg">{group.pruned_trials ?? 0}</b> pruned</span>
          <span><b className="font-medium text-fg">{group.n_trials}</b> total</span>
        </div>
      </header>

      <AdaptiveExperimentWorkspace group={group} />

      <section>
        <div className="grid grid-cols-[52px_minmax(140px,1fr)_90px_110px_90px_28px] items-center border-b border-border-subtle bg-bg-subtle/40 px-4 py-2 font-mono text-[9.5px] uppercase tracking-wider text-fg-disabled">
          <span>Trial</span><span>Changed settings</span><span>Seeds</span><span>Aggregate</span><span>Status</span><span />
        </div>
        <div className="divide-y divide-border-subtle">
          {trials.map((trial) => <TrialRow key={trial.id} trial={trial} best={trial.id === group.best_trial_id} />)}
          {!trials.length ? (
            <div className="px-5 py-16 text-center text-[12px] text-fg-muted">
              Trial records appear here as soon as the group is materialized.
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

function TrialRow({ trial, best }: { trial: ExperimentTrial; best: boolean }) {
  const [open, setOpen] = useState(false);
  const values = Object.entries(trial.parameters ?? {});
  const aggregate = trial.aggregate?.mean;
  return (
    <div className={cn(best && "bg-accent-bg/30")}>
      <button type="button" onClick={() => setOpen((value) => !value)} className="grid w-full grid-cols-[52px_minmax(140px,1fr)_90px_110px_90px_28px] items-center px-4 py-3 text-left hover:bg-surface/45">
        <span className="font-mono text-[11px] text-fg-subtle">{String(trial.ordinal + 1).padStart(2, "0")}</span>
        <span className="truncate font-mono text-[10.5px] text-fg-muted" title={formatParameters(values)}>{formatParameters(values) || "base config"}</span>
        <span className="text-[11px] text-fg-muted">{trial.runs?.length ?? 0}</span>
        <span className={cn("font-mono text-[11px]", best ? "font-medium text-accent" : "text-fg")}>{aggregate == null ? "—" : formatMetric(aggregate)} {best ? <Trophy className="ml-1 inline h-3 w-3" /> : null}</span>
        <span><StatusBadge status={trial.pruned ? "pruned" : trial.status} /></span>
        <ChevronRight className={cn("h-3.5 w-3.5 text-fg-disabled transition-transform", open && "rotate-90")} />
      </button>
      {open ? (
        <div className="border-t border-border-subtle bg-bg-subtle/30 px-5 py-3">
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <div className="mb-2 text-[10px] font-medium uppercase tracking-wider text-fg-disabled">Resolved settings</div>
              <dl className="space-y-1 font-mono text-[10.5px]">
                {values.map(([key, value]) => <div key={key} className="flex justify-between gap-3"><dt className="text-fg-muted">{key}</dt><dd className="truncate text-fg">{String(value)}</dd></div>)}
              </dl>
            </div>
            <div>
              <div className="mb-2 text-[10px] font-medium uppercase tracking-wider text-fg-disabled">Seed outcomes</div>
              <div className="space-y-1.5">
                {(trial.runs ?? []).map((run) => (
                  <div key={run.id} className="grid grid-cols-[60px_1fr_auto] items-center gap-2 text-[10.5px]">
                    <span className="font-mono text-fg-muted">seed {run.seed}</span>
                    <span className="truncate text-fg-subtle">{run.run_id || run.status}</span>
                    <span className="font-mono text-fg">{run.objective_value == null ? "—" : formatMetric(run.objective_value)}</span>
                  </div>
                ))}
                {!trial.runs?.length ? <div className="text-[10.5px] text-fg-disabled">No seed runs materialized yet.</div> : null}
              </div>
            </div>
          </div>
          {trial.prune_reason ? <div className="mt-3 text-[10.5px] text-warning">Stopped: {trial.prune_reason}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

function QueueRail({ items, activeLease, loading, onRetry }: { items: WorkItem[]; activeLease: Record<string, unknown> | null; loading: boolean; onRetry: (id: string) => void }) {
  const visible = items.filter((item) => ["running", "queued", "interrupted", "failed"].includes(item.status)).slice(0, 30);
  return (
    <aside className="bg-bg-subtle/20">
      <div className="border-b border-border-subtle px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Workstation queue</div>
          {activeLease ? <Badge tone="warning" size="sm">reserved</Badge> : <Badge tone="neutral" size="sm">idle</Badge>}
        </div>
        <p className="mt-1 text-[10.5px] leading-4 text-fg-muted">One heavy operation runs at a time. Serving reservations block new work.</p>
      </div>
      <div className="divide-y divide-border-subtle">
        {visible.map((item) => <QueueItem key={item.id} item={item} onRetry={() => onRetry(item.id)} />)}
        {loading ? <RailMessage><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading queue</RailMessage> : null}
        {!loading && !visible.length ? <RailMessage><CheckCircle2 className="h-3.5 w-3.5 text-success" /> Queue is clear</RailMessage> : null}
      </div>
    </aside>
  );
}

function QueueItem({ item, onRetry }: { item: WorkItem; onRetry: () => void }) {
  const progress = item.progress_total ? Math.min(100, 100 * (item.progress_current ?? 0) / item.progress_total) : item.status === "completed" ? 100 : 0;
  return (
    <div className="px-4 py-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[11.5px] font-medium text-fg">{friendlyKind(item.kind)}</div>
          <div className="mt-0.5 truncate font-mono text-[9.5px] text-fg-disabled">{item.run_id || item.trial_id || item.id}</div>
        </div>
        <StatusDot status={item.status} />
      </div>
      <div className="mt-2 h-0.5 overflow-hidden rounded-full bg-surface">
        <div className={cn("h-full transition-[width] duration-500", item.status === "failed" ? "bg-danger" : "bg-accent")} style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-1.5 flex items-center justify-between text-[9.5px] text-fg-disabled">
        <span>{item.stage || item.status}</span>
        {item.status === "failed" || item.status === "interrupted" ? <button type="button" onClick={onRetry} className="text-accent hover:underline">retry</button> : <span>p{item.priority}</span>}
      </div>
      {item.error ? <div className="mt-1 line-clamp-2 text-[9.5px] leading-4 text-danger">{item.error}</div> : null}
    </div>
  );
}

function EmptyWorkspace({ loading, onCreate }: { loading: boolean; onCreate: () => void }) {
  return (
    <div className="flex min-h-[520px] items-center justify-center px-6">
      <div className="max-w-sm text-center">
        {loading ? <Loader2 className="mx-auto h-5 w-5 animate-spin text-fg-disabled" /> : <FlaskConical className="mx-auto h-6 w-6 text-fg-disabled" />}
        <h2 className="mt-3 text-[15px] font-medium text-fg">{loading ? "Loading experiments" : "No run group selected"}</h2>
        <p className="mt-1 text-[12px] leading-5 text-fg-muted">Start with a three-seed repeat to measure variance before spending budget on a sweep.</p>
        {!loading ? <Button className="mt-4" size="sm" onClick={onCreate}><Plus /> New group</Button> : null}
      </div>
    </div>
  );
}

function buildGroupPayload(value: ComposerState): RunGroupCreatePayload {
  const seeds = parseSeeds(value.seeds);
  const baseConfig: Record<string, unknown> = {
    mode: value.trainerMode,
    model: value.model.trim(),
    seed: seeds[0] ?? 42,
    ...(value.checkpointMode === "final_only" ? { epochs: value.epochs } : checkpointUnit(value.trainerMode) === "step" ? { max_steps: value.adaptiveBudget } : { cycles: value.adaptiveBudget }),
  };
  if (value.datasetVersionId.trim()) {
    baseConfig.dataset_version_id = value.datasetVersionId.trim();
    baseConfig.dataset_split = "train";
  }
  if (value.rewardAudit.enabled) {
    baseConfig.reward_system_revision_id = value.rewardAudit.rewardSystemRevisionId;
    baseConfig.reward_audit_protocol_revision_id = value.rewardAudit.auditProtocolRevisionId;
    baseConfig.reward_integrity_profile_revision_id = value.rewardAudit.integrityProfileRevisionId;
    baseConfig.reward_audit_boundaries = parseRewardAuditBoundaries(value.rewardAudit.auditBoundaries);
    if (value.rewardAudit.developmentSuiteRevisionId) baseConfig.development_suite_revision_id = value.rewardAudit.developmentSuiteRevisionId;
  }
  const payload: RunGroupCreatePayload = {
    version: value.checkpointPolicyRevisionId ? 2 : 1,
    name: value.name.trim(),
    kind: value.kind,
    trainer_mode: value.trainerMode,
    suite_revision_id: value.suiteRevisionId,
    base_config: baseConfig,
    seeds,
    n_trials: value.kind === "repeat" ? 1 : value.nTrials,
    checkpoint_policy_revision_id: value.checkpointPolicyRevisionId || null,
  };
  if (value.kind === "sweep") {
    const low = Number(value.learningRateLow);
    const high = Number(value.learningRateHigh);
    const batchSizes = parseSeeds(value.batchSizes);
    const loraRanks = parseSeeds(value.loraRanks);
    payload.search_space = {
      learning_rate: { kind: "log_uniform", low, high },
      ...(batchSizes.length ? { batch_size: { kind: "choice", values: batchSizes } } : {}),
      ...(loraRanks.length ? { lora_rank: { kind: "choice", values: loraRanks } } : {}),
    };
    payload.sampler = "random";
    payload.sampler_seed = 42;
    const cycleBudgetModes = new Set(["raft", "vlm", "audio", "reasoning", "agentic"]);
    payload.pruning = {
      enabled: value.pruning,
      reduction_factor: 3,
      budgets: value.pruning
        ? cycleBudgetModes.has(value.trainerMode)
          ? [1, 2, 3]
          : [100, 300, 900]
        : [],
    };
  }
  return payload;
}

function rewardAuditBindingReady(value: RewardAuditBindingValue): boolean {
  return Boolean(value.rewardSystemRevisionId && value.auditProtocolRevisionId && value.integrityProfileRevisionId);
}

function parseRewardAuditBoundaries(value: string): Array<number | string> | undefined {
  const boundaries = value.split(",").map((part) => part.trim()).filter(Boolean).slice(0, 4).map((part) => {
    if (part.endsWith("%")) return part;
    const number = Number(part);
    return Number.isFinite(number) && number > 0 ? number : part;
  });
  return boundaries.length ? boundaries : undefined;
}

function parseSeeds(value: string): number[] {
  return [...new Set(value.split(",").map((part) => Number(part.trim())).filter((seed) => Number.isInteger(seed) && seed >= 0))];
}

function latestRevisionId(suite: BenchmarkSuite): string {
  return suite.latest_revision?.id || suite.latest_revision_id || suite.revisions?.at(-1)?.id || "";
}

function terminalStatus(status: string): boolean {
  return ["completed", "failed", "cancelled", "stopped"].includes(status);
}

function policyIdentity(policy: CheckpointPolicyRevision): string {
  return policy.id || `${policy.policy_id}:r${policy.revision_number}`;
}

function checkpointUnit(trainer: TrainingMode): "step" | "cycle" {
  return new Set<TrainingMode>(["raft", "vlm", "audio", "reasoning", "agentic"]).has(trainer) ? "cycle" : "step";
}

function formatSchedule(policy: CheckpointPolicyRevision): string {
  const schedule = policy.schedule;
  if (schedule.kind === "final_only" || schedule.mode === "final") return "final only";
  if (schedule.percentages?.length) return `${schedule.percentages.map((value) => `${Math.round(value * 100)}%`).join(" / ")}`;
  if (schedule.boundaries?.length) return `${schedule.boundaries.join(" / ")} ${schedule.unit ?? "steps"}`;
  if (schedule.interval) return `every ${schedule.interval} ${schedule.unit ?? "steps"}`;
  return (schedule.kind || schedule.mode)?.replaceAll("_", " ") || schedule.unit || "declared boundaries";
}

function formatRetention(policy: CheckpointPolicyRevision): string {
  const retention = policy.retention;
  if (!retention) return "Keep last + best · reviewed cleanup";
  const kept = [retention.keep_last ? `last ${retention.keep_last}` : "", retention.keep_best ? `best ${retention.keep_best}` : "", retention.keep_every_n_boundaries ? `every ${retention.keep_every_n_boundaries}` : ""].filter(Boolean).join(" · ");
  return `${kept || "protected references only"}${retention.review_before_cleanup ? " · reviewed" : ""}`;
}

function policyActionLabel(policy: CheckpointPolicyRevision): string {
  if (!policy.automatic_actions) return "Manual review at boundaries";
  return policy.rules.length ? "Rules decide at boundaries" : "Continue after complete evidence";
}

function StudiesWorkspace() {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState("");
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [question, setQuestion] = useState("");
  const [design, setDesign] = useState<"paired_ab" | "dose_response" | "factorial_2x2">("paired_ab");
  const [intervention, setIntervention] = useState("Adapted corpus");
  const studies = useQuery({
    queryKey: ["adaptation-studies"],
    queryFn: () => api.adaptationStudies({ limit: 200 }),
    retry: false,
  });
  const selected = (studies.data?.items ?? []).find((item) => item.id === selectedId)
    ?? studies.data?.items?.[0]
    ?? null;
  const launchPlan = useQuery({
    queryKey: ["adaptation-study-launch-plan", selected?.latest_protocol_revision_id],
    queryFn: () => api.adaptationStudyLaunchPlan(selected!.latest_protocol_revision_id!),
    enabled: Boolean(selected?.latest_protocol_revision_id),
    retry: false,
  });
  const launchStudy = useMutation({
    mutationFn: () => api.launchAdaptationStudy(selected!.latest_protocol_revision_id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["adaptation-studies"] });
      queryClient.invalidateQueries({ queryKey: ["adaptation-study-launch-plan"] });
    },
  });
  const create = useMutation({
    mutationFn: async () => {
      const study = await api.createAdaptationStudy({
        name: name.trim(),
        description: question.trim(),
      });
      const arms = design === "factorial_2x2"
        ? [
            { name: "Control", is_control: true, factor_values: { data: "base", method: "base" }, launch_config: {} },
            { name: "Data only", factor_values: { data: intervention.trim(), method: "base" }, launch_config: {} },
            { name: "Method only", factor_values: { data: "base", method: "adapted" }, launch_config: {} },
            { name: "Data + method", factor_values: { data: intervention.trim(), method: "adapted" }, launch_config: {} },
          ]
        : design === "dose_response"
          ? [
              { name: "Control", is_control: true, factor_values: { dose: 0 }, launch_config: {} },
              { name: "Medium dose", factor_values: { dose: 0.5 }, launch_config: {} },
              { name: "Full dose", factor_values: { dose: 1 }, launch_config: {} },
            ]
          : [
              { name: "Control", is_control: true, factor_values: { intervention: "base" }, launch_config: {} },
              { name: intervention.trim() || "Adapted", factor_values: { intervention: intervention.trim() || "adapted" }, launch_config: {} },
            ];
      await api.createAdaptationStudyProtocol(study.id, {
        design_kind: design,
        question: question.trim(),
        arms,
        seeds: [17, 42, 101],
        development_suite_purpose: "development",
        retention_suite_purpose: "development",
        contrasts: arms.slice(1).map((arm, index) => ({
          name: `${arm.name} versus control`,
          left_arm: "Control",
          right_arm: arm.name,
          metric: "primary_metric",
          direction: "maximize",
          conclusion_kind: "superiority",
          practical_margin: 0,
          ordinal: index,
        })),
      });
      return study;
    },
    onSuccess: (study) => {
      queryClient.invalidateQueries({ queryKey: ["adaptation-studies"] });
      setSelectedId(study.id);
      setCreating(false);
      setName("");
      setQuestion("");
    },
  });
  return (
    <div className="grid min-h-[calc(100vh-152px)] xl:grid-cols-[280px_minmax(0,1fr)_300px]">
      <aside className="border-b border-border-subtle bg-bg-subtle/25 xl:border-b-0 xl:border-r">
        <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3"><div><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Study protocols</div><div className="mt-1 text-xs text-fg-muted">{studies.data?.total ?? 0} controlled questions</div></div><Button size="sm" onClick={() => setCreating(true)}><Plus />New</Button></div>
        <div className="divide-y divide-border-subtle">{(studies.data?.items ?? []).map((study: AdaptationStudy) => <button type="button" key={study.id} onClick={() => setSelectedId(study.id)} className={cn("w-full px-4 py-3 text-left hover:bg-surface", selected?.id === study.id && "bg-accent-bg/45")}><div className="text-[11.5px] font-medium text-fg">{study.name}</div><div className="mt-1 text-[10px] text-fg-subtle">{study.status} · immutable protocol</div></button>)}</div>
      </aside>
      <main className="min-w-0 border-b border-border-subtle px-5 py-5 xl:border-b-0 xl:border-r">
        {creating ? <div className="mx-auto max-w-2xl"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-accent">Question → Approach → Evidence → Cost → Review</div><h2 className="mt-2 text-xl font-medium text-fg">Compare training approaches</h2><p className="mt-2 text-[11px] leading-5 text-fg-muted">Choose a simple question. Halo Forge prepares the matched runs and repeat seeds for you.</p><div className="mt-6 grid gap-4"><Field label="Study name"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Which approach works better?" /></Field><Field label="What do you want to learn?"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} className="min-h-24 w-full rounded-md border border-border bg-surface px-3 py-2 text-xs text-fg outline-none focus:border-accent" placeholder="Does the adapted data improve the result without hurting general capability?" /></Field><Field label="Study template"><NativeSelect value={design} onChange={(value) => setDesign(value as typeof design)}><option value="paired_ab">Compare two approaches</option><option value="dose_response">Try different data amounts</option><option value="factorial_2x2">Test data and method together</option></NativeSelect></Field><Field label="Approach to compare"><Input value={intervention} onChange={(event) => setIntervention(event.target.value)} placeholder="New dataset or training method" /></Field><div className="grid gap-px border border-border-subtle bg-border-subtle sm:grid-cols-3"><ReviewRow label="Repeat runs" value="3 per approach" /><ReviewRow label="Checks" value="Improvement + retention" /><ReviewRow label="Before launch" value="Time and storage estimate" /></div><div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button><Button onClick={() => create.mutate()} disabled={!name.trim() || !question.trim() || create.isPending}>{create.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Save study plan</Button></div>{create.error instanceof Error ? <p className="text-[10px] text-danger">{create.error.message}</p> : null}</div></div> : selected ? <div className="mx-auto max-w-3xl"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-accent">Study plan</div><h2 className="mt-2 text-xl font-medium text-fg">{selected.name}</h2><p className="mt-2 text-[11px] leading-5 text-fg-muted">{selected.description || "No description"}</p><div className="mt-6 border-y border-border-subtle"><ReviewRow label="Status" value={selected.status === "draft" ? "Setup not finished" : selected.status} /><ReviewRow label="Planned runs" value={launchPlan.data ? `${launchPlan.data.run_count} total` : "Calculating"} /><ReviewRow label="Repeat runs" value={launchPlan.data ? `${launchPlan.data.seed_count} per approach` : "Three per approach"} /><ReviewRow label="Result" value="Improvement and retention stay separate" /></div>{launchPlan.data?.blockers.length ? <div className="mt-4 border-l-2 border-warning bg-warning-bg px-4 py-3"><div className="text-xs font-medium text-fg">Choose the training setup</div><div className="mt-1 text-[10.5px] leading-5 text-fg-muted">{launchPlan.data.blockers[0]}</div></div> : <Button className="mt-5" size="lg" disabled={launchStudy.isPending} onClick={() => launchStudy.mutate()}>{launchStudy.isPending ? <Loader2 className="animate-spin" /> : <Play />}Launch {launchPlan.data?.run_count ?? ""} study runs</Button>}</div> : <div className="flex min-h-[420px] items-center justify-center text-center"><div><FlaskConical className="mx-auto h-6 w-6 text-fg-disabled" /><div className="mt-3 text-sm text-fg">No comparison yet</div><p className="mt-1 max-w-sm text-[11px] leading-5 text-fg-muted">Start with a practical question and one of three guided templates.</p></div></div>}
      </main>
      <aside className="bg-bg-subtle/25 px-4 py-5"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Research guardrails</div><div className="mt-4 space-y-3 text-[10.5px] leading-5 text-fg-muted"><p>Development evidence can guide the study. Holdout, operational, test, and canary evidence cannot.</p><p>Domain uptake and general-capability retention are never collapsed into one score.</p><p>Only a completed randomized paired design is labeled causal; other results remain comparative.</p></div></aside>
    </div>
  );
}

function formatParameters(values: Array<[string, unknown]>): string {
  return values.map(([key, value]) => `${key}=${String(value)}`).join(" · ");
}

function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function friendlyKind(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function StatusDot({ status }: { status: string }) {
  if (status === "running") return <span className="mt-1 h-2 w-2 animate-pulse rounded-full bg-accent" title="running" />;
  if (status === "completed") return <span className="mt-1 h-2 w-2 rounded-full bg-success" title="completed" />;
  if (status === "failed") return <span className="mt-1 h-2 w-2 rounded-full bg-danger" title="failed" />;
  if (status === "queued") return <span className="mt-1 h-2 w-2 rounded-full border border-fg-disabled" title="queued" />;
  if (status === "awaiting_review" || status === "paused") return <span className="mt-1 h-2 w-2 rounded-full bg-warning" title="awaiting review" />;
  return <span className="mt-1 h-2 w-2 rounded-full bg-fg-disabled" title={status} />;
}

function StatusBadge({ status }: { status: string }) {
  const tone = status === "completed" ? "success" : status === "failed" ? "danger" : ["running", "awaiting_review", "paused"].includes(status) ? "warning" : "neutral";
  return <Badge tone={tone} size="sm">{status}</Badge>;
}

function datasetFitsTrainer(version: DatasetVersion, trainer: TrainingMode): boolean {
  if (version.status !== "ready") return false;
  const declared = version.compatible_trainers ?? [];
  if (!declared.length) return true;
  const match = declared.find((item) => item.trainer_mode === trainer);
  return match?.compatible === true;
}

function ReviewRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid gap-1 px-4 py-3 sm:grid-cols-[120px_minmax(0,1fr)] sm:items-start">
      <span className="text-[10px] uppercase tracking-wider text-fg-disabled">{label}</span>
      <span className={cn("break-all text-[11.5px] text-fg-muted", mono && "font-mono text-[10.5px]")}>{value}</span>
    </div>
  );
}

function CheckpointModeButton({ active, disabled, icon: Icon, label, description, onClick }: { active: boolean; disabled?: boolean; icon: typeof Gauge; label: string; description: string; onClick: () => void }) {
  return <button type="button" disabled={disabled} onClick={onClick} className={cn("min-h-28 bg-bg px-4 py-3 text-left transition-colors hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50", active && "bg-accent-bg/60")}><Icon className={cn("h-4 w-4", active ? "text-accent" : "text-fg-disabled")} /><div className="mt-3 text-[11.5px] font-medium text-fg">{label}</div><p className="mt-1 text-[10px] leading-relaxed text-fg-subtle">{description}</p></button>;
}

function PolicyValue({ label, value }: { label: string; value: string }) {
  return <div className="flex items-start justify-between gap-3 py-2"><dt className="text-[9.5px] text-fg-subtle">{label}</dt><dd className="max-w-[65%] text-right text-[9.5px] text-fg-muted">{value}</dd></div>;
}

function PolicyCreator({ mode, trainerMode, unit, suiteRevisionId, defaultMetric, defaultDirection, onCreated }: { mode: "periodic" | "guarded"; trainerMode: TrainingMode; unit: "step" | "cycle"; suiteRevisionId: string; defaultMetric: string; defaultDirection: "maximize" | "minimize"; onCreated: (id: string) => void }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(mode === "guarded" ? "Guarded checkpoint policy" : "Periodic observation policy");
  const [metric, setMetric] = useState(defaultMetric);
  const [direction, setDirection] = useState<"maximize" | "minimize">(defaultDirection);
  const [percentages, setPercentages] = useState("25, 50, 75, 100");
  const [patience, setPatience] = useState(2);
  const [keepLast, setKeepLast] = useState(1);
  const [keepBest, setKeepBest] = useState(1);
  const [keepEvery, setKeepEvery] = useState("");
  const boundaries = [...new Set(percentages.split(",").map((value) => Number(value.trim()) / 100).filter((value) => value > 0 && value <= 1))].sort((a, b) => a - b);
  const create = useMutation({
    mutationFn: () => {
      const suffix = globalThis.crypto?.randomUUID?.().slice(0, 8) ?? String(Date.now());
      return api.createCheckpointPolicy({
        policy_id: `${slugify(name)}-${suffix}`,
        revision_number: 1,
        name: name.trim(),
        description: mode === "guarded" ? "Pause at a verified boundary when the development objective plateaus." : "Observe immutable development evidence at declared checkpoints without changing training.",
        development_suite_revision_id: suiteRevisionId,
        primary_metric: metric.trim(),
        direction,
        schedule: { mode: "percentages", unit, percentages: boundaries },
        rules: mode === "guarded" ? [{ kind: "plateau", metric: metric.trim(), direction, comparison: "previous", minimum_delta: 0, practical_delta: 0, patience, on_breach: "pause", required: true }] : [],
        retention: { keep_last: keepLast, keep_every_n_boundaries: keepEvery.trim() ? Math.max(1, Number(keepEvery)) : null, keep_best: keepBest, protect_evaluated: true, protect_decision_referenced: true, protect_lineage_referenced: true, review_before_cleanup: true },
        guardrail_suite_revision_ids: [],
        automatic_actions: true,
        compatible_capabilities: [],
        version: 1,
      });
    },
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["checkpoint-policies", trainerMode] });
      onCreated(policyIdentity(created));
      setOpen(false);
    },
  });
  return <div className="mt-3 border-t border-border-subtle pt-3"><button type="button" onClick={() => setOpen((value) => !value)} className="text-[10.5px] text-accent hover:underline">{open ? "Close policy editor" : "Create a named policy"}</button>{open ? <div className="mt-3 grid gap-3 border-l-2 border-accent pl-3 sm:grid-cols-2"><Field label="Policy name"><Input value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label="Checkpoint percentages"><Input value={percentages} onChange={(event) => setPercentages(event.target.value)} placeholder="25, 50, 75, 100" mono /></Field><Field label="Primary metric"><Input value={metric} onChange={(event) => setMetric(event.target.value)} placeholder="accuracy" /></Field><Field label="Direction"><NativeSelect value={direction} onChange={(value) => setDirection(value as "maximize" | "minimize")}><option value="maximize">Maximize</option><option value="minimize">Minimize</option></NativeSelect></Field>{mode === "guarded" ? <Field label="Plateau patience"><Input type="number" min={1} value={patience} onChange={(event) => setPatience(Math.max(1, Number(event.target.value)))} /></Field> : null}<div className="grid grid-cols-3 gap-2 sm:col-span-2"><Field label="Keep latest"><Input type="number" min={0} value={keepLast} onChange={(event) => setKeepLast(Math.max(0, Number(event.target.value)))} /></Field><Field label="Keep best"><Input type="number" min={0} value={keepBest} onChange={(event) => setKeepBest(Math.max(0, Number(event.target.value)))} /></Field><Field label="Keep every"><Input type="number" min={1} value={keepEvery} onChange={(event) => setKeepEvery(event.target.value)} placeholder="Optional" /></Field></div><p className="text-[9.5px] leading-relaxed text-fg-disabled sm:col-span-2">Evaluated, decision-referenced, and lineage-required checkpoints stay protected. Cleanup always requires review.</p><div className="flex items-end"><Button size="sm" onClick={() => create.mutate()} disabled={!name.trim() || !metric.trim() || !suiteRevisionId || !boundaries.length || (!keepLast && !keepBest && !keepEvery.trim()) || create.isPending}>{create.isPending ? <Loader2 className="animate-spin" /> : <Plus />} Save immutable policy</Button></div>{!suiteRevisionId ? <p className="text-[10px] text-warning sm:col-span-2">Choose the development suite before saving this policy.</p> : null}{create.error instanceof Error ? <p className="text-[10px] text-danger sm:col-span-2">{create.error.message}</p> : null}</div> : null}</div>;
}

function slugify(value: string): string {
  return value.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "checkpoint-policy";
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="space-y-1.5"><Label className="text-[10.5px] uppercase tracking-wider text-fg-subtle">{label}</Label>{children}</div>;
}

function NativeSelect({ value, onChange, children }: { value: string; onChange: (value: string) => void; children: React.ReactNode }) {
  return <select value={value} onChange={(event) => onChange(event.target.value)} className="h-8 w-full rounded-md border border-border bg-surface px-2.5 text-[12px] text-fg outline-none focus:border-accent focus:ring-1 focus:ring-accent">{children}</select>;
}

function RailMessage({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-2 px-4 py-4 text-[11px] text-fg-muted">{children}</div>;
}

function Readout({ label, value }: { label: string; value: string }) {
  return <span className="inline-flex items-center gap-1.5"><span className="tracking-wider text-fg-disabled">{label}</span><span className="text-fg">{value}</span></span>;
}
