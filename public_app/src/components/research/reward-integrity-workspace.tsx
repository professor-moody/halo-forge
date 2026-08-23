import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CircleDashed,
  ChevronLeft,
  ChevronRight,
  GitCompareArrows,
  Image as ImageIcon,
  Loader2,
  Music2,
  Pause,
  Play,
  Plus,
  RotateCcw,
  ShieldCheck,
  Square,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import {
  api,
  type RewardIntegrityAudit,
  type RewardIntegrityComparison,
  type RewardIntegrityComparisonPair,
  type RewardIntegrityMetric,
  type RewardIntegrityObservation,
  type RewardIntegrityProfileRevision,
  type RewardSystem,
  type VerifierProfile,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SearchPicker } from "@/components/ui/search-picker";
import { cn } from "@/lib/utils";

export type RewardAuditStudioView = "profiles" | "results" | "compare";

export function RewardIntegrityWorkspace({
  view,
  selectedAuditId,
  baseAuditId,
  candidateAuditId,
  selectedSampleId,
  page = 1,
  classification,
  onView,
  onAudit,
  onCompare,
  onSample,
  onPage,
  onClassification,
}: {
  view: RewardAuditStudioView;
  selectedAuditId?: string;
  baseAuditId?: string;
  candidateAuditId?: string;
  selectedSampleId?: string;
  page?: number;
  classification?: string;
  onView: (view: RewardAuditStudioView) => void;
  onAudit: (id?: string) => void;
  onCompare: (base?: string, candidate?: string) => void;
  onSample: (id?: string) => void;
  onPage: (page: number) => void;
  onClassification: (classification?: string) => void;
}) {
  return (
    <div className="min-h-[calc(100vh-152px)]">
      <nav className="flex gap-1 overflow-x-auto border-b border-border bg-bg px-4" aria-label="Training audit views">
        {(["profiles", "results", "compare"] as RewardAuditStudioView[]).map((item) => (
          <button key={item} type="button" onClick={() => onView(item)} className={cn("relative h-10 shrink-0 px-3 text-[10.5px] capitalize transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset", item === view ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>
            {item}{item === view ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}
          </button>
        ))}
      </nav>
      {view === "profiles" ? <RewardProfilesView /> : null}
      {view === "results" ? <RewardAuditResultsView selectedAuditId={selectedAuditId} selectedSampleId={selectedSampleId} page={page} classification={classification} onAudit={onAudit} onSample={onSample} onPage={onPage} onClassification={onClassification} /> : null}
      {view === "compare" ? <RewardAuditCompareView baseAuditId={baseAuditId} candidateAuditId={candidateAuditId} selectedPairId={selectedSampleId} page={page} onCompare={onCompare} onPair={onSample} onPage={onPage} /> : null}
    </div>
  );
}

export function RunIntegrityStrip({ runId }: { runId: string }) {
  const audits = useQuery({
    queryKey: ["reward-integrity-audits", "run", runId],
    queryFn: () => api.runRewardIntegrityAudits(runId, { limit: 100 }),
    refetchInterval: (query) => query.state.data?.items.some((item) => ["queued", "running"].includes(item.status)) ? 3_000 : false,
    retry: false,
  });
  const latestAuditId = audits.data?.items[0]?.id ?? "";
  const latestDetail = useQuery({ queryKey: ["reward-integrity-audit", latestAuditId], queryFn: () => api.rewardIntegrityAudit(latestAuditId), enabled: Boolean(latestAuditId), retry: false });
  if (audits.isLoading) return <div className="flex items-center gap-2 border-y border-border-subtle bg-bg-subtle/35 px-4 py-2 text-[10px] text-fg-muted"><Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />Checking training-signal evidence</div>;
  if (audits.isError || !audits.data?.items.length) return null;
  const items = audits.data.items;
  const latest = latestDetail.data ?? items[0];
  const awaiting = latest.decision?.override !== true && (["pause", "awaiting_review"].includes(latest.decision?.action ?? "") || ["fail", "incomplete_evidence"].includes(latest.decision?.decision ?? "")) ? latest : undefined;
  return (
    <section className={cn("flex flex-wrap items-center gap-x-5 gap-y-2 border-y px-4 py-2.5", awaiting ? "border-warning/35 bg-warning/5" : "border-border-subtle bg-bg-subtle/35")} aria-label="Training signal integrity">
      <div className="flex items-center gap-2">
        {awaiting ? <Pause className="h-3.5 w-3.5 text-warning" /> : <ShieldCheck className="h-3.5 w-3.5 text-accent" />}
        <span className="text-[10px] font-medium text-fg">Signal integrity</span>
        <StatusBadge value={awaiting?.decision?.decision || latest.decision?.decision || latest.status} />
      </div>
      <StripValue label="BOUNDARIES" value={`${items.length}`} />
      <StripValue label="DECISION" value={humanize(latest.decision?.decision || latest.status)} />
      <StripValue label="COVERAGE" value={percent(metricValue(latest.metrics, "paired_coverage") ?? metricValue(latest.metrics, "coverage"))} />
      <StripValue label="LATEST" value={auditBoundaryLabel(latest)} />
      <Link to="/runs/$runId" params={{ runId }} search={{ tab: "evaluation", evidence: "training-audits", audit: (awaiting ?? latest).id }} className="ml-auto inline-flex items-center gap-1 text-[9.5px] text-accent hover:underline">
        {awaiting ? "Review pause" : "Open audits"}<ArrowRight className="h-3 w-3" />
      </Link>
    </section>
  );
}

export function RunRewardAuditWorkspace({
  runId,
  selectedAuditId,
  selectedSampleId,
  page = 1,
  classification,
  onAudit,
  onSample,
  onPage,
  onClassification,
}: {
  runId: string;
  selectedAuditId?: string;
  selectedSampleId?: string;
  page?: number;
  classification?: string;
  onAudit: (id?: string) => void;
  onSample: (id?: string) => void;
  onPage: (page: number) => void;
  onClassification: (classification?: string) => void;
}) {
  return <RewardAuditResultsView runId={runId} selectedAuditId={selectedAuditId} selectedSampleId={selectedSampleId} page={page} classification={classification} onAudit={onAudit} onSample={onSample} onPage={onPage} onClassification={onClassification} />;
}

function RewardProfilesView() {
  const queryClient = useQueryClient();
  const systems = useQuery({ queryKey: ["reward-systems", "studio"], queryFn: () => api.listRewardSystems({ limit: 200 }), retry: false });
  const protocols = useQuery({ queryKey: ["reward-audit-protocols", "studio"], queryFn: () => api.listRewardAuditProtocols({ limit: 100 }), retry: false });
  const profiles = useQuery({ queryKey: ["reward-integrity-profiles", "studio"], queryFn: () => api.listRewardIntegrityProfiles({ limit: 100 }), retry: false });
  const verifiers = useQuery({ queryKey: ["verifier-profiles", "reward-system-creator", "qualified"], queryFn: () => api.listVerifierProfiles({ qualification: "pass", limit: 200 }), retry: false });
  const [selectedId, setSelectedId] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const create = useMutation({ mutationFn: (payload: Record<string, unknown>) => api.createRewardSystem(payload), onSuccess: (item) => { queryClient.invalidateQueries({ queryKey: ["reward-systems"] }); setSelectedId(item.id); setCreateOpen(false); } });
  useEffect(() => { if (!selectedId && systems.data?.items[0]) setSelectedId(systems.data.items[0].id); }, [selectedId, systems.data?.items]);
  const selectedSummary = systems.data?.items.find((item) => item.id === selectedId);
  const selectedDetail = useQuery({ queryKey: ["reward-system", selectedId], queryFn: () => api.rewardSystem(selectedId), enabled: Boolean(selectedId && !createOpen), retry: false });
  const selected = selectedDetail.data?.id ? selectedDetail.data : selectedSummary;
  return (
    <div className="grid min-h-[560px] lg:grid-cols-[290px_minmax(0,1fr)]">
      <aside className="border-b border-border bg-bg-subtle/25 lg:border-b-0 lg:border-r">
        <RailHeader eyebrow="REWARD SYSTEMS" title={`${systems.data?.total ?? systems.data?.items.length ?? 0} immutable systems`} action={<Button size="icon" variant="ghost" onClick={() => setCreateOpen(true)} aria-label="Create reward system"><Plus /></Button>} />
        {systems.isLoading ? <Loading label="Loading reward systems" /> : systems.isError ? <Unavailable label="Reward systems are unavailable" /> : <div className="divide-y divide-border-subtle">{systems.data?.items.map((item) => <RewardSystemRow key={item.id} item={item} selected={selectedId === item.id} onSelect={() => setSelectedId(item.id)} />)}{!systems.data?.items.length ? <Empty label="Create a qualified reward system from the CLI or API, then return here to bind it to training." /> : null}</div>}
      </aside>
      <main className="min-w-0 p-5 lg:p-8">
        {createOpen ? <RewardSystemCreator verifiers={verifiers.data?.items ?? []} onCancel={() => setCreateOpen(false)} onCreate={(payload) => create.mutate(payload)} pending={create.isPending} error={create.error instanceof Error ? create.error.message : null} /> : selectedDetail.isLoading ? <Loading label="Opening immutable reward system" /> : selected ? <RewardSystemDetail item={selected} protocols={protocols.data?.items ?? []} profiles={profiles.data?.items ?? []} /> : <Empty label="Select a reward system to inspect its optimizer, sentinel, reward mapping, and compatible training paths." />}
      </main>
    </div>
  );
}

function RewardSystemCreator({ verifiers, onCancel, onCreate, pending, error }: { verifiers: VerifierProfile[]; onCancel: () => void; onCreate: (payload: Record<string, unknown>) => void; pending: boolean; error: string | null }) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [optimizerId, setOptimizerId] = useState("");
  const [sentinelId, setSentinelId] = useState("");
  const [modality, setModality] = useState("text");
  const [taskType, setTaskType] = useState("binary");
  const [minimum, setMinimum] = useState("0");
  const [maximum, setMaximum] = useState("1");
  const [threshold, setThreshold] = useState("0.5");
  const [failureBehavior, setFailureBehavior] = useState("fail_closed");
  const optimizer = verifierRevision(verifiers, optimizerId);
  const sentinel = verifierRevision(verifiers, sentinelId);
  const correlated = Boolean(optimizer && sentinel && verifierFingerprints(optimizer).some((value) => verifierFingerprints(sentinel).includes(value)));
  const verifierOptions = verifiers.flatMap((profile) => profile.latest_revision ? [{ value: profile.latest_revision.id, label: profile.name, description: `${profile.latest_revision.family.replaceAll("_", " ")} · ${profile.latest_revision.modality} · ${profile.latest_revision.task_type}`, status: profile.latest_revision.alias || profile.latest_revision.qualification_state || "qualified", keywords: `${profile.id} ${profile.latest_revision.id}` }] : []);
  const sentinelOptions = verifierOptions.filter((option) => option.value !== optimizerId);
  const steps = ["Training verifier", "Sentinel", "Reward mapping", "Review"];
  const stepReady = step === 0 ? Boolean(name.trim() && optimizer) : step === 1 ? Boolean(sentinel && !correlated) : step === 2 ? Number.isFinite(Number(minimum)) && Number.isFinite(Number(maximum)) && Number(maximum) > Number(minimum) && Number.isFinite(Number(threshold)) && Number(threshold) >= 0 && Number(threshold) <= 1 : Boolean(name.trim() && optimizer && sentinel && !correlated);
  const payload = { name: name.trim(), description: description.trim() || undefined, definition: { optimizer_verifier_revision_id: optimizerId, modality, task_type: taskType, reward_mapping: { normalization: { minimum: Number(minimum), maximum: Number(maximum), direction: "maximize" }, minimum: 0, maximum: 1, threshold: Number(threshold), failure_behavior: failureBehavior === "fail_closed" ? "reject" : failureBehavior === "error" ? "raise" : "abstain", filtering: { mode: "optimizer_only", threshold: Number(threshold) }, scaling: 1, centering: 0, keep_policy: "trainer_declared" }, input_mapping: { prompt: "prompt", output: "output", expected: "expected" }, auditors: [{ role: "primary_sentinel", ordinal: 0, verifier_revision_id: sentinelId }] } };
  return <div className="mx-auto max-w-5xl"><section className="border-b border-border pb-5"><div className="text-[9px] uppercase tracking-[0.14em] text-accent">GUIDED REWARD SYSTEM</div><h2 className="mt-1 text-lg font-semibold text-fg">Pin optimization and independent evidence</h2><p className="mt-1 text-[10px] text-fg-subtle">The sentinel scores captured outputs only; it never changes training reward.</p><div className="mt-5 grid grid-cols-4 gap-px overflow-hidden rounded-md border border-border bg-border">{steps.map((label, index) => <button key={label} type="button" onClick={() => index <= step && setStep(index)} className={cn("min-h-11 bg-bg px-3 text-left text-[9px]", index === step ? "text-accent" : index < step ? "text-fg" : "text-fg-disabled")}><span className="mr-2 font-mono">{index < step ? "✓" : index + 1}</span><span className="hidden sm:inline">{label}</span></button>)}</div></section>
    <div className="min-h-[340px] py-6">{step === 0 ? <div className="grid gap-5 md:grid-cols-2"><div className="space-y-4"><AuditField label="Reward system name"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Independent answer reward" /></AuditField><AuditField label="Purpose"><Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Tracks optimizer/sentinel agreement during RAFT" /></AuditField></div><div><AuditField label="Qualified training verifier"><SearchPicker value={optimizerId} onChange={(value) => { setOptimizerId(value); setSentinelId(""); const revision = verifierRevision(verifiers, value); if (revision) { setModality(revision.modality); setTaskType(revision.task_type); } }} options={verifierOptions} placeholder="Choose optimizer verifier revision" emptyLabel="No pass-qualified verifier revision" /></AuditField><p className="mt-2 text-[8.5px] leading-4 text-fg-disabled">This verifier supplies filtering and training reward under its immutable V7 contract.</p></div></div> : null}
      {step === 1 ? <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_300px]"><div><AuditField label="Independent primary sentinel"><SearchPicker value={sentinelId} onChange={setSentinelId} options={sentinelOptions} placeholder="Choose disjoint qualified sentinel" emptyLabel="No other compatible qualified revision" /></AuditField>{correlated ? <div className="mt-3 flex gap-2 border-l-2 border-warning bg-warning/5 px-3 py-2 text-[9px] leading-4 text-warning"><AlertTriangle className="h-3.5 w-3.5 shrink-0" />The selected verifier shares an implementation, artifact, or chain leaf with the optimizer. It can inspect but cannot gate.</div> : sentinel ? <div className="mt-3 flex gap-2 border-l-2 border-success bg-success/5 px-3 py-2 text-[9px] leading-4 text-fg-subtle"><CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success" />No shared declared fingerprints were found.</div> : null}</div><dl className="divide-y divide-border-subtle border-y border-border-subtle"><DetailRow label="Modality" value={modality} /><DetailRow label="Task" value={taskType} /><DetailRow label="Optimizer" value={optimizerId || "not selected"} mono /><DetailRow label="Sentinel" value={sentinelId || "not selected"} mono /></dl></div> : null}
      {step === 2 ? <div className="grid gap-5 md:grid-cols-2"><div className="grid grid-cols-3 gap-3"><AuditField label="Verifier minimum"><Input value={minimum} onChange={(event) => setMinimum(event.target.value)} type="number" /></AuditField><AuditField label="Verifier maximum"><Input value={maximum} onChange={(event) => setMaximum(event.target.value)} type="number" /></AuditField><AuditField label="Mapped threshold (0–1)"><Input value={threshold} onChange={(event) => setThreshold(event.target.value)} type="number" min="0" max="1" step="0.01" /></AuditField></div><div><AuditField label="Verifier error behavior"><select value={failureBehavior} onChange={(event) => setFailureBehavior(event.target.value)} className="h-9 w-full rounded-md border border-border bg-bg px-2 text-[10px] text-fg outline-none focus:border-accent"><option value="fail_closed">Fail closed</option><option value="error">Stop with error</option><option value="abstain">Record abstention</option></select></AuditField><dl className="mt-4 divide-y divide-border-subtle border-y border-border-subtle"><DetailRow label="Normalization" value="Linear to 0 → 1" /><DetailRow label="Direction" value="Higher is better" /><DetailRow label="Sentinel use" value="Evidence only" /><DetailRow label="Automatic tuning" value="Never" /></dl></div></div> : null}
      {step === 3 ? <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_300px]"><div><SectionTitle title={name || "Untitled reward system"} detail={description || "No purpose supplied."} /><dl className="mt-4 divide-y divide-border-subtle border-y border-border-subtle"><DetailRow label="Training verifier" value={optimizerId} mono /><DetailRow label="Primary sentinel" value={sentinelId} mono /><DetailRow label="Contract" value={`${modality} · ${taskType} · raw ${minimum} → ${maximum} · mapped threshold ${threshold}`} /><DetailRow label="Errors" value={humanize(failureBehavior)} /></dl></div><aside className="border-l border-border-subtle pl-5"><InspectorTitle>Publication checks</InspectorTitle><Readiness label="Optimizer qualified" ready={Boolean(optimizer)} /><Readiness label="Sentinel qualified" ready={Boolean(sentinel)} /><Readiness label="Declared fingerprints disjoint" ready={!correlated && Boolean(sentinel)} /><Readiness label="Reward mapping valid" ready={Number(maximum) > Number(minimum)} /><p className="mt-3 text-[8.5px] leading-4 text-fg-disabled">Any later change creates a new immutable revision.</p></aside></div> : null}</div>
    <div className="flex items-center justify-between border-t border-border pt-4"><Button size="sm" variant="ghost" onClick={step ? () => setStep(step - 1) : onCancel}>{step ? "Back" : "Cancel"}</Button><div className="flex items-center gap-3">{error ? <span role="alert" className="text-[9px] text-danger">{error}</span> : null}{step < steps.length - 1 ? <Button size="sm" onClick={() => setStep(step + 1)} disabled={!stepReady}>Continue <ArrowRight /></Button> : <Button size="sm" variant="primary" onClick={() => onCreate(payload)} disabled={!stepReady || pending}>{pending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Publish immutable system</Button>}</div></div>
  </div>;
}

function RewardSystemRow({ item, selected, onSelect }: { item: RewardSystem; selected: boolean; onSelect: () => void }) {
  const revision = item.latest_revision;
  const published = Boolean(revision || item.latest_revision_id);
  return <button type="button" onClick={onSelect} className={cn("w-full px-4 py-3 text-left transition-colors hover:bg-surface/50", selected && "bg-accent/7")}><div className="flex items-center justify-between gap-2"><span className={cn("truncate text-[11px] font-medium", selected ? "text-accent" : "text-fg")}>{item.name}</span><StatusBadge value={revision?.qualification_state || (published ? "published" : "draft")} /></div><div className="mt-1 truncate text-[9px] text-fg-disabled">{revision ? `${revision.modality} · ${revision.task_type} · r${revision.revision_number}` : published ? "Open to inspect the latest immutable revision" : "No published revision"}</div></button>;
}

function RewardSystemDetail({ item, protocols, profiles }: { item: RewardSystem; protocols: import("@/lib/api").RewardAuditProtocolRevision[]; profiles: RewardIntegrityProfileRevision[] }) {
  const revision = item.latest_revision;
  if (!revision) return <Empty label="This system has no published immutable revision." />;
  const primary = revision.auditors.find((auditor) => auditor.role === "primary_sentinel");
  return <div className="mx-auto max-w-5xl"><section className="border-b border-border pb-6"><div className="flex flex-wrap items-center gap-2"><span className="text-[9px] uppercase tracking-[0.14em] text-accent">REWARD SYSTEM REVISION</span><StatusBadge value={revision.qualification_state || "published"} /></div><h2 className="mt-2 text-lg font-semibold text-fg">{item.name}</h2><p className="mt-1 max-w-2xl text-[10px] leading-4 text-fg-subtle">{item.description || "Pinned optimizer, independent sentinel, normalization, and reward behavior."}</p></section>
    <section className="grid gap-7 py-6 lg:grid-cols-[minmax(0,1fr)_300px]"><div><SectionTitle title="Optimizer and independent sentinel" detail="The sentinel sees the same captured output but never contributes to gradients or filtering." /><dl className="mt-4 divide-y divide-border-subtle border-y border-border-subtle"><DetailRow label="Optimizer verifier" value={revision.optimizer_verifier_revision_id || revision.optimizer_verifier_profile_revision_id || "not bound"} mono /><DetailRow label="Primary sentinel" value={primary?.verifier_revision_id || primary?.verifier_profile_revision_id || "not bound"} mono /><DetailRow label="Sentinel independence" value={primary?.correlated ? `Correlated · ${primary.correlation_reasons?.join("; ") || "inspection only"}` : "Disjoint implementation, artifact, and chain leaves"} /><DetailRow label="Reward range" value={formatUnknown(revision.reward_mapping ?? revision.reward_normalization) || "Normalized to 0 → 1, higher is better"} /><DetailRow label="Shaping" value={formatUnknown(revision.shaping ?? revision.definition?.shaping) || "No additional shaping declared"} /></dl></div><aside className="border-l border-border-subtle pl-5"><SectionTitle title="Guided defaults" detail="Normal launch pickers prefer promotable policies and same-output capture." /><dl className="mt-3 divide-y divide-border-subtle border-y border-border-subtle"><DetailRow label="Capture" value={protocols.find((value) => value.template === "balanced_256")?.name || "balanced_256"} /><DetailRow label="Integrity" value={profiles.find((value) => value.template === "human_aligned_integrity")?.name || "human aligned"} /><DetailRow label="Failure" value="Pause for operator review" /><DetailRow label="Automatic changes" value="None" /></dl></aside></section>
  </div>;
}

function RewardAuditResultsView({ runId, selectedAuditId, selectedSampleId, page = 1, classification, onAudit, onSample, onPage, onClassification }: { runId?: string; selectedAuditId?: string; selectedSampleId?: string; page?: number; classification?: string; onAudit: (id?: string) => void; onSample: (id?: string) => void; onPage: (page: number) => void; onClassification: (classification?: string) => void }) {
  const audits = useQuery({
    queryKey: ["reward-integrity-audits", runId || "all"],
    queryFn: () => runId ? api.runRewardIntegrityAudits(runId, { limit: 200 }) : api.listRewardIntegrityAudits({ limit: 200 }),
    refetchInterval: (query) => query.state.data?.items.some((item) => ["queued", "running"].includes(item.status)) ? 3_000 : false,
    retry: false,
  });
  useEffect(() => { if (!selectedAuditId && audits.data?.items[0]) onAudit(audits.data.items[0].id); }, [audits.data?.items, onAudit, selectedAuditId]);
  useEffect(() => {
    const items = audits.data?.items ?? [];
    if (items.length < 2) return;
    const navigate = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) return;
      if (document.querySelector("[data-mobile-evidence-review]")) return;
      const previous = event.key === "[" || (event.altKey && event.key === "ArrowLeft");
      const next = event.key === "]" || (event.altKey && event.key === "ArrowRight");
      if (!previous && !next) return;
      event.preventDefault();
      const current = Math.max(0, items.findIndex((item) => item.id === (selectedAuditId || items[0]?.id)));
      const index = Math.max(0, Math.min(items.length - 1, current + (next ? 1 : -1)));
      onAudit(items[index]?.id);
      onSample(undefined);
      onPage(1);
    };
    window.addEventListener("keydown", navigate);
    return () => window.removeEventListener("keydown", navigate);
  }, [audits.data?.items, onAudit, onPage, onSample, selectedAuditId]);
  const id = selectedAuditId || audits.data?.items[0]?.id || "";
  const detail = useQuery({ queryKey: ["reward-integrity-audit", id], queryFn: () => api.rewardIntegrityAudit(id), enabled: Boolean(id), refetchInterval: (query) => ["queued", "running"].includes(query.state.data?.status ?? "") ? 3_000 : false, retry: false });
  const metrics = useQuery({ queryKey: ["reward-integrity-audit", id, "metrics"], queryFn: () => api.rewardIntegrityAuditMetrics(id), enabled: Boolean(id) && detail.data?.status === "completed", retry: false });
  const offset = Math.max(0, page - 1) * 50;
  const samples = useQuery({ queryKey: ["reward-integrity-audit", id, "samples", page, classification], queryFn: () => api.rewardIntegrityAuditSamples(id, { limit: 50, offset, classification }), enabled: Boolean(id) && detail.data?.status === "completed", retry: false });
  const selected = samples.data?.items.find((item) => item.id === selectedSampleId) ?? samples.data?.items[0];
  const selectedIndex = selected ? (samples.data?.items ?? []).findIndex((item) => item.id === selected.id) : -1;
  return (
    <div className="grid min-h-[560px] xl:grid-cols-[270px_minmax(0,1fr)_310px]">
      <aside className="border-b border-border bg-bg-subtle/25 xl:border-b-0 xl:border-r"><RailHeader eyebrow={runId ? "RUN AUDITS" : "TRAINING AUDITS"} title={`${audits.data?.total ?? audits.data?.items.length ?? 0} boundary results`} /><div className="border-b border-border-subtle px-4 py-1.5 font-mono text-[8px] text-fg-disabled">[ ] or ⌥←/→ · change boundary</div>{audits.isLoading ? <Loading label="Loading audits" /> : audits.isError ? <Unavailable label={runId ? "This legacy run has no V8 audit endpoint." : "Training audits are unavailable."} /> : <div className="divide-y divide-border-subtle">{audits.data?.items.map((item) => <AuditRow key={item.id} item={item} selected={item.id === id} onSelect={() => onAudit(item.id)} />)}{!audits.data?.items.length ? <Empty label={runId ? "No training-signal evidence was recorded for this run." : "Audited verifier-guided runs will appear here."} /> : null}</div>}</aside>
      <main className="min-w-0 border-b border-border xl:border-b-0 xl:border-r">{detail.isLoading ? <Loading label="Opening audit evidence" /> : detail.isError ? <Unavailable label="This audit could not be opened." /> : detail.data ? <AuditEvidence audit={detail.data} metrics={metrics.data?.items ?? detail.data.metrics ?? []} samples={samples.data?.items ?? []} total={samples.data?.total ?? 0} page={page} classification={classification} selectedId={selected?.id} onClassification={onClassification} onSample={onSample} onPage={onPage} /> : <Empty label="Select a boundary audit." />}</main>
      <aside className="min-w-0 bg-bg-subtle/15 p-4 lg:p-5">{selected ? <ObservationInspector item={selected} /> : detail.data ? <AuditDecisionInspector audit={detail.data} /> : <Empty label="Select an audit or evidence record." />}</aside>
      {selectedSampleId && selected ? <MobileEvidenceReview title="Training audit evidence" position={offset + selectedIndex + 1} total={samples.data?.total ?? 0} onBack={() => onSample(undefined)} onPrevious={selectedIndex > 0 ? () => onSample(samples.data?.items[selectedIndex - 1]?.id) : undefined} onNext={selectedIndex >= 0 && selectedIndex < (samples.data?.items.length ?? 0) - 1 ? () => onSample(samples.data?.items[selectedIndex + 1]?.id) : undefined}><ObservationInspector item={selected} /></MobileEvidenceReview> : null}
    </div>
  );
}

function AuditEvidence({ audit, metrics, samples, total, page, classification, selectedId, onClassification, onSample, onPage }: { audit: RewardIntegrityAudit; metrics: RewardIntegrityMetric[]; samples: RewardIntegrityObservation[]; total: number; page: number; classification?: string; selectedId?: string; onClassification: (value?: string) => void; onSample: (id?: string) => void; onPage: (page: number) => void }) {
  const pairedCoverage = metricValue(metrics, "paired_coverage") ?? metricValue(metrics, "coverage");
  return <div><section className="border-b border-border px-5 py-5"><div className="flex flex-wrap items-start justify-between gap-3"><div><div className="flex items-center gap-2"><span className="text-[9px] uppercase tracking-[0.14em] text-accent">{auditBoundaryLabel(audit)}</span><StatusBadge value={audit.decision?.decision || audit.status} /></div><h2 className="mt-2 text-[15px] font-semibold text-fg">Same-output reward integrity</h2><p className="mt-1 text-[9.5px] text-fg-subtle">Exact captured outputs · {audit.capture_fidelity?.replaceAll("_", " ") || "capture fidelity unavailable"}</p></div>{["queued", "running"].includes(audit.status) ? <span className="font-mono text-[10px] text-accent">{audit.progress_percent?.toFixed(0) ?? 0}%</span> : null}</div></section>
    <section className="grid gap-px border-b border-border bg-border sm:grid-cols-4"><MetricCell label="Paired coverage" value={percent(pairedCoverage)} /><MetricCell label="Pass agreement" value={percent(metricValue(metrics, "pass_agreement"))} /><MetricCell label="Optimizer-only" value={percent(metricValue(metrics, "optimizer_only_acceptance"))} inverse /><MetricCell label="Reward gap" value={number(metricValue(metrics, "absolute_mean_reward_gap") ?? metricValue(metrics, "normalized_mean_gap") ?? metricValue(metrics, "reward_gap"))} inverse /></section>
    <BoundaryTrend metrics={metrics} />
    <section><div className="flex flex-wrap items-end justify-between gap-3 border-b border-border-subtle px-5 py-3"><div><div className="text-[9px] uppercase tracking-[0.12em] text-fg-disabled">PAIRED EVIDENCE</div><div className="mt-0.5 text-[10px] text-fg-muted">{total} retained outputs · stable-record unit</div></div><select value={classification || ""} onChange={(event) => { onClassification(event.target.value || undefined); onPage(1); }} className="h-8 rounded-md border border-border bg-bg px-2 text-[10px] text-fg outline-none focus:border-accent" aria-label="Filter audit evidence"><option value="">All outcomes</option><option value="optimizer_only_accept">Optimizer-only acceptance</option><option value="sentinel_only_accept">Sentinel-only acceptance</option><option value="agreement">Agreement</option><option value="error">Parser or runtime error</option></select></div><ObservationTable items={samples} selectedId={selectedId} onSelect={onSample} /><Pager total={total} page={page} pageSize={50} onPage={onPage} /></section>
  </div>;
}

function BoundaryTrend({ metrics }: { metrics: RewardIntegrityMetric[] }) {
  const visible = ["paired_coverage", "pass_agreement", "optimizer_only_acceptance", "spearman", "top_tail_disagreement"].map((name) => metrics.find((metric) => metric.name === name || metric.name.endsWith(`.${name}`))).filter(Boolean) as RewardIntegrityMetric[];
  if (!visible.length) return null;
  return <section className="border-b border-border-subtle px-5 py-4"><div className="mb-3 flex items-center justify-between"><div className="text-[9px] uppercase tracking-[0.12em] text-fg-disabled">INTEGRITY PROFILE</div><span className="text-[8.5px] text-fg-disabled">95% grouped intervals where available</span></div><div className="space-y-2" aria-label="Reward integrity metrics">{visible.map((metric) => <div key={metric.name} className="grid grid-cols-[140px_minmax(0,1fr)_54px] items-center gap-3 text-[9px]"><span className="truncate text-fg-subtle">{humanize(metric.name)}</span><div className="h-1.5 overflow-hidden bg-surface"><span className={cn("block h-full", lowerIsBetter(metric.name) ? "bg-warning" : "bg-accent")} style={{ width: `${Math.max(0, Math.min(100, (metric.value ?? 0) * 100))}%` }} /></div><span className="text-right font-mono text-fg-muted">{number(metric.value)}</span></div>)}</div><details className="mt-3"><summary className="cursor-pointer text-[8.5px] uppercase tracking-wider text-accent">Table equivalent</summary><table className="mt-2 w-full text-left text-[9px]"><thead><tr className="border-b border-border-subtle"><Th>Metric</Th><Th>Value</Th><Th>95% interval</Th><Th>Records</Th></tr></thead><tbody>{visible.map((metric) => <tr key={metric.name} className="border-b border-border-subtle"><Td>{humanize(metric.name)}</Td><Td mono>{number(metric.value)}</Td><Td mono>{metric.lower_ci == null || metric.upper_ci == null ? "Unavailable" : `${number(metric.lower_ci)}–${number(metric.upper_ci)}`}</Td><Td mono>{String(metric.record_count ?? "—")}</Td></tr>)}</tbody></table></details></section>;
}

function ObservationTable({ items, selectedId, onSelect }: { items: RewardIntegrityObservation[]; selectedId?: string; onSelect: (id?: string) => void }) {
  if (!items.length) return <Empty label="No paired evidence matches this filter." />;
  return <div className="overflow-x-auto"><table className="w-full min-w-[700px] text-left text-[9.5px]"><thead><tr className="border-b border-border-subtle"><Th>Outcome</Th><Th>Record</Th><Th>Optimizer</Th><Th>Sentinel</Th><Th>Gap</Th><Th>Stratum</Th></tr></thead><tbody>{items.map((item) => {
    const selected = selectedId === item.id;
    const select = () => onSelect(item.id);
    return <tr
      key={item.id}
      role="button"
      tabIndex={0}
      aria-label={`Open paired evidence for record ${item.record.record_id}`}
      aria-pressed={selected}
      onClick={select}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        select();
      }}
      className={cn(
        "cursor-pointer border-b border-border-subtle transition-colors hover:bg-surface/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent",
        selected && "bg-accent/7",
      )}
    ><Td><StatusBadge value={item.classification || "unclassified"} /></Td><Td mono>{short(item.record.record_id)}</Td><Td mono>{number(item.normalized_optimizer_reward)}</Td><Td mono>{number(item.normalized_sentinel_reward)}</Td><Td mono>{signed(item.reward_gap)}</Td><Td>{humanize(item.capture_stratum || "core")}</Td></tr>;
  })}</tbody></table></div>;
}

function ObservationInspector({ item, eyebrow = "PAIRED OUTPUT" }: { item: RewardIntegrityObservation; eyebrow?: string }) {
  const toolTrace = toolTraceFrom(item.prompt ?? item.context);
  return <div><div className="text-[9px] uppercase tracking-[0.13em] text-accent">{eyebrow}</div><div className="mt-2 flex flex-wrap items-center gap-2"><StatusBadge value={item.classification || "unclassified"} /><span className="font-mono text-[8px] text-fg-disabled">{item.record.record_id}</span></div><InspectorBlock label="Input" value={item.prompt ?? item.context} />{toolTrace ? <div className="grid gap-3 sm:grid-cols-2"><InspectorBlock label="Tool definitions" value={toolTrace.tools} /><InspectorBlock label="Tool calls / results" value={toolTrace.calls} /></div> : null}<InspectorBlock label="Exact captured output" value={item.output} />{item.expected != null ? <InspectorBlock label="Expected" value={item.expected} /> : null}{item.media?.length ? <MediaEvidence items={item.media} /> : null}<div className="mt-4 grid grid-cols-2 gap-px bg-border-subtle"><InspectorMetric label="Optimizer" value={number(item.normalized_optimizer_reward)} /><InspectorMetric label="Sentinel" value={number(item.normalized_sentinel_reward)} /><InspectorMetric label="Reward gap" value={signed(item.reward_gap)} /><InspectorMetric label="Candidate" value={String(item.candidate_ordinal ?? 0)} /></div><details className="mt-4"><summary className="cursor-pointer text-[8.5px] uppercase tracking-wider text-accent">Component traces</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap border border-border bg-bg p-2 font-mono text-[8px] leading-4 text-fg-subtle">{JSON.stringify({ optimizer: item.optimizer_observation, sentinel: item.sentinel_observation, diagnostics: item.diagnostic_observations }, null, 2)}</pre></details></div>;
}

function MediaEvidence({ items }: { items: NonNullable<RewardIntegrityObservation["media"]> }) {
  return <div className="mt-4"><InspectorTitle>Media evidence</InspectorTitle><div className="space-y-3">{items.map((media) => { const kind = media.kind.toLowerCase(); const metadata = media.metadata ?? {}; const source = renderableMediaSource(media.path); if (kind.includes("image")) return <figure key={`${media.kind}-${media.hash}`} className="overflow-hidden border border-border bg-bg">{source ? <img src={source} alt="Captured training evidence" className="max-h-72 w-full object-contain" loading="lazy" /> : <div className="grid h-28 place-items-center text-fg-disabled"><ImageIcon className="h-5 w-5" /><span className="sr-only">Image reference unavailable</span></div>}<figcaption className="border-t border-border-subtle px-2 py-2 font-mono text-[8px] text-fg-disabled"><div>{String(metadata.width ?? "?")} × {String(metadata.height ?? "?")} · {String(metadata.mime_type ?? media.kind)}</div><div className="mt-1 break-all">sha256 {media.hash}</div></figcaption></figure>; if (kind.includes("audio")) return <figure key={`${media.kind}-${media.hash}`} className="border border-border bg-bg p-3"><div className="flex items-center gap-2 text-[9px] text-fg-subtle"><Music2 className="h-3.5 w-3.5 text-accent" />Captured audio</div>{source ? <audio controls preload="metadata" src={source} className="mt-2 w-full">Your browser cannot play this audio evidence.</audio> : <div className="mt-2 border border-dashed border-border px-2 py-4 text-center text-[8.5px] text-fg-disabled">Audio reference unavailable</div>}<figcaption className="mt-2 font-mono text-[8px] text-fg-disabled">{String(metadata.duration_seconds ?? metadata.duration ?? "?")} sec · {String(metadata.sample_rate ?? "?")} Hz · sha256 {media.hash}</figcaption></figure>; return <div key={`${media.kind}-${media.hash}`} className="border-l-2 border-border pl-2 text-[8.5px]"><div className="text-fg-subtle">{media.kind}</div><div className="break-all font-mono text-fg-disabled">sha256 {media.hash}</div>{Object.keys(metadata).length ? <pre className="mt-1 overflow-auto whitespace-pre-wrap font-mono text-[8px] text-fg-disabled">{JSON.stringify(metadata, null, 2)}</pre> : null}</div>; })}</div></div>;
}

function renderableMediaSource(value?: string | null): string | null {
  if (!value) return null;
  return /^(https?:|data:|blob:)/.test(value) || value.startsWith("/dataset-") ? value : null;
}

function toolTraceFrom(value: unknown): { tools: unknown; calls: unknown } | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const tools = record.tools ?? record.tool_definitions;
  const calls = record.tool_calls ?? record.expected_calls ?? record.results ?? record.tool_results;
  return tools !== undefined || calls !== undefined ? { tools, calls } : null;
}

function isTypingTarget(value: EventTarget | null): boolean {
  return value instanceof HTMLElement && (value.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(value.tagName));
}

function AuditDecisionInspector({ audit }: { audit: RewardIntegrityAudit }) {
  const queryClient = useQueryClient();
  const [reason, setReason] = useState("");
  const [retryReason, setRetryReason] = useState("");
  const review = useMutation({
    mutationFn: (action: "continue" | "stop" | "fork") => api.reviewRewardIntegrityAudit(audit.id, { action, reason: reason.trim() }),
    onSuccess: (result, action) => {
      setReason("");
      queryClient.invalidateQueries({ queryKey: ["reward-integrity-audit", audit.id] });
      queryClient.invalidateQueries({ queryKey: ["reward-integrity-audits"] });
      if (action === "fork" && "href" in result && result.href) window.location.assign(result.href);
    },
  });
  const retry = useMutation({
    mutationFn: () => api.retryRewardIntegrityAudit(audit.id, retryReason.trim()),
    onSuccess: () => {
      setRetryReason("");
      queryClient.invalidateQueries({ queryKey: ["reward-integrity-audit", audit.id] });
      queryClient.invalidateQueries({ queryKey: ["reward-integrity-audits"] });
      queryClient.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  const requiresReview = audit.decision?.override !== true && (["fail", "incomplete_evidence"].includes(audit.decision?.decision ?? "") || ["pause", "awaiting_review"].includes(audit.decision?.action ?? ""));
  const retryable = ["failed", "interrupted", "cancelled", "needs_reconciliation"].includes(audit.status);
  return <div>
    <div className="text-[9px] uppercase tracking-[0.13em] text-accent">AUDIT DECISION</div>
    <div className="mt-2"><StatusBadge value={audit.decision?.decision || audit.status} /></div>
    <ul className="mt-3 space-y-2">{audit.decision?.reasons.map((item) => <li key={item} className="flex gap-2 text-[9px] leading-4 text-fg-subtle"><span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-fg-disabled" />{item}</li>)}</ul>
    {audit.error ? <p className="mt-3 border-l-2 border-danger pl-2 text-[9px] leading-4 text-danger">{audit.error}</p> : null}
    {requiresReview ? <div className="mt-5 border-t border-border-subtle pt-4"><InspectorTitle>Operator review</InspectorTitle><Input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Required decision reason" /><div className="mt-2 grid grid-cols-3 gap-1"><Button size="sm" variant="secondary" disabled={!reason.trim() || review.isPending} onClick={() => review.mutate("continue")}><Play />Continue</Button><Button size="sm" variant="ghost" disabled={!reason.trim() || review.isPending} onClick={() => review.mutate("stop")}><Square />Stop</Button><Button size="sm" variant="ghost" disabled={!reason.trim() || review.isPending} onClick={() => review.mutate("fork")}><GitCompareArrows />Fork</Button></div>{review.error ? <p role="alert" className="mt-2 text-[9px] text-danger">{review.error.message}</p> : null}<Link to="/datasets/review" search={{ new: "1", source: "reward_integrity_audit", sourceRef: audit.id, baseRef: undefined }} className="mt-3 inline-flex text-[9.5px] text-accent hover:underline">Create reviewed proposal</Link><p className="mt-2 text-[8.5px] leading-4 text-fg-disabled">A proposal does not resolve this pause or start training.</p></div> : null}
    {retryable ? <div className="mt-5 border-t border-border-subtle pt-4"><InspectorTitle>Retry audit</InspectorTitle><p className="mb-2 text-[8.5px] leading-4 text-fg-disabled">A forced retry resumes sentinel scoring in a fresh attempt and records your reason.</p><Input value={retryReason} onChange={(event) => setRetryReason(event.target.value)} placeholder="Required retry reason" aria-label="Required retry reason" /><Button size="sm" variant="secondary" className="mt-2" disabled={!retryReason.trim() || retry.isPending} onClick={() => retry.mutate()}>{retry.isPending ? <Loader2 className="animate-spin" /> : <RotateCcw />}Retry audit</Button>{retry.error ? <p role="alert" className="mt-2 text-[9px] text-danger">{retry.error.message}</p> : null}</div> : null}
  </div>;
}

function RewardAuditCompareView({ baseAuditId, candidateAuditId, selectedPairId, page = 1, onCompare, onPair, onPage }: { baseAuditId?: string; candidateAuditId?: string; selectedPairId?: string; page?: number; onCompare: (base?: string, candidate?: string) => void; onPair: (id?: string) => void; onPage: (page: number) => void }) {
  const audits = useQuery({ queryKey: ["reward-integrity-audits", "compare"], queryFn: () => api.listRewardIntegrityAudits({ status: "completed", limit: 200 }), retry: false });
  const options = (audits.data?.items ?? []).map((item) => ({ value: item.id, label: `${item.run_id} · ${auditBoundaryLabel(item).toLowerCase()}`, description: `${item.capture_fidelity?.replaceAll("_", " ") || "capture unknown"} · ${item.decision?.decision || item.status}` }));
  const pageSize = 25;
  const offset = Math.max(0, page - 1) * pageSize;
  const comparison = useQuery({ queryKey: ["reward-integrity-audits", "compare", baseAuditId, candidateAuditId, page], queryFn: () => api.compareRewardIntegrityAudits(baseAuditId!, candidateAuditId!, { limit: pageSize, offset }), enabled: Boolean(baseAuditId && candidateAuditId && baseAuditId !== candidateAuditId), retry: false });
  return <div className="mx-auto max-w-[1500px] p-5 lg:p-8"><section className="border-b border-border pb-6"><SectionTitle title="Compare reward integrity evidence" detail="Exact pairs share a snapshot identity. Stable-input matches are distributional and never presented as causal evidence." /><div className="mt-5 grid items-end gap-3 md:grid-cols-[minmax(0,1fr)_24px_minmax(0,1fr)]"><AuditField label="Base audit"><SearchPicker value={baseAuditId || ""} onChange={(value) => onCompare(value || undefined, candidateAuditId)} options={options} placeholder="Choose base boundary" /></AuditField><GitCompareArrows className="mb-2 h-4 w-4 text-fg-disabled" /><AuditField label="Candidate audit"><SearchPicker value={candidateAuditId || ""} onChange={(value) => onCompare(baseAuditId, value || undefined)} options={options} placeholder="Choose candidate boundary" /></AuditField></div></section>{comparison.isLoading ? <Loading label="Joining bounded evidence pairs" /> : comparison.isError ? <Unavailable label="These audits could not be compared." /> : comparison.data ? <ComparisonEvidence data={comparison.data} selectedPairId={selectedPairId} page={page} onPair={onPair} onPage={onPage} /> : <Empty label="Choose two completed audits to compare reward-quality divergence." />}</div>;
}

function ComparisonEvidence({ data, selectedPairId, page, onPair, onPage }: { data: RewardIntegrityComparison; selectedPairId?: string; page: number; onPair: (id?: string) => void; onPage: (page: number) => void }) {
  const selected = data.pairs.find((item) => item.id === selectedPairId) ?? data.pairs[0];
  const selectedIndex = selected ? data.pairs.findIndex((item) => item.id === selected.id) : -1;
  const exact = data.comparison_kind === "paired_snapshot";
  return <section className="py-6">{!data.compatible ? <div className="mb-4 flex gap-2 border-l-2 border-danger bg-danger/5 px-3 py-2 text-[9.5px] text-fg-subtle"><AlertTriangle className="h-3.5 w-3.5 shrink-0 text-danger" />{data.compatibility_reasons?.join(" ") || "The immutable audit contracts do not match."}</div> : null}<div className={cn("mb-5 flex gap-2 border-l-2 px-3 py-2.5 text-[9.5px] leading-4", exact ? "border-success bg-success/5 text-fg-subtle" : "border-warning bg-warning/5 text-fg-subtle")}><div className="shrink-0 pt-0.5">{exact ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <AlertTriangle className="h-3.5 w-3.5 text-warning" />}</div><div><div className="flex flex-wrap items-center gap-2"><StatusBadge value={data.comparison_kind || "aggregate_only"} /><span>{data.pairing_reason}</span></div><div className="mt-1 font-mono text-[8px] text-fg-disabled">{data.pair_total} pairs · {data.unmatched_base} unmatched base · {data.unmatched_candidate} unmatched candidate</div></div></div>
    <div className="overflow-x-auto border-y border-border-subtle"><table className="w-full min-w-[680px] text-left text-[9.5px]"><thead><tr className="border-b border-border-subtle"><Th>Metric</Th><Th>Base</Th><Th>Candidate</Th><Th>Raw delta</Th><Th>Direction-aware</Th></tr></thead><tbody>{data.metrics.map((metric) => <tr key={metric.name} className="border-b border-border-subtle last:border-0"><Td>{humanize(metric.name)}</Td><Td mono>{number(metric.base_value)}</Td><Td mono>{number(metric.candidate_value)}</Td><Td mono>{signed(metric.raw_delta)}</Td><Td mono className={cn((metric.favorable_delta ?? 0) > 0 && "text-success", (metric.favorable_delta ?? 0) < 0 && "text-danger")}>{signed(metric.favorable_delta)}</Td></tr>)}</tbody></table></div>
    {data.comparison_kind === "aggregate_only" ? <Empty label={data.pairing_reason} /> : <div className="mt-6 grid border-y border-border-subtle xl:grid-cols-[minmax(0,1fr)_390px]"><div className="min-w-0 xl:border-r xl:border-border-subtle"><div className="flex items-end justify-between gap-3 border-b border-border-subtle px-4 py-3"><div><div className="text-[9px] uppercase tracking-[0.12em] text-fg-disabled">BOUNDED EVIDENCE PAIRS</div><div className="mt-0.5 text-[9px] text-fg-muted">{exact ? "Exact snapshot joins" : "Stable-record joins · non-causal"}</div></div><span className="font-mono text-[8px] text-fg-disabled">limit {data.limit}</span></div><ComparisonPairTable items={data.pairs} selectedId={selected?.id} onSelect={onPair} /><Pager total={data.pair_total} page={page} pageSize={data.limit} onPage={onPage} /></div><aside className="hidden min-w-0 bg-bg-subtle/15 p-5 xl:block">{selected ? <ComparisonPairInspector pair={selected} /> : <Empty label="Select an evidence pair." />}</aside></div>}
    {selectedPairId && selected ? <MobileEvidenceReview title={exact ? "Exact snapshot pair" : "Matched input · non-causal"} position={data.offset + selectedIndex + 1} total={data.pair_total} onBack={() => onPair(undefined)} onPrevious={selectedIndex > 0 ? () => onPair(data.pairs[selectedIndex - 1]?.id) : undefined} onNext={selectedIndex >= 0 && selectedIndex < data.pairs.length - 1 ? () => onPair(data.pairs[selectedIndex + 1]?.id) : undefined}><ComparisonPairInspector pair={selected} /></MobileEvidenceReview> : null}
  </section>;
}

function ComparisonPairTable({ items, selectedId, onSelect }: { items: RewardIntegrityComparisonPair[]; selectedId?: string; onSelect: (id?: string) => void }) {
  if (!items.length) return <Empty label="No evidence pairs exist on this page." />;
  return <div className="overflow-x-auto"><table className="w-full min-w-[820px] text-left text-[9.5px]"><thead><tr className="border-b border-border-subtle"><Th>Record</Th><Th>Base opt / sent</Th><Th>Candidate opt / sent</Th><Th>Output identity</Th><Th>Inspect</Th></tr></thead><tbody>{items.map((item) => <tr key={item.id} className={cn("cursor-pointer border-b border-border-subtle transition-colors hover:bg-surface/45 focus-within:bg-surface/45", selectedId === item.id && "bg-accent/7")}><Td mono>{short(item.record_id)}</Td><Td mono>{number(item.base.normalized_optimizer_reward)} / {number(item.base.normalized_sentinel_reward)}</Td><Td mono>{number(item.candidate.normalized_optimizer_reward)} / {number(item.candidate.normalized_sentinel_reward)}</Td><Td><StatusBadge value={item.pairing === "paired_snapshot" ? (item.same_output ? "exact" : "snapshot_drift") : "matched_input"} /></Td><Td><button type="button" onClick={() => onSelect(item.id)} aria-label={`Inspect comparison evidence for ${item.record_id}`} className="inline-flex h-8 items-center gap-1 px-2 text-[9px] text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">Open <ArrowRight className="h-3 w-3" /></button></Td></tr>)}</tbody></table></div>;
}

function ComparisonPairInspector({ pair }: { pair: RewardIntegrityComparisonPair }) {
  const exact = pair.pairing === "paired_snapshot";
  return <div><div className="mb-4 border-b border-border-subtle pb-3"><div className="flex flex-wrap items-center gap-2"><StatusBadge value={exact ? (pair.same_output ? "exact" : "snapshot_drift") : "matched_input"} /><span className="font-mono text-[8px] text-fg-disabled">{pair.record_id}</span></div><p className="mt-2 text-[8.5px] leading-4 text-fg-muted">{exact ? "Both audits reference the same immutable snapshot. Output and verifier evidence are shown side by side." : "Only record_id is shared. These on-policy outputs are distributional evidence and cannot establish a causal boundary effect."}</p></div><div className="grid gap-6 2xl:grid-cols-2"><section className="min-w-0"><ObservationInspector item={pair.base} eyebrow="BASE AUDIT" /></section><section className="min-w-0 border-t border-border pt-5 2xl:border-l 2xl:border-t-0 2xl:pl-5 2xl:pt-0"><ObservationInspector item={pair.candidate} eyebrow="CANDIDATE AUDIT" /></section></div></div>;
}

function MobileEvidenceReview({ title, position, total, onBack, onPrevious, onNext, children }: { title: string; position: number; total: number; onBack: () => void; onPrevious?: () => void; onNext?: () => void; children: ReactNode }) {
  useEffect(() => {
    const navigate = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) return;
      if (event.key === "Escape") { event.preventDefault(); onBack(); return; }
      if (event.key === "ArrowLeft" && onPrevious) { event.preventDefault(); onPrevious(); }
      if (event.key === "ArrowRight" && onNext) { event.preventDefault(); onNext(); }
    };
    window.addEventListener("keydown", navigate);
    return () => window.removeEventListener("keydown", navigate);
  }, [onBack, onNext, onPrevious]);
  return <div data-mobile-evidence-review role="dialog" aria-modal="true" aria-label={title} className="fixed inset-0 z-[70] flex flex-col bg-bg xl:hidden"><header className="flex h-11 shrink-0 items-center border-b border-border bg-bg px-1"><button type="button" onClick={onBack} className="inline-flex h-11 min-w-11 items-center gap-1 px-2 text-[10px] text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent" aria-label="Back to evidence list"><ArrowLeft className="h-4 w-4" /><span>Back</span></button><div className="min-w-0 flex-1 px-2 text-center"><div className="truncate text-[10px] font-medium text-fg">{title}</div><div className="font-mono text-[8px] text-fg-disabled">{position} of {total}</div></div><button type="button" onClick={onPrevious} disabled={!onPrevious} className="grid h-11 w-11 place-items-center text-fg disabled:text-fg-disabled focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent" aria-label="Previous evidence"><ChevronLeft className="h-4 w-4" /></button><button type="button" onClick={onNext} disabled={!onNext} className="grid h-11 w-11 place-items-center text-fg disabled:text-fg-disabled focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent" aria-label="Next evidence"><ChevronRight className="h-4 w-4" /></button></header><main className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4 pb-8 sm:p-5">{children}</main></div>;
}

function AuditRow({ item, selected, onSelect }: { item: RewardIntegrityAudit; selected: boolean; onSelect: () => void }) {
  return <button type="button" onClick={onSelect} className={cn("w-full px-4 py-3 text-left transition-colors hover:bg-surface/50", selected && "bg-accent/7")}><div className="flex items-start justify-between gap-2"><span className={cn("truncate text-[10.5px] font-medium", selected ? "text-accent" : "text-fg")}>{item.run_id}</span><StatusBadge value={item.decision?.decision || item.status} /></div><div className="mt-1 flex items-center gap-1.5 text-[8.5px] text-fg-disabled"><span>{auditBoundaryLabel(item)}</span><span>·</span><span>{item.capture_fidelity?.replaceAll("_", " ") || "capture unavailable"}</span></div>{["queued", "running"].includes(item.status) ? <div className="mt-2 h-0.5 overflow-hidden bg-surface"><span className="block h-full bg-accent" style={{ width: `${item.progress_percent ?? 0}%` }} /></div> : null}</button>;
}

function StatusBadge({ value }: { value: string }) { const normalized = value.toLowerCase(); const tone = ["pass", "completed", "exact", "paired_snapshot", "ready"].includes(normalized) ? "success" : ["fail", "failed", "optimizer_only_accept", "corrupt", "snapshot_drift"].includes(normalized) ? "danger" : ["warn", "incomplete_evidence", "awaiting_review", "running", "queued", "correlated", "aggregate_only", "matched_input"].includes(normalized) ? "warning" : "neutral"; return <Badge tone={tone} size="sm" dot>{humanize(value)}</Badge>; }
function StripValue({ label, value }: { label: string; value: string }) { return <span className="inline-flex items-center gap-1.5 text-[8.5px]"><span className="tracking-wider text-fg-disabled">{label}</span><span className="font-mono text-fg-subtle">{value}</span></span>; }
function RailHeader({ eyebrow, title, action }: { eyebrow: string; title: string; action?: ReactNode }) { return <div className="flex items-center justify-between gap-3 border-b border-border-subtle px-4 py-3"><div><div className="text-[8.5px] uppercase tracking-[0.14em] text-fg-disabled">{eyebrow}</div><div className="mt-0.5 text-[10.5px] font-medium text-fg">{title}</div></div>{action}</div>; }
function SectionTitle({ title, detail }: { title: string; detail: string }) { return <div><h3 className="text-[12px] font-medium text-fg">{title}</h3><p className="mt-1 max-w-2xl text-[9.5px] leading-4 text-fg-subtle">{detail}</p></div>; }
function DetailRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) { return <div className="grid grid-cols-[145px_minmax(0,1fr)] gap-3 py-2.5 text-[9px]"><dt className="text-fg-disabled">{label}</dt><dd className={cn("break-words text-fg-subtle", mono && "font-mono")}>{value}</dd></div>; }
function MetricCell({ label, value, inverse = false }: { label: string; value: string; inverse?: boolean }) { return <div className="bg-bg px-4 py-3"><div className="font-mono text-[16px] text-fg">{value}</div><div className="mt-1 text-[8.5px] uppercase tracking-[0.11em] text-fg-disabled">{label}</div>{inverse ? <div className="mt-0.5 text-[8px] text-fg-disabled">lower is better</div> : null}</div>; }
function InspectorTitle({ children }: { children: ReactNode }) { return <h3 className="mb-2 text-[8.5px] font-medium uppercase tracking-[0.12em] text-fg-muted">{children}</h3>; }
function Readiness({ label, ready }: { label: string; ready: boolean }) { return <div className="flex items-center justify-between border-b border-border-subtle py-2 text-[9px]"><span className="text-fg-subtle">{label}</span><span className={cn("inline-flex items-center gap-1", ready ? "text-success" : "text-warning")}>{ready ? <CheckCircle2 className="h-3 w-3" /> : <CircleDashed className="h-3 w-3" />}{ready ? "ready" : "required"}</span></div>; }
function InspectorBlock({ label, value }: { label: string; value: unknown }) { return <div className="mt-4"><InspectorTitle>{label}</InspectorTitle><pre className="max-h-48 overflow-auto whitespace-pre-wrap border border-border bg-bg p-2 font-mono text-[8.5px] leading-4 text-fg-subtle">{formatUnknown(value) || "Unavailable"}</pre></div>; }
function InspectorMetric({ label, value }: { label: string; value: string }) { return <div className="bg-bg px-2 py-2"><div className="text-[8px] uppercase tracking-wider text-fg-disabled">{label}</div><div className="mt-0.5 font-mono text-[10px] text-fg">{value}</div></div>; }
function AuditField({ label, children }: { label: string; children: ReactNode }) { return <label className="block"><span className="mb-1.5 block text-[8.5px] uppercase tracking-[0.11em] text-fg-disabled">{label}</span>{children}</label>; }
function Loading({ label }: { label: string }) { return <div className="flex min-h-40 items-center justify-center gap-2 p-6 text-[10px] text-fg-muted"><Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />{label}</div>; }
function Unavailable({ label }: { label: string }) { return <div className="flex min-h-40 items-center justify-center gap-2 p-6 text-center text-[10px] text-fg-muted"><AlertTriangle className="h-3.5 w-3.5 text-warning" />{label}</div>; }
function Empty({ label }: { label: string }) { return <div className="grid min-h-40 place-items-center p-6 text-center"><div><CircleDashed className="mx-auto h-4 w-4 text-fg-disabled" /><p className="mx-auto mt-2 max-w-sm text-[9.5px] leading-4 text-fg-muted">{label}</p></div></div>; }
function Pager({ total, page, pageSize, onPage }: { total: number; page: number; pageSize: number; onPage: (page: number) => void }) { const pages = Math.max(1, Math.ceil(total / pageSize)); return <div className="flex items-center justify-between border-t border-border-subtle px-4 py-2"><span className="font-mono text-[8.5px] text-fg-disabled">{total ? `${(page - 1) * pageSize + 1}–${Math.min(total, page * pageSize)} of ${total}` : "0 records"}</span><div className="flex gap-1"><Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</Button><Button size="sm" variant="ghost" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</Button></div></div>; }
function Th({ children }: { children: ReactNode }) { return <th className="px-3 py-2 text-[8px] font-medium uppercase tracking-[0.11em] text-fg-disabled">{children}</th>; }
function Td({ children, mono = false, className }: { children: ReactNode; mono?: boolean; className?: string }) { return <td className={cn("px-3 py-2 text-fg-subtle", mono && "font-mono", className)}>{children}</td>; }
function metricValue(items: RewardIntegrityMetric[] | undefined, name: string) { return items?.find((item) => item.name === name || item.name.endsWith(`.${name}`))?.value ?? null; }
function lowerIsBetter(name: string) { return ["error", "gap", "disagreement", "optimizer_only", "saturation"].some((part) => name.includes(part)); }
function humanize(value: string) { return value.replace(/[_-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function short(value: string) { return value.length > 18 ? `${value.slice(0, 9)}…${value.slice(-6)}` : value; }
function number(value?: number | null) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—"; }
function signed(value?: number | null) { return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(3)}` : "—"; }
function percent(value?: number | null) { return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—"; }
function auditBoundaryLabel(audit: RewardIntegrityAudit) { if (audit.boundary_unit === "final") return "Final boundary"; if (audit.boundary_value != null) return `${humanize(audit.boundary_unit || "boundary")} ${audit.boundary_value}`; return "Recorded boundary"; }
function formatUnknown(value: unknown): string { if (value == null) return ""; if (typeof value === "string") return value; if (typeof value === "number" || typeof value === "boolean") return String(value); return JSON.stringify(value, null, 2); }
function verifierRevision(items: VerifierProfile[], revisionId: string) { return items.flatMap((item) => item.latest_revision ? [item.latest_revision] : []).find((item) => item.id === revisionId); }
function verifierFingerprints(revision: NonNullable<VerifierProfile["latest_revision"]>): string[] { return [...new Set([revision.implementation_fingerprint, revision.artifact_hash, ...(revision.components ?? []).map((item) => item.child?.implementation_fingerprint)].filter((value): value is string => Boolean(value)))]; }
