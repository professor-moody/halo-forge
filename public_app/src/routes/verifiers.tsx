import { createFileRoute, Link, redirect } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Braces,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  GitCompareArrows,
  Loader2,
  Package,
  Play,
  Plug,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TestTube2,
  TriangleAlert,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  api,
  type BenchmarkSuite,
  type LabelSet,
  type VerifierCalibration,
  type VerifierCalibrationComparison,
  type VerifierCalibrationMetric,
  type VerifierCalibrationProtocolRevision,
  type VerifierCalibrationSample,
  type VerifierCapabilityDescriptor,
  type VerifierCatalogEntry,
  type VerifierFamily,
  type VerifierProfile,
  type VerifierProfileRevision,
  type VerifierQualificationDecision,
  type VerifierQualificationProfileRevision,
} from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SearchPicker } from "@/components/ui/search-picker";
import { cn } from "@/lib/utils";
import { RewardIntegrityWorkspace, type RewardAuditStudioView } from "@/components/research/reward-integrity-workspace";

export const Route = createFileRoute("/verifiers")({
  beforeLoad: () => {
    throw redirect({ to: "/eval", search: { section: "verifiers", verifierView: "catalog" }, replace: true });
  },
  component: VerifiersRedirect,
});

function VerifiersRedirect() {
  return null;
}

export type VerifierStudioView = "catalog" | "profiles" | "calibrate" | "compare" | "qualification" | "training-audits";

const VERIFIER_VIEWS: Array<{ id: VerifierStudioView; label: string; icon: ReactNode }> = [
  { id: "catalog", label: "Catalog", icon: <Package /> },
  { id: "profiles", label: "Profiles", icon: <Braces /> },
  { id: "calibrate", label: "Calibrate", icon: <TestTube2 /> },
  { id: "compare", label: "Compare", icon: <GitCompareArrows /> },
  { id: "qualification", label: "Qualification", icon: <ShieldCheck /> },
  { id: "training-audits", label: "Training audits", icon: <Activity /> },
];

export function VerifierReliabilityWorkspace({
  view,
  selectedProfileId,
  selectedCalibrationId,
  onView,
  onProfile,
  onCalibration,
  auditView = "profiles",
  selectedAuditId,
  baseAuditId,
  candidateAuditId,
  selectedAuditSampleId,
  auditPage = 1,
  auditClassification,
  onAuditView = () => undefined,
  onAudit = () => undefined,
  onAuditCompare = () => undefined,
  onAuditSample = () => undefined,
  onAuditPage = () => undefined,
  onAuditClassification = () => undefined,
}: {
  view: VerifierStudioView;
  selectedProfileId?: string;
  selectedCalibrationId?: string;
  onView: (view: VerifierStudioView) => void;
  onProfile: (profileId?: string) => void;
  onCalibration: (calibrationId?: string) => void;
  auditView?: RewardAuditStudioView;
  selectedAuditId?: string;
  baseAuditId?: string;
  candidateAuditId?: string;
  selectedAuditSampleId?: string;
  auditPage?: number;
  auditClassification?: string;
  onAuditView?: (view: RewardAuditStudioView) => void;
  onAudit?: (auditId?: string) => void;
  onAuditCompare?: (baseId?: string, candidateId?: string) => void;
  onAuditSample?: (sampleId?: string) => void;
  onAuditPage?: (page: number) => void;
  onAuditClassification?: (classification?: string) => void;
}) {
  return (
    <div className="min-h-[calc(100vh-112px)] bg-bg">
      <div className="flex gap-1 overflow-x-auto border-b border-border bg-bg-subtle/45 px-4 md:px-5" aria-label="Verifier Reliability views">
        {VERIFIER_VIEWS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => onView(item.id)}
            className={cn(
              "group relative flex h-10 shrink-0 items-center gap-1.5 px-3 text-[11px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset",
              view === item.id ? "font-medium text-fg" : "text-fg-subtle hover:text-fg",
            )}
          >
            <span className={cn("[&_svg]:h-3.5 [&_svg]:w-3.5", view === item.id ? "text-accent" : "text-fg-disabled group-hover:text-fg-subtle")}>{item.icon}</span>
            {item.label}
            {view === item.id ? <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" /> : null}
          </button>
        ))}
      </div>
      {view === "catalog" ? <CatalogView onCreateProfile={() => onView("profiles")} /> : null}
      {view === "profiles" ? <ProfilesView selectedProfileId={selectedProfileId} onProfile={onProfile} onCalibrate={(profileId) => { onProfile(profileId); onView("calibrate"); }} /> : null}
      {view === "calibrate" ? <CalibrationView selectedProfileId={selectedProfileId} selectedCalibrationId={selectedCalibrationId} onProfile={onProfile} onCalibration={onCalibration} onCompare={() => onView("compare")} /> : null}
      {view === "compare" ? <CalibrationCompareView selectedCalibrationId={selectedCalibrationId} /> : null}
      {view === "qualification" ? <QualificationView selectedProfileId={selectedProfileId} onProfile={onProfile} /> : null}
      {view === "training-audits" ? <RewardIntegrityWorkspace view={auditView} selectedAuditId={selectedAuditId} baseAuditId={baseAuditId} candidateAuditId={candidateAuditId} selectedSampleId={selectedAuditSampleId} page={auditPage} classification={auditClassification} onView={onAuditView} onAudit={onAudit} onCompare={onAuditCompare} onSample={onAuditSample} onPage={onAuditPage} onClassification={onAuditClassification} /> : null}
    </div>
  );
}

export function VerifierCatalogWorkspace({ embedded = false }: { embedded?: boolean }) {
  const [view, setView] = useState<VerifierStudioView>("catalog");
  return (
    <>
      {!embedded ? <Topbar eyebrow="Evaluate · Verifiers" title="Verifier Reliability" subtitle="Inspect, calibrate, and qualify exact verifier revisions before they shape data or training." /> : null}
      <VerifierReliabilityWorkspace view={view} onView={setView} onProfile={() => undefined} onCalibration={() => undefined} />
    </>
  );
}

function CatalogView({ onCreateProfile }: { onCreateProfile: () => void }) {
  const catalog = useQuery({ queryKey: ["verifier-catalog"], queryFn: api.verifierCatalog, staleTime: 30_000 });
  const capabilities = useQuery({ queryKey: ["verifier-reliability-capabilities"], queryFn: api.verifierReliabilityCapabilities, staleTime: 30_000, retry: false });
  const [origin, setOrigin] = useState("all");
  const [family, setFamily] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedName, setSelectedName] = useState("");

  const capabilityByImplementation = useMemo(() => new Map((capabilities.data?.items ?? []).map((item) => [item.implementation || item.id, item])), [capabilities.data?.items]);
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (catalog.data?.items ?? []).filter((entry) => {
      const capability = capabilityByImplementation.get(entry.name) ?? capabilityByImplementation.get(entry.cls);
      return (origin === "all" || entry.origin === origin) && (family === "all" || capability?.family === family) && (!needle || `${entry.name} ${entry.cls} ${entry.doc ?? ""} ${capability?.family ?? ""}`.toLowerCase().includes(needle));
    });
  }, [capabilityByImplementation, catalog.data?.items, family, origin, query]);

  useEffect(() => {
    if (!selectedName && filtered[0]) setSelectedName(filtered[0].name);
  }, [filtered, selectedName]);

  const selected = filtered.find((entry) => entry.name === selectedName) ?? filtered[0];
  const selectedCapability = selected ? capabilityByImplementation.get(selected.name) ?? capabilityByImplementation.get(selected.cls) : undefined;

  return (
    <SplitWorkspace
      rail={
        <>
          <RailHeader eyebrow="IMPLEMENTATIONS" title={`${filtered.length} available`} action={<Button size="icon" variant="ghost" onClick={() => { catalog.refetch(); capabilities.refetch(); }} aria-label="Refresh verifier catalog"><RefreshCw /></Button>} />
          <div className="space-y-2 border-b border-border-subtle p-3">
            <SearchField value={query} onChange={setQuery} placeholder="Find an implementation" />
            <div className="grid grid-cols-2 gap-2">
              <select aria-label="Verifier origin" value={origin} onChange={(event) => setOrigin(event.target.value)} className={selectClass}><option value="all">All origins</option><option value="builtin">Built-in</option><option value="user_plugin">User plugin</option><option value="entry_point">Entry point</option></select>
              <select aria-label="Verifier family" value={family} onChange={(event) => setFamily(event.target.value)} className={selectClass}><option value="all">All families</option><option value="deterministic">Deterministic</option><option value="llm_judge">LLM judge</option><option value="reward_model">Reward model</option><option value="chain">Chain</option></select>
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {catalog.isLoading ? <Loading label="Loading verifier implementations" /> : filtered.length ? filtered.map((entry) => <CatalogListItem key={entry.name} entry={entry} capability={capabilityByImplementation.get(entry.name) ?? capabilityByImplementation.get(entry.cls)} selected={selected?.name === entry.name} onSelect={() => setSelectedName(entry.name)} />) : <EmptyState icon={<Package />} title="No compatible verifiers" detail="Change the filters or confirm the plugin loaded." />}
          </div>
        </>
      }
      main={selected ? <CatalogDetail entry={selected} capability={selectedCapability} reliabilityUnavailable={capabilities.isError} onCreateProfile={onCreateProfile} /> : <EmptyState icon={<Package />} title="Choose a verifier" detail="Inspect its identity and reliability capabilities." />}
    />
  );
}

function CatalogListItem({ entry, capability, selected, onSelect }: { entry: VerifierCatalogEntry; capability?: VerifierCapabilityDescriptor; selected: boolean; onSelect: () => void }) {
  return <button type="button" onClick={onSelect} className={cn("w-full border-b border-border-subtle px-4 py-3 text-left transition-colors", selected ? "bg-accent/7" : "hover:bg-surface/50")}><div className="flex items-start justify-between gap-2"><span className={cn("truncate font-mono text-[11px]", selected ? "text-accent" : "text-fg")}>{entry.name}</span><OriginIcon origin={entry.origin} /></div><div className="mt-1 flex flex-wrap items-center gap-1.5 text-[9px] text-fg-disabled"><span>{capability ? familyLabel(capability.family) : "Legacy interface"}</span><span>·</span><span>{entry.origin.replace("_", " ")}</span>{capability?.fingerprintable === false ? <Badge size="sm" tone="warning">not qualifiable</Badge> : null}</div></button>;
}

function CatalogDetail({ entry, capability, reliabilityUnavailable, onCreateProfile }: { entry: VerifierCatalogEntry; capability?: VerifierCapabilityDescriptor; reliabilityUnavailable: boolean; onCreateProfile: () => void }) {
  return <div className="mx-auto max-w-5xl p-5 lg:p-8"><section className="border-b border-border pb-7"><div className="flex flex-wrap items-start justify-between gap-4"><div><div className="flex items-center gap-2 text-[9.5px] uppercase tracking-[0.14em] text-accent"><OriginIcon origin={entry.origin} />{entry.origin.replace("_", " ")}</div><h2 className="mt-2 font-mono text-xl font-semibold text-fg">{entry.name}</h2><p className="mt-2 max-w-2xl text-[11px] leading-5 text-fg-subtle">{entry.doc || "No implementation description is available."}</p></div><Button size="sm" variant="primary" onClick={onCreateProfile}><Plus />Create immutable profile</Button></div></section>
    {reliabilityUnavailable ? <Notice tone="warning" title="Reliability metadata unavailable">The legacy verifier remains inspectable, but qualification controls are unavailable until the v7 service is running.</Notice> : null}
    <section className="grid gap-7 py-7 lg:grid-cols-[minmax(0,1fr)_300px]"><div><SectionTitle eyebrow="CAPABILITY CONTRACT" title={capability ? familyLabel(capability.family) : "Legacy verifier"} detail={capability?.description || "This implementation has not declared a v7 reliability adapter."} /><DefinitionList values={{ implementation: entry.cls, module: entry.module, reliability_adapter: capability?.id ?? "unavailable", fingerprint: capability?.implementation_fingerprint ?? "unavailable", modalities: capability?.modalities?.join(", ") || "undeclared", task_types: capability?.task_types?.join(", ") || "undeclared", seed_support: capability ? yesNo(capability.supports_seed) : "unknown" }} /></div><aside className="border-l border-border-subtle pl-5"><InspectorTitle>Qualification readiness</InspectorTitle><ReadinessRow label="Implementation fingerprint" ready={Boolean(capability?.fingerprintable && capability.implementation_fingerprint)} /><ReadinessRow label="Reliability adapter" ready={Boolean(capability)} /><ReadinessRow label="Reward contract" ready={false} pendingLabel="set in profile" /><ReadinessRow label="Human calibration" ready={false} pendingLabel="not calibrated" /><p className="mt-4 text-[9.5px] leading-4 text-fg-disabled">An implementation is not a research input until a profile pins its contract and a calibration qualifies the exact revision.</p></aside></section>
  </div>;
}

function ProfilesView({ selectedProfileId, onProfile, onCalibrate }: { selectedProfileId?: string; onProfile: (id?: string) => void; onCalibrate: (id: string) => void }) {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const profiles = useQuery({ queryKey: ["verifier-profiles", query], queryFn: () => api.listVerifierProfiles({ q: query || undefined, limit: 100 }), retry: false });
  const capabilities = useQuery({ queryKey: ["verifier-reliability-capabilities"], queryFn: api.verifierReliabilityCapabilities, retry: false });
  const profileId = selectedProfileId ?? profiles.data?.items[0]?.id;
  const detail = useQuery({ queryKey: ["verifier-profile", profileId], queryFn: () => api.verifierProfile(profileId!), enabled: Boolean(profileId), retry: false });
  const create = useMutation({ mutationFn: (payload: Record<string, unknown>) => api.createVerifierProfile(payload), onSuccess: (profile) => { queryClient.invalidateQueries({ queryKey: ["verifier-profiles"] }); onProfile(profile.id); setCreateOpen(false); } });

  useEffect(() => { if (!selectedProfileId && profileId) onProfile(profileId); }, [onProfile, profileId, selectedProfileId]);

  return <SplitWorkspace rail={<><RailHeader eyebrow="IMMUTABLE PROFILES" title={`${profiles.data?.total ?? profiles.data?.items.length ?? 0} profiles`} action={<Button size="icon" variant="ghost" onClick={() => setCreateOpen(true)} aria-label="Create verifier profile"><Plus /></Button>} /><div className="border-b border-border-subtle p-3"><SearchField value={query} onChange={setQuery} placeholder="Search profiles" /></div><div className="min-h-0 flex-1 overflow-y-auto">{profiles.isLoading ? <Loading label="Loading verifier profiles" /> : profiles.isError ? <ServiceUnavailable label="Profile service is unavailable" /> : profiles.data?.items.length ? profiles.data.items.map((profile) => <ProfileListItem key={profile.id} profile={profile} selected={profile.id === profileId} onSelect={() => { onProfile(profile.id); setCreateOpen(false); }} />) : <EmptyState icon={<Braces />} title="No profiles yet" detail="Create one to pin verifier identity, inputs, and reward behavior." />}</div></>} main={createOpen ? <ProfileWizard capabilities={capabilities.data?.items ?? []} availableProfiles={profiles.data?.items ?? []} onCancel={() => setCreateOpen(false)} onCreate={(payload) => create.mutate(payload)} pending={create.isPending} error={create.error?.message} /> : detail.data ? <ProfileDetail profile={detail.data} onCalibrate={() => onCalibrate(detail.data.id)} /> : detail.isLoading ? <Loading label="Opening profile" /> : <EmptyState icon={<Braces />} title="Select a profile" detail="Inspect immutable revisions and qualification state." />} />;
}

function ProfileListItem({ profile, selected, onSelect }: { profile: VerifierProfile; selected: boolean; onSelect: () => void }) {
  const revision = profile.latest_revision;
  return <button type="button" onClick={onSelect} className={cn("w-full border-b border-border-subtle px-4 py-3 text-left transition-colors", selected ? "bg-accent/7" : "hover:bg-surface/50")}><div className="flex items-center justify-between gap-2"><span className={cn("truncate text-[11.5px] font-medium", selected ? "text-accent" : "text-fg")}>{profile.name}</span><QualificationBadge state={revision?.qualification_state || "unqualified"} /></div><div className="mt-1 truncate text-[9px] text-fg-disabled">{revision ? `${familyLabel(revision.family)} · ${revision.modality} · r${revision.revision_number}` : "No revision"}</div></button>;
}

function ProfileWizard({ capabilities, availableProfiles, onCancel, onCreate, pending, error }: { capabilities: VerifierCapabilityDescriptor[]; availableProfiles: VerifierProfile[]; onCancel: () => void; onCreate: (payload: Record<string, unknown>) => void; pending: boolean; error?: string }) {
  const models = useQuery({ queryKey: ["models", "verifier-profile-picker"], queryFn: () => api.modelCatalog(), retry: false });
  const artifacts = useQuery({ queryKey: ["model-artifacts", "verifier-profile-picker"], queryFn: () => api.listModelArtifacts({ limit: 200 }), retry: false });
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [family, setFamily] = useState<VerifierFamily>("deterministic");
  const [implementationId, setImplementationId] = useState("");
  const [modality, setModality] = useState("text");
  const [taskType, setTaskType] = useState("binary");
  const [minimum, setMinimum] = useState("0");
  const [maximum, setMaximum] = useState("1");
  const [threshold, setThreshold] = useState("0.5");
  const [probability, setProbability] = useState(false);
  const [rubric, setRubric] = useState("");
  const [judgeModelId, setJudgeModelId] = useState("");
  const [judgeModelRevision, setJudgeModelRevision] = useState("");
  const [rewardArtifactId, setRewardArtifactId] = useState("");
  const [endpointType, setEndpointType] = useState("local");
  const [components, setComponents] = useState<Array<{ revisionId: string; weight: number; veto: boolean }>>([]);
  const compatible = capabilities.filter((item) => item.family === family);
  const implementationOptions = family === "chain" ? [{ id: "chain", family: "chain" as const, label: "Ordered verifier chain", description: "Aggregate exact child revisions with visible component traces.", modalities: ["text", "vlm", "audio"], task_types: ["binary", "scalar", "pairwise", "ranking"], fingerprintable: true, origin: "builtin", implementation_fingerprint: "ordered-child-revisions" }] : compatible;
  useEffect(() => { if (!implementationOptions.some((item) => item.id === implementationId)) setImplementationId(implementationOptions[0]?.id ?? ""); }, [implementationId, implementationOptions]);
  const selected = implementationOptions.find((item) => item.id === implementationId);
  useEffect(() => {
    if (!selected) return;
    if (!selected.modalities.includes(modality)) setModality(selected.modalities[0] ?? "");
    if (!selected.task_types.includes(taskType)) setTaskType(selected.task_types[0] ?? "");
    if (selected.supports_probability === false) setProbability(false);
  }, [modality, selected, taskType]);
  const judgeModelOptions = (models.data?.items ?? []).filter((item) => !item.modalities?.length || item.modalities.includes(modality) || item.modalities.includes("multimodal")).map((item) => ({ value: item.id, label: item.label || item.id, description: `${item.provider || "catalog"} · ${item.memory_tier || "memory unknown"}`, status: item.status, keywords: item.id }));
  const rewardArtifactOptions = (artifacts.data?.items ?? []).filter((item) => ["final_model", "merged", "converted", "quantized"].includes(item.kind) && ["verified", "valid"].includes(item.integrity || "") && Boolean(item.content_hash && item.path)).map((item) => ({ value: item.id, label: item.model_name || `${humanize(item.kind)} artifact`, description: `${item.format || "managed"} · ${shortHash(item.content_hash)}`, status: "verified", keywords: `${item.content_hash} ${item.run_id || ""}` }));
  const steps = ["Family", "Task & reward", "Rubric & runtime", "Review"];
  const implementationValid = Boolean(name.trim() && implementationId && selected);
  const contractValid = Boolean(modality && taskType && selected?.modalities.includes(modality) && selected.task_types.includes(taskType) && Number.isFinite(Number(minimum)) && Number.isFinite(Number(threshold)) && Number.isFinite(Number(maximum)) && Number(maximum) > Number(minimum) && Number(threshold) >= Number(minimum) && Number(threshold) <= Number(maximum));
  const runtimeValid = family === "llm_judge" ? isPinnedModelRevision(judgeModelRevision) : family === "reward_model" ? rewardArtifactOptions.some((item) => item.value === rewardArtifactId) : family === "chain" ? components.some((item) => item.revisionId) : true;
  const stepValid = [implementationValid, contractValid, runtimeValid, true][step];
  const valid = implementationValid && contractValid && runtimeValid;
  const resolvedModelRevision = family === "llm_judge" ? judgeModelRevision.trim() : family === "reward_model" ? rewardArtifactId : undefined;
  const payload = { name: name.trim(), description: description.trim(), family, implementation_id: implementationId, modality, task_type: taskType, reward_contract: { minimum: Number(minimum), maximum: Number(maximum), direction: "maximize", threshold: Number(threshold), tie_policy: "error", probability_semantics: probability, error_behavior: "fail_closed" }, rubric: rubric || undefined, model_revision: resolvedModelRevision, endpoint_type: family === "llm_judge" ? endpointType : undefined, components: family === "chain" ? components.filter((item) => item.revisionId).map((item, ordinal) => ({ child_revision_id: item.revisionId, ordinal, weight: item.weight, veto: item.veto, aggregation_rule: "weighted_mean" })) : undefined };

  return <div className="mx-auto max-w-5xl p-5 lg:p-8"><div className="border-b border-border pb-5"><div className="text-[9.5px] uppercase tracking-[0.14em] text-accent">GUIDED PROFILE</div><h2 className="mt-1 text-lg font-semibold text-fg">Pin a verifier as a reproducible research input</h2><p className="mt-1 text-[10.5px] text-fg-subtle">Every change after publication creates a new immutable revision.</p><div className="mt-5 grid grid-cols-4 gap-px overflow-hidden rounded-md border border-border bg-border">{steps.map((label, index) => <button key={label} type="button" onClick={() => index <= step && setStep(index)} className={cn("flex min-h-12 items-center gap-2 bg-bg px-3 text-left text-[9.5px]", index === step ? "text-accent" : index < step ? "text-fg" : "text-fg-disabled")}><span className={cn("grid h-5 w-5 shrink-0 place-items-center rounded-full border font-mono text-[8px]", index <= step ? "border-accent/50 bg-accent/7" : "border-border")}>{index < step ? <Check className="h-3 w-3" /> : index + 1}</span><span className="hidden sm:block">{label}</span></button>)}</div></div>
    <div className="min-h-[360px] py-6">{step === 0 ? <div className="grid gap-6 lg:grid-cols-2"><div className="space-y-4"><Field label="Profile name"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Grounded answer judge" /></Field><Field label="Purpose"><Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Scores grounded development answers" /></Field><Field label="Family"><div className="grid grid-cols-2 gap-2">{(["deterministic", "llm_judge", "reward_model", "chain"] as VerifierFamily[]).map((value) => <ChoiceButton key={value} selected={family === value} label={familyLabel(value)} onClick={() => setFamily(value)} />)}</div></Field></div><div><Field label="Implementation"><SearchPicker value={implementationId} onChange={setImplementationId} options={implementationOptions.map((item) => ({ value: item.id, label: item.label, description: `${item.origin ?? "registered"} · ${item.fingerprintable === false ? "inspect only" : "fingerprintable"}`, status: item.fingerprintable === false ? "unqualified" : "ready" }))} placeholder="Choose a compatible implementation" emptyLabel="No implementation declares this family" /></Field>{selected ? <div className="mt-4 border-l-2 border-accent/35 pl-4"><div className="text-[10px] font-medium text-fg">{selected.label}</div><p className="mt-1 text-[9.5px] leading-4 text-fg-subtle">{selected.description}</p><div className="mt-2 font-mono text-[8px] text-fg-disabled">{selected.implementation_fingerprint || "No stable fingerprint · cannot normally qualify"}</div></div> : null}</div></div> : null}
      {step === 1 ? <div className="grid gap-6 lg:grid-cols-2"><div className="space-y-4"><Field label="Modality"><select value={modality} onChange={(event) => setModality(event.target.value)} className={selectClass}>{(selected?.modalities ?? ["text", "vlm", "audio"]).map((value) => <option key={value} value={value}>{value}</option>)}</select></Field><Field label="Task"><select value={taskType} onChange={(event) => setTaskType(event.target.value)} className={selectClass}>{(selected?.task_types ?? ["binary", "categorical", "scalar", "pairwise", "ranking"]).map((value) => <option key={value} value={value}>{value.replace("_", " ")}</option>)}</select></Field><label className="flex items-center gap-2 text-[10px] text-fg-subtle"><input type="checkbox" checked={probability} onChange={(event) => setProbability(event.target.checked)} />Rewards have explicit probability semantics</label></div><div><div className="grid grid-cols-3 gap-3"><Field label="Minimum"><Input type="number" value={minimum} onChange={(event) => setMinimum(event.target.value)} /></Field><Field label="Threshold"><Input type="number" value={threshold} onChange={(event) => setThreshold(event.target.value)} /></Field><Field label="Maximum"><Input type="number" value={maximum} onChange={(event) => setMaximum(event.target.value)} /></Field></div><div className="mt-6 h-2 overflow-hidden rounded-full bg-border"><div className="h-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, (Number(threshold) - Number(minimum)) / Math.max(0.0001, Number(maximum) - Number(minimum)) * 100))}%` }} /></div><div className="mt-2 flex justify-between font-mono text-[9px] text-fg-disabled"><span>{minimum}</span><span>pass at {threshold}</span><span>{maximum}</span></div><Notice tone="neutral" title="Out-of-contract scores are rejected">Halo Forge records the error instead of silently clamping a non-finite or out-of-range reward.</Notice></div></div> : null}
      {step === 2 ? family === "chain" ? <ChainComponentEditor components={components} onChange={setComponents} profiles={availableProfiles} /> : <div className="grid gap-6 lg:grid-cols-2"><div><Field label={family === "llm_judge" ? "Pinned rubric" : "Output contract notes"}><textarea className={textareaClass} rows={10} value={rubric} onChange={(event) => setRubric(event.target.value)} placeholder="State exactly what the verifier should reward, reject, or abstain on." /></Field></div><div className="space-y-4">{family === "llm_judge" ? <><Field label="Judge model family"><SearchPicker allowEmpty value={judgeModelId} onChange={(value) => { setJudgeModelId(value); if (isPinnedModelRevision(value)) setJudgeModelRevision(value); }} options={judgeModelOptions} placeholder="Choose a compatible catalog model" emptyLabel="No catalog model declares this modality" /></Field><Field label="Pinned model revision"><Input value={judgeModelRevision} onChange={(event) => setJudgeModelRevision(event.target.value)} placeholder="organization/model@commit or dated provider model" /><p className={cn("mt-1 text-[8.5px] leading-4", judgeModelRevision && !isPinnedModelRevision(judgeModelRevision) ? "text-warning" : "text-fg-disabled")}>Use an immutable endpoint-resolvable revision: a repository commit, digest, or dated provider model. Moving aliases cannot be published in Guided mode.</p></Field><Field label="Endpoint runtime"><select value={endpointType} onChange={(event) => setEndpointType(event.target.value)} className={selectClass}><option value="local">Local OpenAI-compatible</option><option value="ollama">Ollama</option><option value="hosted">Configured hosted provider</option></select></Field></> : null}{family === "reward_model" ? <Field label="Verified reward-model artifact"><SearchPicker value={rewardArtifactId} onChange={setRewardArtifactId} options={rewardArtifactOptions} placeholder="Choose a verified managed artifact" emptyLabel="No verified compatible reward-model artifact is available" /><p className="mt-1 text-[8.5px] leading-4 text-fg-disabled">Only loadable final, merged, converted, or quantized occurrences with verified content identity are shown.</p></Field> : null}<DefinitionList values={{ execution: family === "deterministic" ? "fresh process · two repeats" : "three fixed-seed repeats", errors: "propagate into observation", credentials: "never stored", parser: "pinned by revision" }} /></div></div> : null}
      {step === 3 ? <div className="grid gap-7 lg:grid-cols-[minmax(0,1fr)_300px]"><div><SectionTitle eyebrow="READY TO PUBLISH" title={name || "Untitled verifier profile"} detail={description || "No purpose supplied."} /><DefinitionList values={{ family: familyLabel(family), implementation: selected?.label || implementationId, modality, task: taskType, reward_range: `${minimum} → ${maximum}`, pass_threshold: threshold, probability_semantics: yesNo(probability), model_revision: family === "llm_judge" ? judgeModelRevision || "required" : "not applicable", reward_model_artifact: family === "reward_model" ? rewardArtifactId || "required" : "not applicable", runtime_contract: "captured from the current Python, toolchain, and relevant hardware", ordered_components: family === "chain" ? components.filter((item) => item.revisionId).length : "not applicable" }} /></div><aside className="border-l border-border-subtle pl-5"><InspectorTitle>After publication</InspectorTitle><p className="text-[9.5px] leading-4 text-fg-subtle">The revision can be calibrated against human-reference evidence. It will remain hidden from guided training pickers until qualification succeeds.</p><ReadinessRow label="Stable implementation" ready={selected?.fingerprintable !== false} /><ReadinessRow label="Reward contract" ready /><ReadinessRow label={family === "llm_judge" ? "Pinned judge revision" : family === "reward_model" ? "Verified artifact" : "Runtime identity"} ready={runtimeValid} /><ReadinessRow label="Calibration" ready={false} pendingLabel="next step" /></aside></div> : null}</div>
    <div className="flex items-center justify-between border-t border-border pt-4"><Button variant="ghost" size="sm" onClick={step ? () => setStep(step - 1) : onCancel}>{step ? "Back" : "Cancel"}</Button><div className="flex items-center gap-3">{error ? <span role="alert" className="text-[10px] text-danger">{error}</span> : null}{step < 3 ? <Button size="sm" variant="primary" onClick={() => setStep(step + 1)} disabled={!stepValid}>Continue <ArrowRight /></Button> : <Button size="sm" variant="primary" onClick={() => onCreate(payload)} disabled={!valid || pending}>{pending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Publish immutable revision</Button>}</div></div>
  </div>;
}

function ProfileDetail({ profile, onCalibrate }: { profile: VerifierProfile & { revisions?: VerifierProfileRevision[] }; onCalibrate: () => void }) {
  const revision = profile.latest_revision ?? profile.revisions?.[0];
  if (!revision) return <EmptyState icon={<Braces />} title="Profile has no revision" detail="Publish a revision before calibration." />;
  return <div className="mx-auto max-w-5xl p-5 lg:p-8"><section className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-6"><div><div className="flex items-center gap-2"><span className="text-[9.5px] uppercase tracking-[0.14em] text-accent">{familyLabel(revision.family)}</span><QualificationBadge state={revision.qualification_state || "unqualified"} /></div><h2 className="mt-2 text-xl font-semibold text-fg">{profile.name}</h2><p className="mt-1 max-w-2xl text-[10.5px] text-fg-subtle">{profile.description}</p></div><Button size="sm" variant="primary" onClick={onCalibrate}><TestTube2 />Calibrate this revision</Button></section><section className="grid gap-7 py-7 lg:grid-cols-[minmax(0,1fr)_300px]"><div><SectionTitle eyebrow="REVISION CONTRACT" title={`Revision ${revision.revision_number}`} detail="Immutable identity used by datasets, evaluations, suggestions, and training." /><DefinitionList values={{ content_hash: revision.content_hash || "pending", implementation_fingerprint: revision.implementation_fingerprint || "unavailable", reliability_adapter: `${revision.reliability_adapter_id || "unavailable"}@${revision.reliability_adapter_version || "—"}`, modality: revision.modality, task: revision.task_type, reward: `${revision.reward_contract.minimum} → ${revision.reward_contract.maximum} · threshold ${revision.reward_contract.threshold ?? "—"}`, probability_semantics: yesNo(revision.reward_contract.probability_semantics), model_revision: revision.model_revision || "not applicable", tokenizer_revision: revision.tokenizer_revision || "not applicable" }} />{revision.components?.length ? <div className="mt-7"><InspectorTitle>Ordered chain</InspectorTitle><ol className="mt-2 divide-y divide-border-subtle border-y border-border-subtle">{revision.components.map((component) => <li key={component.id || component.child_revision_id} className="grid grid-cols-[28px_1fr_auto] items-center gap-2 py-2.5 text-[10px]"><span className="font-mono text-fg-disabled">{component.ordinal + 1}</span><span className="truncate text-fg">{component.child?.profile_id || component.child_revision_id}</span><span className="text-fg-disabled">weight {component.weight ?? 1}{component.veto ? " · veto" : ""}</span></li>)}</ol></div> : null}</div><aside className="border-l border-border-subtle pl-5"><InspectorTitle>Runtime scope</InspectorTitle><ReadinessRow label="Current runtime" ready={revision.runtime_compatible !== false} pendingLabel="stale" /><ReadinessRow label="Development qualification" ready={revision.qualification_state === "pass" || revision.alias === "candidate" || revision.alias === "approved"} /><ReadinessRow label="Approved confirmation" ready={revision.alias === "approved"} /><p className="mt-4 break-all font-mono text-[8px] leading-4 text-fg-disabled">{revision.content_hash}</p></aside></section></div>;
}

function ChainComponentEditor({ components, onChange, profiles }: { components: Array<{ revisionId: string; weight: number; veto: boolean }>; onChange: (components: Array<{ revisionId: string; weight: number; veto: boolean }>) => void; profiles: VerifierProfile[] }) {
  const options = profiles.flatMap((profile) => { const revision = profile.latest_revision; const qualified = revision && (revision.qualification_state === "pass" || ["candidate", "approved"].includes(revision.alias || "")); return revision && qualified ? [{ value: revision.id, label: profile.name, description: `${familyLabel(revision.family)} · r${revision.revision_number} · ${revision.alias || revision.qualification_state}` }] : []; });
  const rows = components.length ? components : [{ revisionId: "", weight: 1, veto: false }];
  function update(index: number, patch: Partial<{ revisionId: string; weight: number; veto: boolean }>) { onChange(rows.map((item, position) => position === index ? { ...item, ...patch } : item)); }
  function move(index: number, delta: number) { const target = index + delta; if (target < 0 || target >= rows.length) return; const next = [...rows]; [next[index], next[target]] = [next[target], next[index]]; onChange(next); }
  return <div className="lg:col-span-2"><SectionTitle eyebrow="ORDERED COMPONENTS" title="Compose candidate-qualified child revisions" detail="Every child error remains visible in the aggregate trace. Cycles are rejected when this revision is published." /><div className="mt-4 divide-y divide-border-subtle overflow-hidden rounded-md border border-border">{rows.map((component, index) => <div key={index} className="grid gap-3 bg-bg p-3 md:grid-cols-[28px_minmax(0,1fr)_100px_90px_auto]"><span className="pt-2 text-center font-mono text-[9px] text-fg-disabled">{index + 1}</span><SearchPicker value={component.revisionId} onChange={(revisionId) => update(index, { revisionId })} options={options.filter((option) => option.value === component.revisionId || !rows.some((row) => row.revisionId === option.value))} placeholder="Choose a qualified child revision" emptyLabel="No compatible candidate-qualified revision" /><Field label="Weight"><Input type="number" min="0" step="0.1" value={component.weight} onChange={(event) => update(index, { weight: Number(event.target.value) })} /></Field><label className="flex items-center gap-2 pt-5 text-[9.5px] text-fg-subtle"><input type="checkbox" checked={component.veto} onChange={(event) => update(index, { veto: event.target.checked })} />Veto</label><div className="flex items-center"><button type="button" className="px-1 text-fg-disabled hover:text-fg disabled:opacity-30" disabled={index === 0} onClick={() => move(index, -1)}>↑</button><button type="button" className="px-1 text-fg-disabled hover:text-fg disabled:opacity-30" disabled={index === rows.length - 1} onClick={() => move(index, 1)}>↓</button><button type="button" className="px-1 text-fg-disabled hover:text-danger disabled:opacity-30" disabled={rows.length === 1} onClick={() => onChange(rows.filter((_, position) => position !== index))}>×</button></div></div>)}</div><button type="button" className="mt-2 inline-flex items-center gap-1 text-[10px] text-accent hover:underline" onClick={() => onChange([...rows, { revisionId: "", weight: 1, veto: false }])}><Plus className="h-3 w-3" />Add component</button><Notice tone="neutral" title="Aggregation stays explicit">The default weighted mean, weights, order, and vetoes become part of the immutable profile hash.</Notice></div>;
}

function CalibrationView({ selectedProfileId, selectedCalibrationId, onProfile, onCalibration, onCompare }: { selectedProfileId?: string; selectedCalibrationId?: string; onProfile: (id?: string) => void; onCalibration: (id?: string) => void; onCompare: () => void }) {
  const queryClient = useQueryClient();
  const profiles = useQuery({ queryKey: ["verifier-profiles", "calibration"], queryFn: () => api.listVerifierProfiles({ limit: 200 }), retry: false });
  const calibrations = useQuery({ queryKey: ["verifier-calibrations"], queryFn: () => api.listVerifierCalibrations({ limit: 100 }), refetchInterval: 5_000, retry: false });
  const protocols = useQuery({ queryKey: ["verifier-calibration-protocols"], queryFn: () => api.listVerifierProtocols({ limit: 100 }), retry: false });
  const qualificationProfiles = useQuery({ queryKey: ["verifier-qualification-profiles"], queryFn: () => api.listVerifierQualificationProfiles({ limit: 100 }), retry: false });
  const labelSets = useQuery({ queryKey: ["label-sets", "calibration"], queryFn: () => api.listLabelSets({ limit: 200 }), retry: false });
  const suites = useQuery({ queryKey: ["benchmark-suites"], queryFn: api.listBenchmarkSuites, retry: false });
  const [launchOpen, setLaunchOpen] = useState(false);
  const [profileId, setProfileId] = useState(selectedProfileId || "");
  const [sourceKind, setSourceKind] = useState<"label_set_revision" | "benchmark_suite_revision">("label_set_revision");
  const [sourceId, setSourceId] = useState("");
  const [protocolId, setProtocolId] = useState("");
  const [qualificationId, setQualificationId] = useState("");
  const [confirmation, setConfirmation] = useState(true);
  const [pairwiseOrder, setPairwiseOrder] = useState(true);
  const [paraphraseProbe, setParaphraseProbe] = useState(false);
  const launch = useMutation({ mutationFn: () => { const revision = profiles.data?.items.find((item) => item.id === profileId)?.latest_revision; if (!revision) throw new Error("Choose a profile with a published revision."); return api.createVerifierCalibration({ profile_revision_id: revision.id, source_kind: sourceKind, source_revision_id: sourceId, protocol_revision_id: protocolId, qualification_profile_revision_id: qualificationId, confirmation_requested: confirmation, perturbations: [pairwiseOrder ? "counterbalanced_order" : null, paraphraseProbe ? "reviewed_paraphrase" : null].filter(Boolean) }); }, onSuccess: (value) => { queryClient.invalidateQueries({ queryKey: ["verifier-calibrations"] }); onCalibration(value.id); setLaunchOpen(false); } });
  useEffect(() => { if (selectedProfileId) { setProfileId(selectedProfileId); onProfile(selectedProfileId); } }, [onProfile, selectedProfileId]);
  useEffect(() => { if (!protocolId && protocols.data?.items[0]) setProtocolId(protocols.data.items[0].id); }, [protocolId, protocols.data?.items]);
  useEffect(() => { if (!qualificationId && qualificationProfiles.data?.items[0]) setQualificationId(qualificationProfiles.data.items[0].id); }, [qualificationId, qualificationProfiles.data?.items]);
  const sourceOptions = sourceKind === "label_set_revision" ? (labelSets.data?.items ?? []).flatMap((item: LabelSet) => item.latest_revision_id ? [{ value: item.latest_revision_id, label: item.name, description: "Published human-reference label set" }] : []) : (suites.data?.items ?? []).flatMap((suite: BenchmarkSuite) => suite.latest_revision && !["operational", "holdout", "final_holdout", "test", "canary"].includes(suite.purpose || "") ? [{ value: suite.latest_revision.id, label: suite.name, description: `${suite.purpose || "unspecified"} · ${suite.latest_revision.items.length} items` }] : []);
  const selectedCalibration = selectedCalibrationId ? calibrations.data?.items.find((item) => item.id === selectedCalibrationId) : calibrations.data?.items[0];
  useEffect(() => { if (!selectedCalibrationId && selectedCalibration) onCalibration(selectedCalibration.id); }, [onCalibration, selectedCalibration, selectedCalibrationId]);
  useEffect(() => { if (calibrations.isSuccess && !calibrations.data.items.length) setLaunchOpen(true); }, [calibrations.data?.items.length, calibrations.isSuccess]);

  return <SplitWorkspace rail={<><RailHeader eyebrow="CALIBRATIONS" title={`${calibrations.data?.total ?? calibrations.data?.items.length ?? 0} runs`} action={<Button size="icon" variant="ghost" onClick={() => setLaunchOpen(true)} aria-label="Launch calibration"><Plus /></Button>} /><div className="min-h-0 flex-1 overflow-y-auto">{calibrations.isLoading ? <Loading label="Loading calibrations" /> : calibrations.isError ? <ServiceUnavailable label="Calibration service is unavailable" /> : calibrations.data?.items.length ? calibrations.data.items.map((item) => <CalibrationListItem key={item.id} item={item} selected={item.id === selectedCalibration?.id} onSelect={() => { onCalibration(item.id); setLaunchOpen(false); }} />) : <EmptyState icon={<TestTube2 />} title="No calibrations" detail="Launch a replicated calibration against human-reference evidence." />}</div></>} main={launchOpen ? <CalibrationLauncher profiles={profiles.data?.items ?? []} protocols={protocols.data?.items ?? []} qualificationProfiles={qualificationProfiles.data?.items ?? []} profileId={profileId} onProfileId={setProfileId} sourceKind={sourceKind} onSourceKind={(value) => { setSourceKind(value); setSourceId(""); }} sourceId={sourceId} onSourceId={setSourceId} sourceOptions={sourceOptions} protocolId={protocolId} onProtocolId={setProtocolId} qualificationId={qualificationId} onQualificationId={setQualificationId} confirmation={confirmation} onConfirmation={setConfirmation} pairwiseOrder={pairwiseOrder} onPairwiseOrder={setPairwiseOrder} paraphraseProbe={paraphraseProbe} onParaphraseProbe={setParaphraseProbe} onLaunch={() => launch.mutate()} onCancel={() => setLaunchOpen(false)} pending={launch.isPending} error={launch.error?.message} /> : selectedCalibration ? <CalibrationDetail calibration={selectedCalibration} onCompare={onCompare} /> : <EmptyState icon={<TestTube2 />} title="Select a calibration" detail="Inspect replicated evidence, metrics, and exact qualification reasons." />} />;
}

function CalibrationLauncher({ profiles, protocols, qualificationProfiles, profileId, onProfileId, sourceKind, onSourceKind, sourceId, onSourceId, sourceOptions, protocolId, onProtocolId, qualificationId, onQualificationId, confirmation, onConfirmation, pairwiseOrder, onPairwiseOrder, paraphraseProbe, onParaphraseProbe, onLaunch, onCancel, pending, error }: { profiles: VerifierProfile[]; protocols: VerifierCalibrationProtocolRevision[]; qualificationProfiles: VerifierQualificationProfileRevision[]; profileId: string; onProfileId: (id: string) => void; sourceKind: "label_set_revision" | "benchmark_suite_revision"; onSourceKind: (kind: "label_set_revision" | "benchmark_suite_revision") => void; sourceId: string; onSourceId: (id: string) => void; sourceOptions: Array<{ value: string; label: string; description?: string }>; protocolId: string; onProtocolId: (id: string) => void; qualificationId: string; onQualificationId: (id: string) => void; confirmation: boolean; onConfirmation: (value: boolean) => void; pairwiseOrder: boolean; onPairwiseOrder: (value: boolean) => void; paraphraseProbe: boolean; onParaphraseProbe: (value: boolean) => void; onLaunch: () => void; onCancel: () => void; pending: boolean; error?: string }) {
  const profile = profiles.find((item) => item.id === profileId);
  const protocol = protocols.find((item) => item.id === protocolId);
  const qualification = qualificationProfiles.find((item) => item.id === qualificationId);
  return <div className="mx-auto max-w-5xl p-5 lg:p-8"><section className="border-b border-border pb-5"><div className="text-[9.5px] uppercase tracking-[0.14em] text-accent">GUIDED CALIBRATION</div><h2 className="mt-1 text-lg font-semibold text-fg">Measure reliability against human-reference evidence</h2><p className="mt-1 text-[10.5px] text-fg-subtle">The protocol is replicated, restart-safe, and never tunes the verifier automatically.</p></section><div className="grid gap-7 py-6 lg:grid-cols-[minmax(0,1fr)_300px]"><div className="space-y-5"><Field label="1 · Human reference"><div className="grid grid-cols-[150px_minmax(0,1fr)] gap-2"><select value={sourceKind} onChange={(event) => onSourceKind(event.target.value as typeof sourceKind)} className={selectClass}><option value="label_set_revision">Published label set</option><option value="benchmark_suite_revision">Development suite</option></select><SearchPicker value={sourceId} onChange={onSourceId} options={sourceOptions} placeholder="Choose eligible evidence" emptyLabel="No eligible reference evidence" /></div></Field><Field label="2 · Verifier profile"><SearchPicker value={profileId} onChange={onProfileId} options={profiles.map((item) => ({ value: item.id, label: item.name, description: item.latest_revision ? `${familyLabel(item.latest_revision.family)} · revision ${item.latest_revision.revision_number}` : "No published revision", status: item.latest_revision?.qualification_state }))} placeholder="Choose an immutable verifier profile" /></Field><div className="grid gap-4 md:grid-cols-2"><Field label="3 · Replication protocol"><SearchPicker value={protocolId} onChange={onProtocolId} options={protocols.map((item) => ({ value: item.id, label: item.name, description: `${item.repeats ?? "family default"} repeats · ${item.bootstrap_resamples ?? 10_000} bootstrap draws` }))} placeholder="Choose protocol" /></Field><Field label="4 · Qualification policy"><SearchPicker value={qualificationId} onChange={onQualificationId} options={qualificationProfiles.map((item) => ({ value: item.id, label: item.name, description: item.promotable === false ? "Reporting only · cannot promote" : "Pass, warn, and fail gates" }))} placeholder="Choose qualification policy" /></Field></div><Field label="Perturbations"><div className="divide-y divide-border-subtle rounded-md border border-border"><ToggleRow label="Grouped 70/30 confirmation" detail="Seed 42; shared records and media stay together." checked={confirmation} onChange={onConfirmation} /><ToggleRow label="Order counterbalancing" detail="A/B and B/A; rankings use deterministic rotations." checked={pairwiseOrder} onChange={onPairwiseOrder} /><ToggleRow label="Reviewed paraphrase probes" detail="Runs only when the source includes explicit reviewed variants." checked={paraphraseProbe} onChange={onParaphraseProbe} /></div></Field></div><aside className="border-l border-border-subtle pl-5"><InspectorTitle>Request preview</InspectorTitle><DefinitionList values={{ verifier: profile?.name || "not selected", family: profile?.latest_revision ? familyLabel(profile.latest_revision.family) : "—", source: sourceOptions.find((item) => item.value === sourceId)?.label || "not selected", repeats: protocol?.repeats ?? "family default", seeds: protocol?.seeds?.join(", ") || "17, 42, 101", bootstrap: protocol?.bootstrap_resamples ?? 10_000, qualification: qualification?.name || "not selected", lease: profile?.latest_revision?.family === "llm_judge" || profile?.latest_revision?.family === "reward_model" ? "accelerator or hosted provider" : "CPU" }} /><Notice tone="neutral" title="No automatic tuning">Threshold curves are evidence. Applying a suggested change creates a new profile revision and a new calibration.</Notice></aside></div><div className="flex items-center justify-between border-t border-border pt-4"><Button size="sm" variant="ghost" onClick={onCancel}>Cancel</Button><div className="flex items-center gap-3">{error ? <span role="alert" className="text-[10px] text-danger">{error}</span> : null}<Button size="sm" variant="primary" onClick={onLaunch} disabled={!profile?.latest_revision || !sourceId || !protocolId || !qualificationId || pending}>{pending ? <Loader2 className="animate-spin" /> : <Play />}Launch calibration</Button></div></div></div>;
}

function CalibrationListItem({ item, selected, onSelect }: { item: VerifierCalibration; selected: boolean; onSelect: () => void }) {
  return <button type="button" onClick={onSelect} className={cn("w-full border-b border-border-subtle px-4 py-3 text-left transition-colors", selected ? "bg-accent/7" : "hover:bg-surface/50")}><div className="flex items-center justify-between gap-2"><span className={cn("truncate text-[11px] font-medium", selected ? "text-accent" : "text-fg")}>{item.source_name || item.profile_revision?.profile_id || "Verifier calibration"}</span><StatusBadge status={item.qualification?.decision || item.status} /></div><div className="mt-1 flex items-center gap-2 text-[9px] text-fg-disabled"><span>{humanize(item.status)}</span><span>·</span><span>{item.processed_records ?? 0}/{item.total_records ?? "—"}</span></div>{["queued", "running"].includes(item.status) ? <Progress value={item.progress_percent} /> : null}</button>;
}

function CalibrationDetail({ calibration, onCompare }: { calibration: VerifierCalibration; onCompare: () => void }) {
  const live = useQuery({ queryKey: ["verifier-calibration", calibration.id], queryFn: () => api.verifierCalibration(calibration.id), initialData: calibration, refetchInterval: ["queued", "running"].includes(calibration.status) ? 3_000 : false, retry: false });
  const metrics = useQuery({ queryKey: ["verifier-calibration-metrics", calibration.id], queryFn: () => api.verifierCalibrationMetrics(calibration.id), enabled: live.data?.status === "completed", retry: false });
  const [sampleOffset, setSampleOffset] = useState(0);
  const [sampleOutcome, setSampleOutcome] = useState("");
  const [sampleQuery, setSampleQuery] = useState("");
  const samples = useQuery({ queryKey: ["verifier-calibration-samples", calibration.id, sampleOffset, sampleOutcome, sampleQuery], queryFn: () => api.verifierCalibrationSamples(calibration.id, { offset: sampleOffset, limit: 50, outcome: sampleOutcome || undefined, q: sampleQuery || undefined }), enabled: live.data?.status === "completed", retry: false });
  const current = live.data ?? calibration;
  const primary = current.primary_metric ?? metrics.data?.items.find((item) => item.value !== null);
  return <div className="min-w-0"><section className="border-b border-border px-5 py-5 lg:px-8"><div className="flex flex-wrap items-start justify-between gap-4"><div><div className="flex items-center gap-2"><span className="text-[9.5px] uppercase tracking-[0.14em] text-accent">CALIBRATION EVIDENCE</span><StatusBadge status={current.qualification?.decision || current.status} /></div><h2 className="mt-2 text-lg font-semibold text-fg">{current.source_name || "Verifier reliability run"}</h2><div className="mt-1 font-mono text-[8.5px] text-fg-disabled">{current.id}</div></div><div className="flex flex-wrap items-center gap-2">{current.status === "completed" ? <Button size="sm" variant="ghost" asChild><Link to="/datasets/review" search={{ new: "1", source: "verifier_calibration", sourceRef: current.id, baseRef: undefined }}><ShieldCheck />Open Review Proposal</Link></Button> : null}<Button size="sm" variant="secondary" onClick={onCompare} disabled={current.status !== "completed"}><GitCompareArrows />Compare</Button></div></div>{["queued", "running"].includes(current.status) ? <div className="mt-5 max-w-xl"><div className="mb-2 flex justify-between text-[9.5px] text-fg-subtle"><span>{current.stage || "Preparing calibration"}</span><span>{current.processed_records ?? 0} / {current.total_records ?? "—"}</span></div><Progress value={current.progress_percent} /></div> : null}</section>
    {current.status === "completed" ? <><section className="grid grid-cols-2 gap-px border-b border-border bg-border md:grid-cols-4"><EvidenceMetric label={primary?.name || "Primary agreement"} value={formatMetric(primary?.value)} detail={metricInterval(primary)} /><EvidenceMetric label="Coverage" value={formatMetric(metricValue(metrics.data?.items, "coverage"))} detail="valid observations" /><EvidenceMetric label="Error rate" value={formatMetric(metricValue(metrics.data?.items, "error_rate"))} detail="parser + runtime" inverse /><EvidenceMetric label="Repeat agreement" value={formatMetric(metricValue(metrics.data?.items, "repeat_agreement"))} detail="stable-record unit" /></section><CalibrationDiagnostics metrics={metrics.data?.items ?? []} /><section className="grid gap-0 xl:grid-cols-[minmax(0,1fr)_320px]"><div className="min-w-0 border-b border-border xl:border-b-0 xl:border-r"><div className="flex flex-wrap items-end justify-between gap-3 border-b border-border-subtle px-5 py-4"><SectionTitle eyebrow="SAMPLES" title="Human reference and verifier observations" detail={`${samples.data?.total ?? 0} stable records · server filtered`} /><div className="flex gap-2"><SearchField value={sampleQuery} onChange={(value) => { setSampleQuery(value); setSampleOffset(0); }} placeholder="Find record" compact /><select value={sampleOutcome} onChange={(event) => { setSampleOutcome(event.target.value); setSampleOffset(0); }} className={selectClass}><option value="">All outcomes</option><option value="false_accept">False accepts</option><option value="false_reject">False rejects</option><option value="repeat_flip">Repeat flips</option><option value="error">Errors</option></select></div></div><CalibrationSampleTable items={samples.data?.items ?? []} loading={samples.isLoading} /><Pager total={samples.data?.total ?? 0} offset={sampleOffset} limit={50} onOffset={setSampleOffset} /></div><aside className="p-5"><InspectorTitle>Qualification reasons</InspectorTitle><QualificationReasons decision={current.qualification} /><QualificationAction calibration={current} /><div className="mt-6"><InspectorTitle>Runtime identity</InspectorTitle><DefinitionList values={{ evidence_hash: current.evidence_hash || "—", runtime_hash: current.runtime_hash || "—", request_hash: current.request_hash || "—", finished: formatDate(current.completed_at) }} /></div></aside></section></> : current.status === "failed" ? <EmptyState icon={<XCircle />} title="Calibration failed" detail={current.error || "Open Activity to inspect the failed attempt and retry."} /> : <EmptyState icon={<Clock3 />} title="Calibration is in progress" detail="Progress, retries, and resource ownership are available in Activity." />}
  </div>;
}

function QualificationAction({ calibration }: { calibration: VerifierCalibration }) {
  const queryClient = useQueryClient();
  const [scope, setScope] = useState<"development" | "operational" | "confirmation">("development");
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [overrideNote, setOverrideNote] = useState("");
  const qualify = useMutation({
    mutationFn: () => api.qualifyVerifierCalibration(calibration.id, { scope, overrideNote: overrideOpen ? overrideNote.trim() || undefined : undefined }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["verifier-calibration", calibration.id] });
      queryClient.invalidateQueries({ queryKey: ["verifier-qualification-decisions"] });
      setOverrideNote("");
      setOverrideOpen(false);
    },
  });
  return <div className="mt-5 border-t border-border-subtle pt-4"><div className="mb-2 text-[8.5px] uppercase tracking-[0.11em] text-fg-disabled">Append qualification decision</div><select value={scope} onChange={(event) => setScope(event.target.value as typeof scope)} className={selectClass}><option value="development">Development evidence</option><option value="operational">Operational evidence</option><option value="confirmation">Confirmation evidence</option></select><details className="mt-2" open={overrideOpen} onToggle={(event) => setOverrideOpen(event.currentTarget.open)}><summary className="cursor-pointer text-[8.5px] uppercase tracking-wider text-fg-disabled">Advanced override</summary><Input className="mt-2" value={overrideNote} onChange={(event) => setOverrideNote(event.target.value)} placeholder="Required override reason" /></details><Button className="mt-2 w-full" size="sm" variant="secondary" onClick={() => qualify.mutate()} disabled={qualify.isPending || (overrideOpen && !overrideNote.trim())}>{qualify.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Record {scope} decision</Button>{qualify.error ? <p role="alert" className="mt-2 text-[9px] text-danger">{qualify.error.message}</p> : null}<p className="mt-2 text-[8.5px] leading-4 text-fg-disabled">This appends a policy decision; it never changes the verifier threshold or rubric.</p></div>;
}

function CalibrationSampleTable({ items, loading }: { items: VerifierCalibrationSample[]; loading: boolean }) {
  if (loading) return <Loading label="Loading calibration evidence" />;
  if (!items.length) return <EmptyState icon={<CircleDot />} title="No samples match" detail="Change the server-side filter to inspect other records." />;
  return <div className="overflow-x-auto"><table className="w-full min-w-[760px] text-left text-[10px]"><thead><tr className="border-b border-border-subtle"><Th>Record</Th><Th>Split</Th><Th>Expected</Th><Th>Observed</Th><Th>Reward</Th><Th>Repeat</Th><Th>Evidence</Th></tr></thead><tbody>{items.map((item) => <CalibrationSampleRow key={item.id} item={item} />)}</tbody></table></div>;
}

function CalibrationDiagnostics({ metrics }: { metrics: VerifierCalibrationMetric[] }) {
  const order = metrics.find((item) => ["order_consistency", "order_flip_rate"].includes(item.name));
  const repeat = metrics.find((item) => ["repeat_drift", "repeat_agreement", "pass_flip_rate"].includes(item.name));
  const primary = metrics.find((item) => item.details?.primary === true) ?? metrics.find((item) => item.value !== null);
  const diagnosticDetails = primary?.details ?? {};
  const confusion = diagnosticDetails.confusion_matrix;
  const perClass = diagnosticDetails.per_class ?? diagnosticDetails.per_label;
  const curve = (Array.isArray(diagnosticDetails.threshold_curve) ? diagnosticDetails.threshold_curve : []) as Array<{ threshold?: number; accuracy?: number; primary?: number; false_accept_rate?: number; false_reject_rate?: number }>;
  return <section className="grid gap-px border-b border-border bg-border lg:grid-cols-4"><DiagnosticPanel title="Threshold curve" icon={<SlidersHorizontal />}>{curve.length ? <><div className="flex h-14 items-end gap-1" aria-label="Threshold curve">{curve.slice(0, 24).map((point, index) => { const score = Number(point.accuracy ?? point.primary ?? 0); return <span key={index} className="min-w-1 flex-1 bg-accent/70" style={{ height: `${Math.max(3, Math.min(100, score * 100))}%` }} title={`threshold ${point.threshold ?? "—"} · accuracy ${point.accuracy ?? point.primary ?? "—"} · false accept ${point.false_accept_rate ?? "—"} · false reject ${point.false_reject_rate ?? "—"}`} />; })}</div><p className="mt-1 text-[8px] text-fg-disabled">Report only · no threshold is applied automatically.</p></> : <UnavailableEvidence label="No threshold curve was reported for the primary metric." />}</DiagnosticPanel><DiagnosticPanel title="Confusion matrix" icon={<BarChart3 />}>{confusion ? <ConfusionEvidence matrix={confusion} perClass={perClass} /> : <UnavailableEvidence label="Unavailable for this task or adapter." />}</DiagnosticPanel><DiagnosticPanel title="Order diagnostics" icon={<GitCompareArrows />}>{order ? <DiagnosticValue metric={order} /> : <UnavailableEvidence label="No compatible pair or ranking orientations." />}</DiagnosticPanel><DiagnosticPanel title="Repeat diagnostics" icon={<RefreshCw />}>{repeat ? <DiagnosticValue metric={repeat} /> : <UnavailableEvidence label="Repeat evidence was not reported." />}</DiagnosticPanel></section>;
}

function ConfusionEvidence({ matrix, perClass }: { matrix: unknown; perClass: unknown }) {
  return <div className="space-y-1.5"><pre className="max-h-14 overflow-auto whitespace-pre-wrap font-mono text-[8.5px] leading-4 text-fg-subtle">{JSON.stringify(matrix, null, 2)}</pre>{perClass ? <details><summary className="cursor-pointer text-[8px] uppercase tracking-wider text-accent">Per-class evidence</summary><pre className="mt-1 max-h-20 overflow-auto whitespace-pre-wrap font-mono text-[8px] leading-4 text-fg-disabled">{JSON.stringify(perClass, null, 2)}</pre></details> : null}</div>;
}

function DiagnosticPanel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) { return <div className="min-h-24 bg-bg px-4 py-3"><div className="mb-2 flex items-center gap-2 text-[8.5px] uppercase tracking-[0.11em] text-fg-disabled"><span className="text-accent [&_svg]:h-3.5 [&_svg]:w-3.5">{icon}</span>{title}</div>{children}</div>; }
function UnavailableEvidence({ label }: { label: string }) { return <p className="text-[8.5px] leading-4 text-fg-disabled">{label}</p>; }
function DiagnosticValue({ metric }: { metric: VerifierCalibrationMetric }) { return <div><div className="font-mono text-[15px] text-fg">{formatMetric(metric.value)}</div><div className="mt-1 text-[8.5px] text-fg-disabled">{humanize(metric.name)} · {metricInterval(metric)}</div></div>; }

function CalibrationSampleRow({ item }: { item: VerifierCalibrationSample }) {
  const [open, setOpen] = useState(false);
  const trace = item.observation.component_trace ?? [];
  return <><tr className="border-b border-border-subtle align-top"><Td mono>{item.record_id}</Td><Td>{humanize(item.split || "calibration")}</Td><Td>{compact(item.expected)}</Td><Td>{item.observation.error ? <span className="text-danger">{item.observation.error}</span> : compact(item.observation.parsed_value ?? item.observation.passed)}</Td><Td mono>{typeof item.observation.reward === "number" ? item.observation.reward.toFixed(4) : "—"}</Td><Td mono>{item.repeat_index ?? 0}{item.seed !== null && item.seed !== undefined ? ` · seed ${item.seed}` : ""}</Td><Td><button type="button" className="inline-flex items-center gap-1 text-accent hover:underline" onClick={() => setOpen(!open)}>{trace.length ? `${trace.length} components` : "Observation"}<ChevronRight className={cn("h-3 w-3 transition-transform", open && "rotate-90")} /></button></Td></tr>{open ? <tr className="border-b border-border-subtle bg-bg-subtle/25"><td colSpan={7} className="px-4 py-3"><SampleInputEvidence input={item.input} /><div className="mt-3 grid gap-3 lg:grid-cols-3"><TraceBlock label="Raw output" value={item.observation.raw_output} /><TraceBlock label="Details" value={item.observation.details} /><TraceBlock label="Component trace" value={trace} /></div></td></tr> : null}</>;
}

function SampleInputEvidence({ input }: { input: unknown }) { if (!input || typeof input !== "object" || Array.isArray(input)) return <TraceBlock label="Input" value={input} />; const record = input as Record<string, unknown>; const image = firstString(record, ["image_url", "image", "image_reference"]); const audio = firstString(record, ["audio_url", "audio", "audio_reference"]); const tools = record.tools ?? record.tool_definitions ?? record.tool_calls; return <div className="grid gap-3 lg:grid-cols-3">{image ? <figure><div className="mb-1 text-[8.5px] uppercase tracking-wider text-fg-disabled">Image evidence</div><img src={image} alt={String(record.image_alt ?? record.alt ?? "Calibration image")} className="max-h-40 w-full rounded border border-border bg-black/20 object-contain" /><figcaption className="mt-1 font-mono text-[8px] text-fg-disabled">{String(record.width ?? "?")} × {String(record.height ?? "?")}</figcaption></figure> : null}{audio ? <figure><div className="mb-1 text-[8.5px] uppercase tracking-wider text-fg-disabled">Audio evidence</div><audio controls preload="metadata" src={audio} className="w-full">Audio evidence is unavailable.</audio><figcaption className="mt-1 font-mono text-[8px] text-fg-disabled">{String(record.duration_seconds ?? record.duration ?? "?")} sec · {String(record.sample_rate ?? "?")} Hz</figcaption></figure> : null}{tools ? <TraceBlock label="Tool trace" value={tools} /> : null}{!image && !audio && !tools ? <TraceBlock label="Input" value={input} /> : null}</div>; }

function CalibrationCompareView({ selectedCalibrationId }: { selectedCalibrationId?: string }) {
  const calibrations = useQuery({ queryKey: ["verifier-calibrations", "compare"], queryFn: () => api.listVerifierCalibrations({ status: "completed", limit: 200 }), retry: false });
  const completed = calibrations.data?.items ?? [];
  const [baseId, setBaseId] = useState("");
  const [candidateId, setCandidateId] = useState(selectedCalibrationId || "");
  useEffect(() => { if (!candidateId && completed[0]) setCandidateId(completed[0].id); }, [candidateId, completed]);
  useEffect(() => { if (!baseId && completed.find((item) => item.id !== candidateId)) setBaseId(completed.find((item) => item.id !== candidateId)!.id); }, [baseId, candidateId, completed]);
  const comparison = useQuery({ queryKey: ["verifier-calibration-comparison", baseId, candidateId], queryFn: () => api.compareVerifierCalibrations(baseId, candidateId), enabled: Boolean(baseId && candidateId && baseId !== candidateId), retry: false });
  const options = completed.map((item) => ({ value: item.id, label: item.source_name || item.profile_revision?.profile_id || "Completed calibration", description: `${item.qualification?.decision || "unqualified"} · ${formatDate(item.completed_at)}` }));
  return <div className="mx-auto max-w-6xl p-5 lg:p-8"><section className="border-b border-border pb-6"><SectionTitle eyebrow="CALIBRATION DELTAS" title="Compare exact profile revisions" detail="Comparisons require compatible sources, protocols, task contracts, and qualification scopes." /><div className="mt-5 grid items-end gap-3 md:grid-cols-[1fr_24px_1fr]"><Field label="Base calibration"><SearchPicker value={baseId} onChange={setBaseId} options={options} placeholder="Choose base evidence" /></Field><GitCompareArrows className="mb-2 h-4 w-4 text-fg-disabled" /><Field label="Candidate calibration"><SearchPicker value={candidateId} onChange={setCandidateId} options={options} placeholder="Choose candidate evidence" /></Field></div></section>{comparison.isLoading ? <Loading label="Joining calibration evidence" /> : comparison.data ? <ComparisonEvidence data={comparison.data} /> : comparison.isError ? <ServiceUnavailable label={comparison.error.message} /> : <EmptyState icon={<GitCompareArrows />} title="Choose two calibrations" detail="Metric deltas preserve direction and sample counts." />}</div>;
}

function ComparisonEvidence({ data }: { data: VerifierCalibrationComparison }) {
  return <div className="py-6">{!data.compatible ? <Notice tone="danger" title="These calibrations are not comparable">{data.compatibility_reasons?.join(" ") || "Their immutable protocols or evidence sources differ."}</Notice> : null}<div className="overflow-hidden rounded-md border border-border"><table className="w-full text-left text-[10px]"><thead><tr className="border-b border-border-subtle"><Th>Metric</Th><Th>Base</Th><Th>Candidate</Th><Th>Delta</Th><Th>Direction-aware</Th></tr></thead><tbody>{data.metrics.map((metric) => <tr key={metric.name} className="border-b border-border-subtle last:border-0"><Td>{humanize(metric.name)}</Td><Td mono>{formatMetric(metric.base_value)}</Td><Td mono>{formatMetric(metric.candidate_value)}</Td><Td mono>{formatSigned(metric.raw_delta)}</Td><Td mono className={typeof metric.favorable_delta === "number" ? metric.favorable_delta > 0 ? "text-success" : metric.favorable_delta < 0 ? "text-danger" : "" : ""}>{formatSigned(metric.favorable_delta)}</Td></tr>)}</tbody></table></div></div>;
}

function QualificationView({ selectedProfileId, onProfile }: { selectedProfileId?: string; onProfile: (id?: string) => void }) {
  const queryClient = useQueryClient();
  const profiles = useQuery({ queryKey: ["verifier-profiles", "qualification"], queryFn: () => api.listVerifierProfiles({ limit: 200 }), retry: false });
  const profileId = selectedProfileId || profiles.data?.items[0]?.id || "";
  const profile = profiles.data?.items.find((item) => item.id === profileId);
  const revision = profile?.latest_revision;
  const decisions = useQuery({ queryKey: ["verifier-qualification-decisions", revision?.id], queryFn: () => api.listVerifierQualificationDecisions({ profileRevisionId: revision!.id, limit: 100 }), enabled: Boolean(revision?.id), retry: false });
  const runtime = useQuery({ queryKey: ["verifier-runtime-compatibility", revision?.id], queryFn: () => api.verifierRuntimeCompatibility(revision!.id), enabled: Boolean(revision?.id), retry: false });
  const usage = useQuery({ queryKey: ["verifier-usage", revision?.id], queryFn: () => api.verifierRevisionUsage(revision!.id), enabled: Boolean(revision?.id), retry: false });
  const [alias, setAlias] = useState<"candidate" | "approved">("candidate");
  const [note, setNote] = useState("");
  const [override, setOverride] = useState(false);
  const passedScopes = new Set((decisions.data?.items ?? []).filter((item) => item.decision === "pass" && !item.override).map((item) => item.scope || "development"));
  const promotionEligible = runtime.data?.compatible === true && passedScopes.has("development") && passedScopes.has("operational") && (alias !== "approved" || passedScopes.has("confirmation"));
  const missingGates = [!passedScopes.has("development") ? "development pass" : null, !passedScopes.has("operational") ? "operational pass" : null, alias === "approved" && !passedScopes.has("confirmation") ? "confirmation pass" : null, runtime.data?.compatible !== true ? "compatible runtime" : null].filter(Boolean).join(", ");
  const promote = useMutation({ mutationFn: () => api.promoteVerifierRevision(revision!.id, { alias, note: note || undefined, override }), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["verifier-profiles"] }); setNote(""); setOverride(false); } });
  const rail = <><RailHeader eyebrow="QUALIFICATION" title="Verifier revisions" /><div className="min-h-0 flex-1 overflow-y-auto">{profiles.data?.items.map((item) => <ProfileListItem key={item.id} profile={item} selected={item.id === profileId} onSelect={() => onProfile(item.id)} />)}</div></>;
  if (!revision) return <SplitWorkspace rail={rail} main={<EmptyState icon={<ShieldCheck />} title="Select a verifier revision" detail="Inspect decisions, runtime scope, usage, and promotion gates." />} />;
  return <SplitWorkspace rail={rail} main={<div className="mx-auto max-w-5xl p-5 lg:p-8">
    <section className="border-b border-border pb-6"><div className="flex flex-wrap items-start justify-between gap-4"><div><div className="flex items-center gap-2"><span className="text-[9.5px] uppercase tracking-[0.14em] text-accent">PROMOTION GATES</span><QualificationBadge state={revision.qualification_state || "unqualified"} /></div><h2 className="mt-2 text-lg font-semibold text-fg">{profile?.name}</h2><p className="mt-1 text-[10.5px] text-fg-subtle">Promotion is explicit, append-only, and scoped to the exact runtime contract.</p></div><div className="font-mono text-[8.5px] text-fg-disabled">revision {revision.revision_number}</div></div></section>
    <section className="grid gap-7 py-6 lg:grid-cols-[minmax(0,1fr)_320px]"><div><InspectorTitle>Qualification history</InspectorTitle>{decisions.data?.items.length ? <ol className="mt-3 space-y-3">{decisions.data.items.map((decision) => <QualificationDecisionRow key={decision.id} decision={decision} />)}</ol> : <EmptyState icon={<ShieldCheck />} title="No qualification decision" detail="Complete a promotable calibration to evaluate this revision." />}</div><aside className="border-l border-border-subtle pl-5"><InspectorTitle>Runtime compatibility</InspectorTitle>{runtime.data ? <><ReadinessRow label="Implementation" ready={runtime.data.compatible} pendingLabel={runtime.data.status} />{runtime.data.differences?.map((item) => <div key={item.field} className="mt-2 text-[9px] text-warning">{humanize(item.field)} changed</div>)}</> : <p className="text-[9.5px] text-fg-disabled">Compatibility has not been checked.</p>}
      <div className="mt-6"><InspectorTitle>Promote exact revision</InspectorTitle><Field label="Alias"><select value={alias} onChange={(event) => setAlias(event.target.value as typeof alias)} className={selectClass}><option value="candidate">candidate · development + operational pass</option><option value="approved">approved · confirmation pass</option></select></Field><Field label="Promotion note"><Input value={note} onChange={(event) => setNote(event.target.value)} placeholder="Why this revision is ready" /></Field>{!promotionEligible ? <p className="mb-2 text-[9px] leading-4 text-warning">Missing gates: {missingGates || "qualification evidence"}</p> : null}<details className="mb-2"><summary className="cursor-pointer text-[8.5px] uppercase tracking-wider text-fg-disabled">Advanced override</summary><label className="mt-2 flex items-start gap-2 text-[9px] leading-4 text-warning"><input type="checkbox" checked={override} onChange={(event) => setOverride(event.target.checked)} />Record an override. This revision remains excluded from normal guided pickers.</label></details><Button className="mt-2 w-full" size="sm" variant="primary" onClick={() => promote.mutate()} disabled={promote.isPending || !note.trim() || (!promotionEligible && !override)}>{promote.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Promote revision</Button>{promote.error ? <p role="alert" className="mt-2 text-[9.5px] text-danger">{promote.error.message}</p> : null}</div>
    </aside></section>
    <section className="border-t border-border pt-6"><InspectorTitle>Linked use</InspectorTitle><div className="mt-3 divide-y divide-border-subtle border-y border-border-subtle">{usage.data?.items.map((item) => <div key={item.id} className="grid grid-cols-[120px_minmax(0,1fr)_auto] gap-3 py-2.5 text-[10px]"><span className="text-fg-disabled">{humanize(item.kind)}</span><span className="truncate text-fg">{item.label || item.id}</span><span className="text-fg-disabled">{item.role}</span></div>)}{usage.data && !usage.data.items.length ? <div className="py-5 text-center text-[10px] text-fg-disabled">This revision has no downstream bindings.</div> : null}</div></section>
  </div>} />;
}

function QualificationDecisionRow({ decision }: { decision: VerifierQualificationDecision }) {
  return <li className="border-l-2 border-border pl-4"><div className="flex flex-wrap items-center gap-2"><StatusBadge status={decision.decision} /><span className="text-[10px] font-medium text-fg">{humanize(decision.scope || "development")}</span><span className="font-mono text-[8.5px] text-fg-disabled">{formatDate(decision.created_at)}</span></div><ul className="mt-2 space-y-1">{decision.reasons.map((reason) => <li key={reason} className="flex gap-2 text-[9.5px] leading-4 text-fg-subtle"><span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-fg-disabled" />{reason}</li>)}</ul></li>;
}

function SplitWorkspace({ rail, main }: { rail: ReactNode; main: ReactNode }) {
  return <div className="grid min-h-[calc(100vh-152px)] lg:grid-cols-[290px_minmax(0,1fr)]"><aside className="flex max-h-80 min-h-0 flex-col border-b border-border bg-bg-subtle/30 lg:max-h-none lg:border-b-0 lg:border-r">{rail}</aside><main className="min-w-0 overflow-y-auto">{main}</main></div>;
}

function RailHeader({ eyebrow, title, action }: { eyebrow: string; title: string; action?: ReactNode }) { return <div className="flex items-center justify-between gap-3 border-b border-border-subtle px-4 py-3"><div><div className="text-[9px] uppercase tracking-[0.14em] text-fg-disabled">{eyebrow}</div><div className="mt-0.5 text-[11px] font-medium text-fg">{title}</div></div>{action}</div>; }
function SectionTitle({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) { return <div><div className="text-[9px] uppercase tracking-[0.14em] text-accent">{eyebrow}</div><h3 className="mt-1 text-[13px] font-medium text-fg">{title}</h3><p className="mt-1 max-w-2xl text-[10px] leading-4 text-fg-subtle">{detail}</p></div>; }
function InspectorTitle({ children }: { children: ReactNode }) { return <h3 className="mb-3 text-[9.5px] font-medium uppercase tracking-[0.12em] text-fg-muted">{children}</h3>; }
function Field({ label, children }: { label: string; children: ReactNode }) { return <label className="mb-3 block space-y-1"><span className="block text-[9px] uppercase tracking-[0.11em] text-fg-disabled">{label}</span>{children}</label>; }
function SearchField({ value, onChange, placeholder, compact = false }: { value: string; onChange: (value: string) => void; placeholder: string; compact?: boolean }) { return <div className={cn("relative", compact && "w-48")}><Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-disabled" /><Input aria-label={placeholder} value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} className="pl-8" /></div>; }
function DefinitionList({ values }: { values: Record<string, unknown> }) { return <dl className="mt-3 divide-y divide-border-subtle border-y border-border-subtle">{Object.entries(values).map(([key, value]) => <div key={key} className="grid grid-cols-[150px_minmax(0,1fr)] gap-3 py-2.5 text-[9.5px]"><dt className="text-fg-disabled">{humanize(key)}</dt><dd className="break-words font-mono text-fg-subtle">{compact(value)}</dd></div>)}</dl>; }
function ReadinessRow({ label, ready, pendingLabel = "not ready" }: { label: string; ready: boolean; pendingLabel?: string }) { return <div className="flex items-center justify-between border-b border-border-subtle py-2 text-[9.5px]"><span className="text-fg-subtle">{label}</span><span className={cn("inline-flex items-center gap-1", ready ? "text-success" : "text-fg-disabled")}>{ready ? <CheckCircle2 className="h-3 w-3" /> : <CircleDot className="h-3 w-3" />}{ready ? "ready" : pendingLabel}</span></div>; }
function ChoiceButton({ selected, label, onClick }: { selected: boolean; label: string; onClick: () => void }) { return <button type="button" onClick={onClick} className={cn("min-h-11 rounded-md border px-3 text-left text-[10.5px] transition-colors", selected ? "border-accent/60 bg-accent/7 text-accent" : "border-border bg-bg text-fg-subtle hover:border-border-strong hover:text-fg")}>{label}</button>; }
function ToggleRow({ label, detail, checked, onChange }: { label: string; detail: string; checked: boolean; onChange: (value: boolean) => void }) { return <label className="flex cursor-pointer items-center gap-3 px-3 py-3"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span className="min-w-0"><span className="block text-[10.5px] font-medium text-fg">{label}</span><span className="mt-0.5 block text-[9px] leading-4 text-fg-disabled">{detail}</span></span></label>; }
function Notice({ tone, title, children }: { tone: "neutral" | "warning" | "danger"; title: string; children: ReactNode }) { const Icon = tone === "danger" ? XCircle : tone === "warning" ? TriangleAlert : SlidersHorizontal; return <div className={cn("mt-5 border-l-2 px-3 py-2.5", tone === "danger" ? "border-danger bg-danger/5" : tone === "warning" ? "border-warning bg-warning/5" : "border-accent/35 bg-accent/5")}><div className="flex items-center gap-2 text-[10px] font-medium text-fg"><Icon className={cn("h-3.5 w-3.5", tone === "danger" ? "text-danger" : tone === "warning" ? "text-warning" : "text-accent")} />{title}</div><div className="mt-1 text-[9.5px] leading-4 text-fg-subtle">{children}</div></div>; }
function EmptyState({ icon, title, detail }: { icon: ReactNode; title: string; detail: string }) { return <div className="grid min-h-64 place-items-center px-5 py-10 text-center"><div><span className="mx-auto grid h-9 w-9 place-items-center rounded-full border border-border text-fg-disabled [&_svg]:h-4 [&_svg]:w-4">{icon}</span><h3 className="mt-3 text-[12px] font-medium text-fg">{title}</h3><p className="mx-auto mt-1 max-w-sm text-[9.5px] leading-4 text-fg-subtle">{detail}</p></div></div>; }
function ServiceUnavailable({ label }: { label: string }) { return <EmptyState icon={<AlertTriangle />} title="Service unavailable" detail={label} />; }
function Loading({ label }: { label: string }) { return <div className="flex min-h-32 items-center justify-center gap-2 px-5 py-10 text-[10px] text-fg-disabled"><Loader2 className="h-4 w-4 animate-spin text-accent" />{label}</div>; }
function OriginIcon({ origin }: { origin: VerifierCatalogEntry["origin"] }) { const Icon = origin === "builtin" ? ShieldCheck : origin === "user_plugin" ? Plug : Package; return <Icon className="h-3.5 w-3.5 shrink-0 text-fg-disabled" aria-label={origin.replace("_", " ")} />; }
function QualificationBadge({ state }: { state: string }) { const tone = state === "pass" || state === "approved" || state === "candidate" ? "success" : state === "warn" || state === "stale_runtime" ? "warning" : state === "fail" ? "danger" : "neutral"; return <Badge tone={tone} size="sm">{humanize(state)}</Badge>; }
function StatusBadge({ status }: { status: string }) { const tone = status === "completed" || status === "pass" ? "success" : status === "warn" || status === "cancelled" || status === "interrupted" ? "warning" : status === "failed" || status === "fail" ? "danger" : ["running", "queued"].includes(status) ? "accent" : "neutral"; return <Badge tone={tone} size="sm" dot>{humanize(status)}</Badge>; }
function Progress({ value }: { value?: number | null }) { const percent = Math.max(0, Math.min(100, value ?? 0)); return <div className="mt-2 h-1 overflow-hidden rounded-full bg-border"><div className="h-full bg-accent transition-[width]" style={{ width: `${percent}%` }} /></div>; }
function EvidenceMetric({ label, value, detail, inverse = false }: { label: string; value: string; detail: string; inverse?: boolean }) { return <div className="bg-bg px-4 py-4"><div className="font-mono text-lg text-fg">{value}</div><div className="mt-1 text-[9px] uppercase tracking-[0.11em] text-fg-disabled">{label}</div><div className="mt-0.5 text-[8.5px] text-fg-disabled">{inverse ? "lower is better · " : ""}{detail}</div></div>; }
function QualificationReasons({ decision }: { decision?: VerifierQualificationDecision | null }) { if (!decision) return <p className="text-[9.5px] leading-4 text-fg-disabled">No qualification decision is attached.</p>; return <div><StatusBadge status={decision.decision} /><ul className="mt-3 space-y-2">{decision.reasons.map((reason) => <li key={reason} className="flex gap-2 text-[9.5px] leading-4 text-fg-subtle"><span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-fg-disabled" />{reason}</li>)}</ul></div>; }
function TraceBlock({ label, value }: { label: string; value: unknown }) { return <div><div className="mb-1 text-[8.5px] uppercase tracking-wider text-fg-disabled">{label}</div><pre className="max-h-44 overflow-auto whitespace-pre-wrap rounded border border-border bg-bg p-2 font-mono text-[8.5px] leading-4 text-fg-subtle">{JSON.stringify(value ?? null, null, 2)}</pre></div>; }
function Pager({ total, offset, limit, onOffset }: { total: number; offset: number; limit: number; onOffset: (offset: number) => void }) { return <div className="flex items-center justify-between border-t border-border-subtle px-4 py-2"><span className="font-mono text-[8.5px] text-fg-disabled">{total ? `${offset + 1}–${Math.min(total, offset + limit)} of ${total}` : "0 records"}</span><div className="flex gap-1"><Button size="sm" variant="ghost" disabled={offset === 0} onClick={() => onOffset(Math.max(0, offset - limit))}>Previous</Button><Button size="sm" variant="ghost" disabled={offset + limit >= total} onClick={() => onOffset(offset + limit)}>Next</Button></div></div>; }
function Th({ children }: { children: ReactNode }) { return <th className="px-4 py-2 text-[8.5px] font-medium uppercase tracking-[0.11em] text-fg-disabled">{children}</th>; }
function Td({ children, mono = false, className }: { children: ReactNode; mono?: boolean; className?: string }) { return <td className={cn("max-w-56 px-4 py-2.5 text-fg-subtle", mono && "font-mono", className)}>{children}</td>; }

const selectClass = "h-9 w-full rounded-md border border-border bg-bg px-2 text-[10.5px] text-fg outline-none focus:border-accent";
const textareaClass = "w-full rounded-md border border-border bg-bg px-3 py-2 text-[10.5px] leading-5 text-fg outline-none focus:border-accent";
function familyLabel(family: VerifierFamily) { return family === "llm_judge" ? "LLM judge" : family === "reward_model" ? "Reward model" : family === "chain" ? "Verifier chain" : "Deterministic"; }
function humanize(value: string) { return value.replace(/[_-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function shortHash(value: string) { return value.length > 14 ? `${value.slice(0, 8)}…${value.slice(-5)}` : value; }
function isPinnedModelRevision(value: string) {
  const revision = value.trim();
  return /@[^\s@]{6,}$/.test(revision) || /sha256:[0-9a-f]{12,}/i.test(revision) || /(?:^|[-_.])20\d{2}[-_.]\d{2}[-_.]\d{2}(?:$|[-_.])/.test(revision) || /[-_.]\d{4,8}$/.test(revision);
}
function yesNo(value?: boolean) { return value === undefined ? "unknown" : value ? "yes" : "no"; }
function compact(value: unknown) { if (value === undefined || value === null || value === "") return "—"; if (typeof value === "string") return value; if (typeof value === "number" || typeof value === "boolean") return String(value); return JSON.stringify(value); }
function formatMetric(value?: number | null) { return typeof value === "number" ? value.toFixed(4) : "—"; }
function formatSigned(value?: number | null) { return typeof value === "number" ? `${value >= 0 ? "+" : ""}${value.toFixed(4)}` : "—"; }
function metricValue(items: VerifierCalibrationMetric[] | undefined, name: string) { return items?.find((item) => item.name === name || item.name.endsWith(`.${name}`))?.value ?? null; }
function metricInterval(metric?: VerifierCalibrationMetric | null) { return metric && typeof metric.lower_ci === "number" && typeof metric.upper_ci === "number" ? `95% CI ${metric.lower_ci.toFixed(3)}–${metric.upper_ci.toFixed(3)}` : "interval unavailable"; }
function formatDate(value?: string | null) { if (!value) return "—"; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date); }
function firstString(record: Record<string, unknown>, keys: string[]): string | null { for (const key of keys) { const value = record[key]; if (typeof value === "string" && value.trim()) return value; } return null; }
