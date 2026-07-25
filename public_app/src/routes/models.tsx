import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  ArrowRight,
  Box,
  Boxes,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  Copy,
  Download,
  ExternalLink,
  FileArchive,
  GitBranch,
  HardDrive,
  Loader2,
  Merge,
  MessageSquare,
  PackageCheck,
  Pin,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  Square,
  Tag,
  Trash2,
  X,
} from "lucide-react";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ApiError,
  api,
  type ArtifactLocation,
  type ArtifactOperation,
  type ArtifactQualification,
  type BenchmarkSuite,
  type CleanupPlan,
  type ModelArtifactOccurrence,
  type ModelCatalogEntry,
  type PlaygroundMessage,
  type PlaygroundReviewPairing,
  type PlaygroundSession,
  type PlaygroundSessionMessage,
  type QualificationComparison,
  type StorageInventory,
} from "@/lib/api";
import {
  queryKeys,
  useArtifactOperations,
  useArtifactQualifications,
  useBackendInfo,
  useModelArtifacts,
  useModelCatalog,
  useServeStart,
  useServeStatus,
  useServeStop,
  useStorageInventory,
} from "@/lib/hooks";
import { cn, relativeTime } from "@/lib/utils";

type ModelTab = "catalog" | "artifacts" | "cached" | "serve";

const MODEL_TABS: Array<{ id: ModelTab; label: string; description: string }> = [
  { id: "catalog", label: "Catalog", description: "Compatible base models" },
  { id: "artifacts", label: "Trained Artifacts", description: "Transform and qualify" },
  { id: "cached", label: "Cached Models", description: "Local storage" },
  { id: "serve", label: "Serve & Test", description: "Managed inference" },
];

export const Route = createFileRoute("/models")({
  validateSearch: (search: Record<string, unknown>): { tab?: ModelTab; artifact?: string } => ({
    tab: isModelTab(search.tab) ? search.tab : "catalog",
    artifact: typeof search.artifact === "string" ? search.artifact : undefined,
  }),
  component: ModelsRoute,
});

function ModelsRoute() {
  const tab = Route.useSearch().tab ?? "catalog";
  return (
    <>
      <Topbar
        eyebrow="Artifact Studio"
        title="Models"
        subtitle="Move a completed run from immutable artifact to qualified local service or export."
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/train"><Play /> Train a model</Link>
          </Button>
        }
      />
      <ModelTabs active={tab} />
      {tab === "catalog" ? <CatalogWorkspace /> : null}
      {tab === "artifacts" ? <ArtifactsWorkspace /> : null}
      {tab === "cached" ? <CachedModelsWorkspace /> : null}
      {tab === "serve" ? <ServeTestWorkspace /> : null}
    </>
  );
}

function ModelTabs({ active }: { active: ModelTab }) {
  const navigate = useNavigate();
  return (
    <div className="sticky top-[49px] z-10 flex overflow-x-auto border-b border-border bg-bg-subtle/95 px-3 backdrop-blur md:px-5">
      {MODEL_TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => navigate({ to: "/models", search: { tab: tab.id } })}
          aria-current={active === tab.id ? "page" : undefined}
          className={cn(
            "relative flex h-11 shrink-0 items-center gap-2 px-3 text-[12px] transition-colors",
            active === tab.id ? "text-fg" : "text-fg-subtle hover:text-fg",
          )}
        >
          <span className={cn(active === tab.id && "font-medium")}>{tab.label}</span>
          <span className="hidden text-[10px] text-fg-disabled xl:inline">{tab.description}</span>
          {active === tab.id ? <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" /> : null}
        </button>
      ))}
    </div>
  );
}

/* ----------------------------------------------------------------------
 * Catalog
 * ------------------------------------------------------------------- */

function CatalogWorkspace() {
  const catalog = useModelCatalog();
  const backend = useBackendInfo();
  const [query, setQuery] = useState("");
  const [provider, setProvider] = useState("all");
  const [scope, setScope] = useState<"compatible" | "all">("compatible");
  const [selectedId, setSelectedId] = useState("");
  const detectedBackend = backend.data?.name ?? "";
  const providers = catalog.data?.facets.providers ?? [];
  const items = catalog.data?.items ?? [];
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const result = items.filter((model) => {
      if (provider !== "all" && model.provider !== provider) return false;
      if (scope === "compatible" && detectedBackend && !model.backend_support.includes(detectedBackend)) return false;
      if (!q) return true;
      return [model.id, model.label, model.provider, model.family, model.recommended_use, ...(model.tasks ?? []), ...(model.trainer_support ?? [])]
        .some((value) => String(value ?? "").toLowerCase().includes(q));
    });
    if (result.length || scope === "all" || !detectedBackend) return result;
    return items.filter((model) => {
      if (provider !== "all" && model.provider !== provider) return false;
      if (!q) return true;
      return [model.id, model.label, model.provider, model.family, model.recommended_use, ...(model.tasks ?? [])]
        .some((value) => String(value ?? "").toLowerCase().includes(q));
    });
  }, [detectedBackend, items, provider, query, scope]);
  const showingBackendFallback = scope === "compatible" && Boolean(detectedBackend) && filtered.length > 0 && !filtered.some((model) => model.backend_support.includes(detectedBackend));

  useEffect(() => {
    if (selectedId && filtered.some((item) => item.id === selectedId)) return;
    setSelectedId(filtered[0]?.id ?? "");
  }, [filtered, selectedId]);
  const selected = filtered.find((item) => item.id === selectedId) ?? null;

  return (
    <div className="grid min-h-[calc(100vh-154px)] lg:grid-cols-[minmax(340px,0.9fr)_minmax(380px,1.1fr)]">
      <section className="min-w-0 border-b border-border lg:border-b-0 lg:border-r">
        <WorkspaceToolbar
          title="Model catalog"
          detail={`${filtered.length} of ${items.length} models${detectedBackend ? ` · ${detectedBackend}` : ""}`}
          query={query}
          onQuery={setQuery}
          placeholder="Search model, task, or trainer…"
        >
          <NativeSelect value={provider} onChange={setProvider} ariaLabel="Provider filter">
            <option value="all">All providers</option>
            {providers.map((value) => <option key={value} value={value}>{value}</option>)}
          </NativeSelect>
          <NativeSelect value={scope} onChange={(value) => setScope(value as "compatible" | "all")} ariaLabel="Compatibility filter">
            <option value="compatible">Fits this workstation</option>
            <option value="all">All catalog models</option>
          </NativeSelect>
        </WorkspaceToolbar>
        {showingBackendFallback ? <div className="border-b border-warning/20 bg-warning-bg/35 px-4 py-2 text-[10px] text-warning">No exact {detectedBackend} matches were reported; showing all catalog models with backend guidance in the inspector.</div> : null}
        <div className="divide-y divide-border-subtle">
          {catalog.isLoading ? <WorkspaceMessage icon={Loader2} spin label="Loading model catalog" /> : catalog.isError ? <WorkspaceMessage icon={CircleDashed} label="Catalog unavailable" detail="Reconnect the workstation and try again." tone="danger" /> : filtered.length ? filtered.map((model) => (
            <CatalogRow key={model.id} model={model} selected={model.id === selectedId} detectedBackend={detectedBackend} onSelect={() => setSelectedId(model.id)} />
          )) : <WorkspaceMessage icon={Search} label="No matching models" detail="Broaden compatibility or clear the search." />}
        </div>
      </section>
      <aside className="min-w-0 bg-bg-subtle/20">
        {selected ? <CatalogInspector model={selected} detectedBackend={detectedBackend} /> : <WorkspaceMessage icon={Boxes} label="Select a model" detail="Compatibility, access notes, and next actions appear here." />}
      </aside>
    </div>
  );
}

function CatalogRow({ model, selected, detectedBackend, onSelect }: { model: ModelCatalogEntry; selected: boolean; detectedBackend: string; onSelect: () => void }) {
  const compatible = !detectedBackend || model.backend_support.includes(detectedBackend);
  return (
    <button type="button" onClick={onSelect} className={cn("group relative w-full px-4 py-3 text-left transition-colors hover:bg-surface/50", selected && "bg-accent-bg/50")}>
      {selected ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}
      <div className="flex items-start gap-3">
        <div className={cn("mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-md border", compatible ? "border-success/25 bg-success-bg text-success" : "border-border-subtle bg-surface text-fg-disabled")}><Box className="h-3.5 w-3.5" /></div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2"><span className="truncate font-mono text-[11.5px] font-medium text-fg">{model.id}</span>{model.recommended_first_run ? <Sparkles className="h-3 w-3 shrink-0 text-accent" /> : null}</div>
          <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-fg-subtle">{model.recommended_use || "Catalog metadata pending."}</p>
          <div className="mt-2 flex flex-wrap items-center gap-1.5"><SmallChip>{model.parameter_count || "size unknown"}</SmallChip><SmallChip>{model.memory_tier || "memory unknown"}</SmallChip><SmallChip tone={compatible ? "success" : "neutral"}>{compatible ? "compatible" : "other backend"}</SmallChip></div>
        </div>
        <ChevronRight className={cn("mt-1 h-3.5 w-3.5 shrink-0 text-fg-disabled transition-transform", selected && "translate-x-0.5 text-accent")} />
      </div>
    </button>
  );
}

function CatalogInspector({ model, detectedBackend }: { model: ModelCatalogEntry; detectedBackend: string }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const serve = useServeStart();
  const status = useServeStatus();
  const serveModel = model.mlx_variant ?? model.id;
  const compatible = !detectedBackend || model.backend_support.includes(detectedBackend);
  const workstationFit = detectedBackend ? `Fits ${detectedBackend}` : "Fits this workstation";
  const servingThis = Boolean(status.data?.running && status.data.model === serveModel);
  const caveats = model.known_caveats ?? [];

  function startServing() {
    serve.mutate({ model: serveModel, backend: model.mlx_variant ? "mlx" : undefined, trust_remote_code: model.trust_remote_code_required }, {
      onSuccess: () => navigate({ to: "/models", search: { tab: "serve" } }),
      onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.serve }),
    });
  }

  return (
    <div className="mx-auto max-w-3xl">
      <InspectorHeader eyebrow={`${model.provider} · ${model.family}`} title={model.label || model.id} subtitle={model.id} badges={<><Badge tone={compatible ? "success" : "warning"} size="sm" dot>{compatible ? workstationFit : "Different backend"}</Badge><Badge tone={model.risk_level === "safe" ? "success" : "warning"} size="sm">{model.risk_level}</Badge></>} />
      <section className="border-b border-border-subtle px-5 py-4">
        <p className="max-w-2xl text-[12.5px] leading-relaxed text-fg-muted">{model.recommended_use}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button size="sm" asChild><Link to="/train" search={{ model: model.id, mode: preferredTrainMode(model) }}><Play /> Use in Train</Link></Button>
          <Button size="sm" variant="secondary" onClick={startServing} disabled={serve.isPending || Boolean(status.data?.running && !servingThis)}>{serve.isPending ? <Loader2 className="animate-spin" /> : <Server />}{servingThis ? "Serving" : "Serve & Test"}</Button>
          <Button size="sm" variant="ghost" asChild><a href={model.model_url ?? `https://huggingface.co/${model.id}`} target="_blank" rel="noreferrer"><ExternalLink /> Model page</a></Button>
        </div>
        {serve.error ? <InlineNotice tone="danger">{serve.error.message}</InlineNotice> : null}
      </section>
      <InspectorSection title="Workstation fit">
        <KeyValue label="Backends" value={model.backend_support.join(" · ") || "Not reported"} />
        <KeyValue label="Estimated memory" value={model.estimated_memory_gb ? `${model.estimated_memory_gb} GB` : model.memory_tier || "Not reported"} />
        <KeyValue label="Preferred local variant" value={model.mlx_variant ?? "Upstream format"} mono />
        <KeyValue label="Training modes" value={model.trainer_support.join(" · ") || "Not listed"} />
      </InspectorSection>
      <InspectorSection title="Use and access">
        <KeyValue label="Modalities" value={model.modalities.join(" · ") || "Text"} />
        <KeyValue label="Tasks" value={model.tasks.join(" · ") || "General"} />
        <KeyValue label="Remote code" value={model.trust_remote_code_required ? "Required — review before enabling" : "Not required"} />
        <KeyValue label="Last verified" value={model.last_verified || "Not reported"} />
        {model.license_note ? <InlineNotice tone="warning">{model.license_note}</InlineNotice> : null}
        {model.download_note ? <InlineNotice>{model.download_note}</InlineNotice> : null}
      </InspectorSection>
      {caveats.length ? <InspectorSection title="Known caveats"><ul className="space-y-2 text-[11.5px] leading-relaxed text-warning">{caveats.map((caveat) => <li key={caveat} className="flex gap-2"><CircleDashed className="mt-1 h-2.5 w-2.5 shrink-0" />{caveat}</li>)}</ul></InspectorSection> : null}
    </div>
  );
}

/* ----------------------------------------------------------------------
 * Trained artifacts
 * ------------------------------------------------------------------- */

function ArtifactsWorkspace() {
  const navigate = useNavigate();
  const { artifact: requestedId } = Route.useSearch();
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("all");
  const [selectedId, setSelectedId] = useState(requestedId ?? "");
  const artifacts = useModelArtifacts({ limit: 200 });
  const operations = useArtifactOperations();
  const items = artifacts.data?.items ?? [];
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((item) => {
      if (kind !== "all" && item.kind !== kind) return false;
      if (!q) return true;
      return [item.id, item.model_name, item.path, item.content_hash, item.format, item.quantization, ...artifactAliasNames(item)]
        .some((value) => String(value ?? "").toLowerCase().includes(q));
    });
  }, [items, kind, query]);
  const kinds = Array.from(new Set(items.map((item) => item.kind))).sort();

  useEffect(() => {
    if (selectedId && visible.some((item) => item.id === selectedId)) return;
    setSelectedId(visible[0]?.id ?? "");
  }, [selectedId, visible]);
  const selected = items.find((item) => item.id === selectedId) ?? null;
  const activeOperations = (operations.data?.items ?? []).filter((operation) => ["queued", "running", "preparing"].includes(operation.status));

  function selectArtifact(id: string) {
    setSelectedId(id);
    navigate({ to: "/models", search: { tab: "artifacts", artifact: id }, replace: true });
  }

  return (
    <div className="grid min-h-[calc(100vh-154px)] xl:grid-cols-[330px_minmax(0,1fr)]">
      <section className="min-w-0 border-b border-border xl:border-b-0 xl:border-r">
        <WorkspaceToolbar title="Artifact library" detail={`${items.length} occurrence${items.length === 1 ? "" : "s"} · ${activeOperations.length} active`} query={query} onQuery={setQuery} placeholder="Search hash, run, model, or alias…">
          <NativeSelect value={kind} onChange={setKind} ariaLabel="Artifact kind filter"><option value="all">All artifact kinds</option>{kinds.map((value) => <option key={value} value={value}>{prettyKind(value)}</option>)}</NativeSelect>
          <Button size="sm" variant="ghost" onClick={() => { artifacts.refetch(); operations.refetch(); }}><RefreshCw className={cn((artifacts.isFetching || operations.isFetching) && "animate-spin")} /> Refresh</Button>
        </WorkspaceToolbar>
        {activeOperations.length ? <OperationStrip operations={activeOperations} /> : null}
        <div className="divide-y divide-border-subtle">
          {artifacts.isLoading ? <WorkspaceMessage icon={Loader2} spin label="Indexing artifacts" /> : artifacts.isError ? <WorkspaceMessage icon={CircleDashed} label="Artifact index unavailable" detail="Existing v3 checkpoints remain safe. Reconnect to load the library." tone="danger" /> : visible.length ? visible.map((artifact) => <ArtifactRow key={artifact.id} artifact={artifact} selected={artifact.id === selectedId} onSelect={() => selectArtifact(artifact.id)} />) : <WorkspaceMessage icon={FileArchive} label="No trained artifacts" detail="Complete a run or import a local artifact to begin." action={<Button size="sm" asChild><Link to="/train"><Play /> Start training</Link></Button>} />}
        </div>
      </section>
      <aside className="min-w-0 bg-bg-subtle/15">
        {selected ? <ArtifactInspector artifact={selected} allArtifacts={items} /> : <WorkspaceMessage icon={PackageCheck} label="Select an artifact" detail="Inspect lineage, verify integrity, transform, qualify, promote, or export it." />}
      </aside>
    </div>
  );
}

function ArtifactRow({ artifact, selected, onSelect }: { artifact: ModelArtifactOccurrence; selected: boolean; onSelect: () => void }) {
  const name = artifact.model_name || artifact.path.split(/[\\/]/).filter(Boolean).pop() || artifact.id;
  const aliases = artifactAliasNames(artifact);
  return (
    <button type="button" onClick={onSelect} className={cn("group relative w-full px-4 py-3 text-left transition-colors hover:bg-surface/55", selected && "bg-accent-bg/55")}>
      {selected ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}
      <div className="flex items-start gap-2.5">
        <ArtifactKindIcon kind={artifact.kind} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5"><span className="truncate text-[12px] font-medium text-fg">{name}</span>{artifact.pinned ? <Pin className="h-3 w-3 shrink-0 fill-accent text-accent" /> : null}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 font-mono text-[9.5px] uppercase tracking-wide text-fg-disabled"><span>{prettyKind(artifact.kind)}</span>{artifact.format ? <><span>·</span><span>{artifact.format}</span></> : null}{artifact.quantization ? <><span>·</span><span>{artifact.quantization}</span></> : artifact.dtype ? <><span>·</span><span>{artifact.dtype}</span></> : null}</div>
          <div className="mt-2 flex items-center justify-between gap-2"><span className="truncate font-mono text-[9.5px] text-fg-disabled">{shortHash(artifact.content_hash)}</span><span className="text-[9.5px] text-fg-disabled">{artifact.size_bytes != null ? formatBytes(artifact.size_bytes) : artifact.created_at ? relativeTime(artifact.created_at) : ""}</span></div>
          {aliases.length ? <div className="mt-2 flex flex-wrap gap-1">{aliases.map((alias) => <SmallChip key={alias} tone={alias === "approved" ? "success" : "accent"}>{alias}</SmallChip>)}</div> : null}
        </div>
        <ChevronRight className={cn("mt-1 h-3.5 w-3.5 shrink-0 text-fg-disabled transition-transform", selected && "translate-x-0.5 text-accent")} />
      </div>
    </button>
  );
}

type StudioAction = "bake" | "merge" | "convert" | "quantize" | "qualify" | "export" | null;

function ArtifactInspector({ artifact, allArtifacts }: { artifact: ModelArtifactOccurrence; allArtifacts: ModelArtifactOccurrence[] }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [action, setAction] = useState<StudioAction>(null);
  const [tagText, setTagText] = useState((artifact.tags ?? []).join(", "));
  const [feedback, setFeedback] = useState<string | null>(null);
  const [promotionTarget, setPromotionTarget] = useState<"candidate" | "approved" | null>(null);
  const [promotionOverride, setPromotionOverride] = useState(false);
  const [promotionNote, setPromotionNote] = useState("");
  const qualifications = useArtifactQualifications(artifact.id);
  const aliases = artifactAliasNames(artifact);
  const integrity = artifact.integrity ?? artifact.blob?.integrity ?? "unverified";
  const pin = useMutation({
    mutationFn: () => api.pinArtifact(artifact.id, !artifact.pinned),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["artifacts"] }),
  });
  const tag = useMutation({
    mutationFn: () => api.tagArtifact(artifact.id, tagText.split(",").map((value) => value.trim()).filter(Boolean)),
    onSuccess: () => { setFeedback("Tags updated."); queryClient.invalidateQueries({ queryKey: ["artifacts"] }); },
  });
  const verify = useMutation({
    mutationFn: () => api.verifyArtifact(artifact.id),
    onSuccess: (result) => { setFeedback(`Verification queued${result.work_item_id ? ` · ${result.work_item_id}` : ""}.`); queryClient.invalidateQueries({ queryKey: queryKeys.activity }); },
  });
  const promote = useMutation({
    mutationFn: ({ alias, override, note }: { alias: "candidate" | "approved"; override: boolean; note: string }) => api.promoteArtifact(artifact.id, { alias, override, note: note || undefined }),
    onSuccess: (result) => { setFeedback(`Promoted as ${result.alias}.`); setPromotionTarget(null); setPromotionOverride(false); setPromotionNote(""); queryClient.invalidateQueries({ queryKey: ["artifacts"] }); },
  });

  useEffect(() => { setAction(null); setFeedback(null); setTagText((artifact.tags ?? []).join(", ")); setPromotionTarget(null); setPromotionOverride(false); setPromotionNote(""); }, [artifact.id, artifact.tags]);

  return (
    <div className="mx-auto max-w-4xl">
      <InspectorHeader eyebrow={prettyKind(artifact.kind)} title={artifact.model_name || artifact.path.split(/[\\/]/).filter(Boolean).pop() || artifact.id} subtitle={artifact.id} badges={<><Badge tone={integrity === "verified" ? "success" : integrity === "invalid" ? "danger" : "warning"} dot size="sm">{integrity}</Badge>{aliases.map((alias) => <Badge key={alias} tone={alias === "approved" ? "success" : "accent"} size="sm">{alias}</Badge>)}</>} />

      <section className="border-b border-border-subtle px-5 py-4">
        <div className="grid gap-2 sm:grid-cols-3">
          <PrimaryArtifactAction label="Bake adapter" detail="Fold one adapter into its base" icon={Merge} onClick={() => setAction(action === "bake" ? null : "bake")} active={action === "bake"} disabled={artifact.kind !== "adapter"} />
          <PrimaryArtifactAction label="Convert / quantize" detail="Publish another local format" icon={Archive} onClick={() => setAction(action === "convert" ? null : "convert")} active={action === "convert" || action === "quantize"} />
          <PrimaryArtifactAction label="Qualify" detail="Measure quality and operation" icon={ShieldCheck} onClick={() => setAction(action === "qualify" ? null : "qualify")} active={action === "qualify"} />
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Button size="sm" variant="ghost" onClick={() => setAction(action === "merge" ? null : "merge")}><GitBranch /> Combine adapters</Button>
          <Button size="sm" variant="ghost" onClick={() => setAction(action === "export" ? null : "export")}><Download /> Export bundle</Button>
          <Button size="sm" variant="ghost" onClick={() => verify.mutate()} disabled={verify.isPending}>{verify.isPending ? <Loader2 className="animate-spin" /> : <CheckCircle2 />} Verify</Button>
          <Button size="sm" variant="ghost" onClick={() => pin.mutate()} disabled={pin.isPending}><Pin /> {artifact.pinned ? "Unpin" : "Pin"}</Button>
          <Button size="sm" variant="ghost" onClick={() => navigate({ to: "/models", search: { tab: "serve", artifact: artifact.id } })}><Server /> Serve & Test</Button>
        </div>
        {feedback ? <InlineNotice tone="success">{feedback}</InlineNotice> : null}
        {[pin.error, tag.error, verify.error, promote.error].find(Boolean) instanceof Error ? <InlineNotice tone="danger">{([pin.error, tag.error, verify.error, promote.error].find(Boolean) as Error).message}</InlineNotice> : null}
      </section>

      {action ? <OperationComposer action={action} artifact={artifact} allArtifacts={allArtifacts} onAction={setAction} onQueued={(message) => { setFeedback(message); setAction(null); }} /> : null}

      <InspectorSection title="Identity and integrity">
        <KeyValue label="Content hash" value={artifact.content_hash || "Hashing pending"} mono copyable />
        <KeyValue label="Format" value={[artifact.format ?? artifact.blob?.format, artifact.quantization ?? artifact.blob?.quantization ?? artifact.dtype ?? artifact.blob?.dtype].filter(Boolean).join(" · ") || "Not reported"} />
        <KeyValue label="Size" value={formatBytes(artifact.size_bytes ?? artifact.blob?.size_bytes)} />
        <KeyValue label="Storage" value={artifact.locations?.some((location) => location.kind === "managed") ? "Managed library copy" : "Referenced run output"} />
        <KeyValue label="Created" value={artifact.created_at ? relativeTime(artifact.created_at) : "Not reported"} />
      </InspectorSection>

      {artifact.specialized_task ? <InspectorSection title="Specialized task contract">
        <KeyValue label="Task" value={`${artifact.specialized_task.task_kind} · ${artifact.specialized_task.modality}`} />
        <KeyValue label="Loss adapter" value={`${artifact.specialized_task.loss_adapter}@${artifact.specialized_task.loss_adapter_version}`} mono />
        <KeyValue label="Model head" value={artifact.specialized_task.model_head_hash} mono copyable />
        <KeyValue label="Processor" value={artifact.specialized_task.processor_hash} mono copyable />
        {artifact.specialized_task.label_schema_revision_id ? <KeyValue label="Label schema" value={artifact.specialized_task.label_schema_revision_id} mono /> : null}
        {artifact.specialized_task.retrieval_corpus_hash ? <KeyValue label="Retrieval corpus" value={artifact.specialized_task.retrieval_corpus_hash} mono copyable /> : null}
      </InspectorSection> : null}

      <InspectorSection title="Provenance">
        <KeyValue label="Run" value={artifact.run_id ?? "Derived operation"} mono />
        {artifact.run_group_id ? <KeyValue label="Experiment" value={artifact.run_group_id} mono /> : null}
        {artifact.step != null ? <KeyValue label="Checkpoint step" value={String(artifact.step)} mono /> : null}
        {artifact.cycle != null ? <KeyValue label="Cycle" value={String(artifact.cycle)} mono /> : null}
        <KeyValue label="Path" value={artifact.path} mono copyable />
        {artifact.parents?.length ? <div className="mt-2"><div className="mb-1 text-[10px] text-fg-disabled">Ordered parents</div>{artifact.parents.sort((a, b) => a.ordinal - b.ordinal).map((edge) => <div key={edge.id} className="border-l border-border-strong py-1 pl-2 font-mono text-[10px] text-fg-subtle">{edge.ordinal + 1}. {edge.parent_artifact_id}</div>)}</div> : null}
      </InspectorSection>

      <InspectorSection title="Qualification">
        {qualifications.isLoading ? <div className="flex items-center gap-2 text-[11px] text-fg-subtle"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading evidence</div> : qualifications.data?.items.length ? qualifications.data.items.map((qualification) => <QualificationRow key={qualification.id} qualification={qualification} />) : <div className="flex items-center justify-between gap-4"><p className="text-[11px] text-fg-subtle">No qualification decision is attached to this artifact.</p><Button size="sm" variant="ghost" onClick={() => setAction("qualify")}>Run qualification</Button></div>}
        <QualificationComparisonSurface artifact={artifact} allArtifacts={allArtifacts} qualifications={qualifications.data?.items ?? []} />
      </InspectorSection>

      <InspectorSection title="Aliases and notes">
        <div className="flex flex-wrap gap-2"><Button size="sm" variant="secondary" onClick={() => setPromotionTarget("candidate")} disabled={promote.isPending || aliases.includes("candidate")}><Sparkles /> Promote candidate</Button><Button size="sm" variant="secondary" onClick={() => setPromotionTarget("approved")} disabled={promote.isPending || aliases.includes("approved")}><Check /> Promote approved</Button></div>
        {promotionTarget ? <div className="mt-3 rounded-md border border-border-subtle bg-bg-subtle/55 p-3"><div className="flex items-center justify-between gap-3"><div><div className="text-[10.5px] font-medium text-fg">Review {promotionTarget} promotion</div><p className="mt-0.5 text-[9.5px] text-fg-subtle">The decision and any override note remain in append-only alias history.</p></div><button type="button" onClick={() => setPromotionTarget(null)} className="text-fg-disabled hover:text-fg" aria-label="Close promotion review"><X className="h-3.5 w-3.5" /></button></div><label className="mt-3 flex items-center gap-2 text-[10.5px] text-fg-muted"><input type="checkbox" checked={promotionOverride} onChange={(event) => setPromotionOverride(event.target.checked)} />Override qualification gates</label><FormField label={promotionOverride ? "Override note · required" : "Promotion note"} className="mt-3"><textarea value={promotionNote} onChange={(event) => setPromotionNote(event.target.value)} rows={2} placeholder={promotionOverride ? "Explain why this exception is acceptable…" : "Optional decision context…"} className="w-full resize-none rounded-md border border-border bg-bg px-2.5 py-2 text-[11px] text-fg outline-none focus:border-accent" /></FormField><div className="mt-3 flex justify-end gap-2"><Button size="sm" variant="ghost" onClick={() => setPromotionTarget(null)}>Cancel</Button><Button size="sm" onClick={() => promote.mutate({ alias: promotionTarget, override: promotionOverride, note: promotionNote.trim() })} disabled={promote.isPending || (promotionOverride && !promotionNote.trim())}>{promote.isPending ? <Loader2 className="animate-spin" /> : <Check />} Confirm promotion</Button></div></div> : null}
        <div className="mt-3 flex gap-2"><Input value={tagText} onChange={(event) => setTagText(event.target.value)} placeholder="Tags separated by commas" className="h-8 text-[11px]" /><Button size="sm" variant="ghost" onClick={() => tag.mutate()} disabled={tag.isPending}><Tag /> Save tags</Button></div>
        {artifact.notes ? <p className="mt-3 text-[11.5px] leading-relaxed text-fg-muted">{artifact.notes}</p> : null}
      </InspectorSection>

      <details className="border-b border-border-subtle px-5 py-3"><summary className="cursor-pointer text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled hover:text-fg">Advanced manifest</summary><pre className="mt-3 max-h-72 overflow-auto rounded-sm border border-border-subtle bg-bg-subtle p-3 font-mono text-[9.5px] leading-relaxed text-fg-subtle">{JSON.stringify({ blob: artifact.blob, locations: artifact.locations, metadata: artifact.metadata, specialized_task: artifact.specialized_task }, null, 2)}</pre></details>
    </div>
  );
}

function OperationComposer({ action, artifact, allArtifacts, onAction, onQueued }: { action: Exclude<StudioAction, null>; artifact: ModelArtifactOccurrence; allArtifacts: ModelArtifactOccurrence[]; onAction: (action: StudioAction) => void; onQueued: (message: string) => void }) {
  const queryClient = useQueryClient();
  const [targetFormat, setTargetFormat] = useState("gguf");
  const [precision, setPrecision] = useState("q4");
  const [mergeMethod, setMergeMethod] = useState("linear");
  const [additionalInputs, setAdditionalInputs] = useState<string[]>([]);
  const [profileId, setProfileId] = useState("");
  const [bundleName, setBundleName] = useState("");
  const profiles = useQuery({ queryKey: ["qualification-profiles"], queryFn: () => api.listQualificationProfiles({ limit: 100 }), enabled: action === "qualify", retry: false });
  const create = useMutation({
    mutationFn: async () => {
      if (action === "qualify") return api.qualifyArtifact({ artifact_id: artifact.id, profile_revision_id: profileId });
      const inputIds = action === "merge" ? [artifact.id, ...additionalInputs] : [artifact.id];
      return api.createArtifactOperation({
        kind: action,
        input_artifact_ids: inputIds,
        config: action === "convert" ? { target_format: targetFormat, precision } : action === "quantize" ? { quantization: precision, post_training: true } : action === "merge" ? { method: mergeMethod } : action === "export" ? { bundle_name: bundleName || undefined, portable: true } : { publish_format: "huggingface" },
      });
    },
    onSuccess: (result) => {
      const identity = "work_item_id" in result ? result.work_item_id : null;
      onQueued(`${prettyKind(action)} queued${identity ? ` · ${identity}` : ""}.`);
      queryClient.invalidateQueries({ queryKey: queryKeys.activity });
      queryClient.invalidateQueries({ queryKey: queryKeys.artifactOperations });
      queryClient.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const candidates = allArtifacts.filter((item) => item.id !== artifact.id && (action !== "merge" || item.kind === "adapter"));
  const ready = action === "qualify" ? Boolean(profileId) : action === "merge" ? additionalInputs.length > 0 : true;

  return (
    <section className="border-b border-accent/30 bg-accent-bg/25 px-5 py-4">
      <div className="flex items-start justify-between gap-3"><div><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-accent">New operation</div><h3 className="mt-1 text-[14px] font-medium text-fg">{operationTitle(action)}</h3><p className="mt-1 text-[10.5px] text-fg-subtle">Inputs resolve now; work publishes atomically after verification.</p></div><Button size="icon" variant="ghost" onClick={() => onAction(null)}><X /></Button></div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {action === "convert" ? <><FormField label="Target format"><NativeSelect value={targetFormat} onChange={setTargetFormat} ariaLabel="Target format"><option value="huggingface">Hugging Face</option><option value="mlx">MLX</option><option value="gguf">GGUF</option></NativeSelect><p className="mt-1.5 text-[9.5px] text-fg-subtle">Only formats verified by the local conversion service are shown.</p></FormField><FormField label="Precision"><NativeSelect value={precision} onChange={setPrecision} ariaLabel="Precision"><option value="fp16">FP16</option><option value="bf16">BF16</option><option value="fp32">FP32</option><option value="q8">Q8</option><option value="q4">Q4</option></NativeSelect></FormField></> : null}
        {action === "quantize" ? <FormField label="Post-training precision"><NativeSelect value={precision} onChange={setPrecision} ariaLabel="Post-training precision"><option value="q4">Q4</option><option value="q8">Q8</option><option value="fp16">FP16</option><option value="bf16">BF16</option></NativeSelect><p className="mt-1.5 text-[9.5px] leading-relaxed text-warning">This is post-training quantization, not quantization-aware training.</p></FormField> : null}
        {action === "merge" ? <><FormField label="Merge method"><NativeSelect value={mergeMethod} onChange={setMergeMethod} ariaLabel="Merge method"><option value="linear">Linear</option><option value="ties">TIES</option><option value="dare">DARE</option><option value="magnitude_pruning">Magnitude pruning</option></NativeSelect></FormField><FormField label="Additional adapters"><div className="max-h-36 space-y-1 overflow-y-auto rounded-sm border border-border-subtle bg-bg p-2">{candidates.length ? candidates.map((item) => <label key={item.id} className="flex items-center gap-2 py-1 text-[10.5px] text-fg-muted"><input type="checkbox" checked={additionalInputs.includes(item.id)} onChange={(event) => setAdditionalInputs((values) => event.target.checked ? [...values, item.id] : values.filter((id) => id !== item.id))} /><span className="truncate">{item.model_name || item.id}</span></label>) : <span className="text-[10px] text-fg-disabled">No compatible adapters indexed.</span>}</div></FormField></> : null}
        {action === "qualify" ? <div className="space-y-3 sm:col-span-2"><FormField label="Qualification profile"><NativeSelect value={profileId} onChange={setProfileId} ariaLabel="Qualification profile"><option value="">Choose an immutable profile revision</option>{profiles.data?.items.map((profile) => <option key={profile.id} value={profile.id}>{profile.name || profile.id} · {profile.target_backend}</option>)}</NativeSelect>{profiles.isError ? <p className="mt-1.5 text-[10px] text-warning">Stored profiles are temporarily unavailable.</p> : null}</FormField><QualificationProfileCreator onCreated={setProfileId} /></div> : null}
        {action === "export" ? <FormField label="Bundle name" className="sm:col-span-2"><Input value={bundleName} onChange={(event) => setBundleName(event.target.value)} placeholder="approved-local-model" className="h-8 text-[11px]" /><p className="mt-1.5 text-[9.5px] text-fg-subtle">Includes checksums, lineage, qualification evidence, metadata, and a generated model card.</p></FormField> : null}
        {action === "bake" ? <div className="sm:col-span-2 rounded-sm border border-border-subtle bg-bg/50 px-3 py-2 text-[10.5px] leading-relaxed text-fg-muted">The adapter will be folded into its recorded base. The source adapter and base remain immutable lineage parents.</div> : null}
      </div>
      <div className="mt-4 flex items-center justify-between gap-3"><span className="font-mono text-[9.5px] text-fg-disabled">input {shortHash(artifact.content_hash)}</span><div className="flex gap-2"><Button size="sm" variant="ghost" onClick={() => onAction(null)}>Cancel</Button><Button size="sm" onClick={() => create.mutate()} disabled={!ready || create.isPending}>{create.isPending ? <Loader2 className="animate-spin" /> : <ArrowRight />} Queue operation</Button></div></div>
      {create.error ? <InlineNotice tone="danger">{create.error.message}</InlineNotice> : null}
    </section>
  );
}

function OperationStrip({ operations }: { operations: ArtifactOperation[] }) {
  return <div className="border-b border-border-subtle bg-accent-bg/20 px-4 py-2"><div className="flex items-center gap-2 text-[10.5px] text-fg-muted"><Loader2 className="h-3 w-3 animate-spin text-accent" /><span className="font-medium text-fg">{operations.length} operation{operations.length === 1 ? "" : "s"} active</span><span className="truncate text-fg-disabled">{operations.slice(0, 2).map((item) => prettyKind(item.kind)).join(" · ")}</span></div></div>;
}

function PrimaryArtifactAction({ label, detail, icon: Icon, onClick, active, disabled }: { label: string; detail: string; icon: typeof Merge; onClick: () => void; active: boolean; disabled?: boolean }) {
  return <button type="button" onClick={onClick} disabled={disabled} className={cn("rounded-md border px-3 py-2.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40", active ? "border-accent bg-accent-bg" : "border-border-subtle bg-surface/45 hover:border-border-strong hover:bg-surface")}><div className="flex items-center gap-2"><Icon className={cn("h-3.5 w-3.5", active ? "text-accent" : "text-fg-subtle")} /><span className="text-[11.5px] font-medium text-fg">{label}</span></div><p className="mt-1 text-[9.5px] text-fg-disabled">{detail}</p></button>;
}

function QualificationRow({ qualification }: { qualification: ArtifactQualification }) {
  const decision = qualification.decision ?? qualification.status;
  return <div className="border-b border-border-subtle py-2.5 last:border-0"><div className="flex items-center justify-between gap-3"><div className="min-w-0"><div className="truncate font-mono text-[10px] text-fg-muted">{qualification.profile_revision_id}</div><div className="mt-1 text-[10px] text-fg-disabled">{qualification.completed_at ? relativeTime(qualification.completed_at) : qualification.status}</div></div><Badge tone={decision === "pass" ? "success" : decision === "fail" ? "danger" : decision === "warn" ? "warning" : "neutral"} dot size="sm">{decision}</Badge></div>{qualification.reasons?.length ? <p className="mt-2 text-[10.5px] leading-relaxed text-fg-subtle">{qualification.reasons.join(" · ")}</p> : null}</div>;
}

type QualificationThresholdDraft = {
  id: string;
  stage: "development" | "operational" | "holdout";
  metric: string;
  direction: "maximize" | "minimize";
  passThreshold: string;
  maximumRegression: string;
};

function QualificationProfileCreator({ onCreated }: { onCreated: (id: string) => void }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [developmentSuite, setDevelopmentSuite] = useState("");
  const [operationalSuite, setOperationalSuite] = useState("");
  const [holdoutSuite, setHoldoutSuite] = useState("");
  const [backend, setBackend] = useState("auto");
  const [seed, setSeed] = useState(42);
  const [thresholds, setThresholds] = useState<QualificationThresholdDraft[]>([
    { id: "quality-primary", stage: "development", metric: "primary_metric", direction: "maximize", passThreshold: "", maximumRegression: "0" },
    { id: "throughput", stage: "operational", metric: "output_tokens_per_second", direction: "maximize", passThreshold: "", maximumRegression: "" },
  ]);
  const suites = useQuery({ queryKey: ["benchmark-suites", "qualification-profile-form"], queryFn: () => api.listBenchmarkSuites(), enabled: open, retry: false });
  const create = useMutation({
    mutationFn: () => api.createQualificationProfile({
      name: name.trim(),
      description: description.trim() || undefined,
      development_suite_revision_id: developmentSuite,
      operational_suite_revision_id: operationalSuite,
      holdout_suite_revision_id: holdoutSuite || null,
      thresholds: thresholds.filter((item) => item.metric.trim()).map((item) => ({
        stage: item.stage,
        metric: item.metric.trim(),
        direction: item.direction,
        pass_threshold: optionalNumber(item.passThreshold),
        maximum_regression: optionalNumber(item.maximumRegression),
        required: true,
      })),
      target_backend: backend.trim() || "auto",
      generation_settings: { seed },
      performance_settings: { warmups: 2, measured_repetitions: 5, concurrency: 1, seed },
    }),
    onSuccess: (profile) => {
      queryClient.invalidateQueries({ queryKey: ["qualification-profiles"] });
      onCreated(profile.id);
      setOpen(false);
    },
  });
  const items = suites.data?.items ?? [];
  const development = suitesForPurpose(items, "development");
  const operational = suitesForPurpose(items, "operational");
  const holdout = suitesForPurpose(items, "holdout");
  const canCreate = Boolean(name.trim() && developmentSuite && operationalSuite && thresholds.some((item) => item.metric.trim()));

  if (!open) return <Button size="sm" variant="ghost" onClick={() => setOpen(true)}><Plus /> Create structured profile</Button>;
  return (
    <div className="rounded-md border border-border-subtle bg-bg/65 p-3">
      <div className="flex items-start justify-between gap-3"><div><div className="text-[10.5px] font-medium text-fg">New qualification profile</div><p className="mt-0.5 text-[9.5px] text-fg-subtle">Immutable suite revisions, gates, backend, and deterministic measurement defaults.</p></div><button type="button" onClick={() => setOpen(false)} aria-label="Close profile creator" className="text-fg-disabled hover:text-fg"><X className="h-3.5 w-3.5" /></button></div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <FormField label="Profile name"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Local GGUF candidate" className="h-8 text-[11px]" /></FormField>
        <FormField label="Target backend"><Input value={backend} onChange={(event) => setBackend(event.target.value)} placeholder="llama.cpp" className="h-8 text-[11px]" /></FormField>
        <FormField label="Development quality suite"><SuiteSelect value={developmentSuite} onChange={setDevelopmentSuite} suites={development} placeholder="Choose development revision" /></FormField>
        <FormField label="Operational performance suite"><SuiteSelect value={operationalSuite} onChange={setOperationalSuite} suites={operational} placeholder="Choose operational revision" /></FormField>
        <FormField label="Final holdout · optional"><SuiteSelect value={holdoutSuite} onChange={setHoldoutSuite} suites={holdout} placeholder="No holdout confirmation" /></FormField>
        <FormField label="Fixed generation seed"><Input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} mono className="h-8 text-[11px]" /></FormField>
        <FormField label="Description" className="sm:col-span-2"><Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="What this gate protects" className="h-8 text-[11px]" /></FormField>
      </div>
      <div className="mt-4"><div className="flex items-center justify-between"><span className="text-[9.5px] font-medium uppercase tracking-[0.1em] text-fg-disabled">Required metric gates</span><Button size="sm" variant="ghost" onClick={() => setThresholds((values) => [...values, { id: globalThis.crypto?.randomUUID?.() ?? `metric-${Date.now()}`, stage: "development", metric: "", direction: "maximize", passThreshold: "", maximumRegression: "" }])}><Plus /> Metric</Button></div><div className="mt-2 space-y-2">{thresholds.map((threshold) => <div key={threshold.id} className="grid gap-2 rounded-sm border border-border-subtle p-2 sm:grid-cols-[120px_minmax(150px,1fr)_110px_100px_100px_auto]"><NativeSelect value={threshold.stage} onChange={(value) => setThresholds((items) => items.map((item) => item.id === threshold.id ? { ...item, stage: value as QualificationThresholdDraft["stage"] } : item))} ariaLabel="Metric stage"><option value="development">Development</option><option value="operational">Operational</option><option value="holdout">Holdout</option></NativeSelect><Input value={threshold.metric} onChange={(event) => setThresholds((items) => items.map((item) => item.id === threshold.id ? { ...item, metric: event.target.value } : item))} placeholder="metric name" className="h-8 font-mono text-[10px]" /><NativeSelect value={threshold.direction} onChange={(value) => setThresholds((items) => items.map((item) => item.id === threshold.id ? { ...item, direction: value as QualificationThresholdDraft["direction"] } : item))} ariaLabel="Metric direction"><option value="maximize">Maximize</option><option value="minimize">Minimize</option></NativeSelect><Input value={threshold.passThreshold} onChange={(event) => setThresholds((items) => items.map((item) => item.id === threshold.id ? { ...item, passThreshold: event.target.value } : item))} placeholder="threshold" className="h-8 font-mono text-[10px]" /><Input value={threshold.maximumRegression} onChange={(event) => setThresholds((items) => items.map((item) => item.id === threshold.id ? { ...item, maximumRegression: event.target.value } : item))} placeholder="max Δ" className="h-8 font-mono text-[10px]" /><Button size="icon" variant="ghost" onClick={() => setThresholds((items) => items.filter((item) => item.id !== threshold.id))} disabled={thresholds.length === 1} title="Remove metric"><Trash2 /></Button></div>)}</div></div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3"><p className="text-[9.5px] text-fg-subtle">Performance defaults: 2 warmups · 5 measurements · concurrency 1.</p><Button size="sm" onClick={() => create.mutate()} disabled={!canCreate || create.isPending}>{create.isPending ? <Loader2 className="animate-spin" /> : <Check />} Save immutable profile</Button></div>
      {suites.isError ? <InlineNotice tone="warning">Suite revisions are unavailable. The profile cannot be saved until both development and operational pickers resolve.</InlineNotice> : null}
      {create.error ? <InlineNotice tone="danger">{create.error.message}</InlineNotice> : null}
    </div>
  );
}

function SuiteSelect({ value, onChange, suites, placeholder }: { value: string; onChange: (value: string) => void; suites: BenchmarkSuite[]; placeholder: string }) {
  return <NativeSelect value={value} onChange={onChange} ariaLabel={placeholder}><option value="">{placeholder}</option>{suites.map((suite) => { const revision = suite.latest_revision_id ?? suite.latest_revision?.id; return revision ? <option key={revision} value={revision}>{suite.name} · {revision}</option> : null; })}</NativeSelect>;
}

function QualificationComparisonSurface({ artifact, allArtifacts, qualifications }: { artifact: ModelArtifactOccurrence; allArtifacts: ModelArtifactOccurrence[]; qualifications: ArtifactQualification[] }) {
  const [candidateId, setCandidateId] = useState("");
  const [baseId, setBaseId] = useState("");
  const all = useQuery({ queryKey: ["qualifications", "comparison-library"], queryFn: () => api.listQualifications({ limit: 200 }), enabled: qualifications.length > 0, retry: false });
  const candidate = qualifications.find((item) => item.id === candidateId) ?? qualifications[0];
  const baselines = (all.data?.items ?? []).filter((item) => item.artifact_id !== artifact.id && item.profile_revision_id === candidate?.profile_revision_id && ["pass", "warn", "fail"].includes(String(item.decision ?? item.status)));
  const baseline = baselines.find((item) => item.id === baseId) ?? baselines.find((item) => item.artifact_id === candidate?.parent_artifact_id) ?? baselines[0];
  const comparison = useQuery<QualificationComparison>({ queryKey: ["qualifications", "compare", baseline?.id, candidate?.id], queryFn: () => api.compareQualifications(baseline!.id, candidate!.id), enabled: Boolean(baseline?.id && candidate?.id), retry: false });
  useEffect(() => { if (candidate && candidate.id !== candidateId) setCandidateId(candidate.id); }, [candidate, candidateId]);
  useEffect(() => { if (baseline && baseline.id !== baseId) setBaseId(baseline.id); }, [baseId, baseline]);
  if (!qualifications.length) return null;
  const baseArtifact = allArtifacts.find((item) => item.id === baseline?.artifact_id);
  const axes = buildQualificationAxes(comparison.data, baseline, candidate, baseArtifact, artifact);
  return (
    <div className="mt-3 rounded-md border border-border-subtle bg-bg-subtle/45 p-3">
      <div className="flex flex-wrap items-start justify-between gap-2"><div><div className="text-[10.5px] font-medium text-fg">Qualification comparison</div><p className="mt-0.5 text-[9.5px] text-fg-subtle">Same-profile evidence across quality, speed, memory, and footprint.</p></div>{comparison.isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" /> : null}</div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2"><NativeSelect value={candidate?.id ?? ""} onChange={(value) => { setCandidateId(value); setBaseId(""); }} ariaLabel="Candidate qualification">{qualifications.map((item) => <option key={item.id} value={item.id}>{item.profile_revision_id} · {item.decision ?? item.status}</option>)}</NativeSelect><NativeSelect value={baseline?.id ?? ""} onChange={setBaseId} ariaLabel="Baseline qualification"><option value="">Choose same-profile baseline</option>{baselines.map((item) => <option key={item.id} value={item.id}>{artifactName(allArtifacts.find((artifactItem) => artifactItem.id === item.artifact_id))} · {item.decision ?? item.status}</option>)}</NativeSelect></div>
      {baseline ? <div className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-sm border border-border-subtle bg-border-subtle sm:grid-cols-4">{axes.map((axis) => <div key={axis.label} className="bg-bg px-3 py-2.5"><div className="text-[8.5px] uppercase tracking-[0.12em] text-fg-disabled">{axis.label}</div><div className={cn("mt-1 font-mono text-[12px]", axis.tone === "good" ? "text-success" : axis.tone === "bad" ? "text-danger" : "text-fg-muted")}>{axis.delta}</div><div className="mt-1 truncate text-[9px] text-fg-disabled" title={axis.detail}>{axis.detail}</div></div>)}</div> : <p className="mt-3 text-[10px] text-fg-subtle">No completed baseline exists under this exact profile revision yet.</p>}
      {comparison.isError ? <InlineNotice tone="warning">Comparison evidence could not be matched. Both artifacts must use the same profile revision.</InlineNotice> : null}
    </div>
  );
}

/* ----------------------------------------------------------------------
 * Cached models and storage
 * ------------------------------------------------------------------- */

function CachedModelsWorkspace() {
  const storage = useStorageInventory();
  const catalog = useModelCatalog();
  const [selectedPath, setSelectedPath] = useState("");
  const [query, setQuery] = useState("");
  const [cleanupPlan, setCleanupPlan] = useState<CleanupPlan | null>(null);
  const queryClient = useQueryClient();
  const inventoryItems = storage.data?.cache_items ?? [];
  const fallbackItems = useMemo(() => (catalog.data?.items ?? []).filter((model) => model.mlx_variant).map((model, index): ArtifactLocation => ({ id: `catalog-${index}`, path: model.mlx_variant!, kind: "cache", available: undefined })), [catalog.data?.items]);
  const items = inventoryItems.length ? inventoryItems : fallbackItems;
  const visible = items.filter((item) => item.path.toLowerCase().includes(query.trim().toLowerCase()));
  useEffect(() => { if (!selectedPath || !items.some((item) => item.path === selectedPath)) setSelectedPath(visible[0]?.path ?? ""); }, [items, selectedPath, visible]);
  const selected = items.find((item) => item.path === selectedPath) ?? null;
  const previewCleanup = useMutation({ mutationFn: () => api.previewCleanup({ include_temporary: true, include_trash: true, older_than_days: 7 }), onSuccess: setCleanupPlan });
  const executeCleanup = useMutation({ mutationFn: ({ id, note }: { id: string; note: string }) => api.executeCleanup(id, note), onSuccess: () => { setCleanupPlan(null); queryClient.invalidateQueries({ queryKey: queryKeys.storage }); queryClient.invalidateQueries({ queryKey: queryKeys.activity }); } });
  return (
    <div>
      <StorageOverview storage={storage.data ?? null} loading={storage.isLoading} onReview={() => previewCleanup.mutate()} reviewing={previewCleanup.isPending} />
      {cleanupPlan ? <CleanupReview plan={cleanupPlan} onCancel={() => setCleanupPlan(null)} onApprove={(note) => executeCleanup.mutate({ id: cleanupPlan.id, note })} pending={executeCleanup.isPending} error={executeCleanup.error?.message ?? previewCleanup.error?.message ?? null} /> : null}
      <div className="grid min-h-[calc(100vh-266px)] lg:grid-cols-[minmax(340px,0.9fr)_minmax(380px,1.1fr)]">
        <section className="border-b border-border lg:border-b-0 lg:border-r"><WorkspaceToolbar title="Local model cache" detail={inventoryItems.length ? `${inventoryItems.length} indexed locations` : "Catalog hints until cache indexing is available"} query={query} onQuery={setQuery} placeholder="Search local model path…" />
          <div className="divide-y divide-border-subtle">{storage.isLoading && catalog.isLoading ? <WorkspaceMessage icon={Loader2} spin label="Reading storage inventory" /> : visible.length ? visible.map((location) => <button key={location.id} type="button" onClick={() => setSelectedPath(location.path)} className={cn("relative flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-surface/55", selectedPath === location.path && "bg-accent-bg/55")} >{selectedPath === location.path ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}<div className="grid h-7 w-7 place-items-center rounded-md border border-border-subtle bg-surface text-fg-subtle"><HardDrive className="h-3.5 w-3.5" /></div><div className="min-w-0 flex-1"><div className="truncate font-mono text-[11px] text-fg">{location.path}</div><div className="mt-1 text-[9.5px] uppercase tracking-wide text-fg-disabled">{location.kind ?? "cache"}{location.available === false ? " · missing" : ""}</div></div><ChevronRight className="h-3.5 w-3.5 text-fg-disabled" /></button>) : <WorkspaceMessage icon={HardDrive} label="No cache entries indexed" detail="Downloaded model locations appear after the storage inventory refreshes." />}</div>
        </section>
        <aside className="bg-bg-subtle/15">{selected ? <CacheInspector location={selected} storage={storage.data ?? null} indexed={inventoryItems.length > 0} /> : <WorkspaceMessage icon={HardDrive} label="Select a cached model" detail="Storage location and safe cleanup context appear here." />}</aside>
      </div>
    </div>
  );
}

function StorageOverview({ storage, loading, onReview, reviewing }: { storage: StorageInventory | null; loading: boolean; onReview: () => void; reviewing: boolean }) {
  const used = storage?.used_bytes ?? null;
  const total = storage?.total_bytes ?? null;
  const percent = used != null && total ? Math.min(100, used / total * 100) : null;
  return <section className="border-b border-border bg-bg-subtle/35 px-5 py-3"><div className="flex flex-wrap items-center gap-x-6 gap-y-3"><div className="min-w-[180px] flex-1"><div className="flex items-center justify-between text-[10px]"><span className="uppercase tracking-wider text-fg-disabled">Workstation storage</span><span className={cn("font-mono", storage?.low_disk ? "text-warning" : "text-fg-muted")}>{loading ? "Measuring…" : storage ? `${formatBytes(storage.free_bytes)} free` : "Inventory unavailable"}</span></div><div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-pressed"><div className={cn("h-full transition-[width] duration-500", storage?.low_disk ? "bg-warning" : "bg-accent")} style={{ width: `${percent ?? 0}%` }} /></div></div><StorageReadout label="ARTIFACTS" value={formatBytes(storage?.artifact_bytes)} /><StorageReadout label="CACHE" value={formatBytes(storage?.cache_bytes)} /><StorageReadout label="TEMP" value={formatBytes(storage?.temporary_bytes)} /><StorageReadout label="TRASH" value={formatBytes(storage?.trash_bytes)} /><Button size="sm" variant="ghost" onClick={onReview} disabled={reviewing}>{reviewing ? <Loader2 className="animate-spin" /> : <Trash2 />} Review cleanup</Button></div>{storage?.low_disk ? <div className="mt-2 text-[10.5px] text-warning">Heavy work is blocked until projected free space clears the workstation minimum or an operator records an override.</div> : null}</section>;
}

function CleanupReview({ plan, onCancel, onApprove, pending, error }: { plan: CleanupPlan; onCancel: () => void; onApprove: (note: string) => void; pending: boolean; error: string | null }) {
  const safe = plan.items.filter((item) => !item.protected);
  const [reviewNote, setReviewNote] = useState("");
  return <section className="border-b border-warning/30 bg-warning-bg/45 px-5 py-4"><div className="flex flex-wrap items-start justify-between gap-4"><div><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-warning">Cleanup review</div><h3 className="mt-1 text-[14px] font-medium text-fg">{safe.length} item{safe.length === 1 ? "" : "s"} · {formatBytes(plan.reclaimable_bytes)} reclaimable</h3><p className="mt-1 text-[10.5px] text-fg-subtle">Protected and lineage-required artifacts remain untouched. Removed files stay in trash for {plan.trash_retention_days ?? 7} days.</p></div></div><div className="mt-3 grid gap-3 sm:grid-cols-[minmax(260px,1fr)_auto] sm:items-end"><FormField label="Review note · required"><Input value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="Why this cleanup is safe to approve" className="h-8 text-[11px]" /></FormField><div className="flex gap-2"><Button size="sm" variant="ghost" onClick={onCancel}>Cancel</Button><Button size="sm" onClick={() => onApprove(reviewNote.trim())} disabled={pending || !safe.length || !reviewNote.trim()}>{pending ? <Loader2 className="animate-spin" /> : <Trash2 />} Move to trash</Button></div></div>{error ? <InlineNotice tone="danger">{error}</InlineNotice> : null}</section>;
}

function CacheInspector({ location, storage, indexed }: { location: ArtifactLocation; storage: StorageInventory | null; indexed: boolean }) {
  return <div className="mx-auto max-w-3xl"><InspectorHeader eyebrow="Cached model" title={location.path.split(/[\\/]/).filter(Boolean).pop() || location.path} subtitle={location.path} badges={<Badge tone={location.available === false ? "danger" : indexed ? "success" : "neutral"} dot size="sm">{location.available === false ? "missing" : indexed ? "indexed" : "catalog hint"}</Badge>} /><section className="border-b border-border-subtle px-5 py-4"><div className="flex flex-wrap gap-2"><Button size="sm" asChild><Link to="/models" search={{ tab: "serve" }}><Server /> Serve & Test</Link></Button><Button size="sm" variant="ghost" asChild><Link to="/train"><Play /> Use in Train</Link></Button></div></section><InspectorSection title="Location"><KeyValue label="Path" value={location.path} mono copyable /><KeyValue label="Kind" value={location.kind ?? "cache"} /><KeyValue label="Verified" value={location.verified_at ? relativeTime(location.verified_at) : "Not verified by artifact library"} /><KeyValue label="Workspace free" value={formatBytes(storage?.free_bytes)} /></InspectorSection><InlineNoticeBlock>Cleanup is always reviewed. Active, pinned, promoted, serving, evaluation-referenced, and lineage-required content is protected automatically.</InlineNoticeBlock></div>;
}

/* ----------------------------------------------------------------------
 * Serve and test
 * ------------------------------------------------------------------- */

const SESSION_STORAGE_KEY = "halo-forge:playground-sessions:v4";
const PLAYGROUND_DRAFT_STORAGE_KEY = "halo-forge:playground-reviewed-drafts:v4";

function ServeTestWorkspace() {
  const queryClient = useQueryClient();
  const requestedArtifact = Route.useSearch().artifact;
  const catalog = useModelCatalog();
  const artifacts = useModelArtifacts({ limit: 200 });
  const status = useServeStatus();
  const start = useServeStart();
  const stop = useServeStop();
  const remoteSessions = useQuery({ queryKey: ["playground-sessions"], queryFn: () => api.listPlaygroundSessions({ limit: 100 }), retry: false });
  const [sessions, setSessions] = useState<PlaygroundSession[]>(loadSessions);
  const [activeId, setActiveId] = useState(() => sessions[0]?.id ?? "");
  const [input, setInput] = useState("");
  const [target, setTarget] = useState("");
  const [compareTarget, setCompareTarget] = useState("");
  const [seed, setSeed] = useState(42);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);
  const [selectedMessages, setSelectedMessages] = useState<string[]>([]);
  const [reviewNote, setReviewNote] = useState("");
  const [draftFeedback, setDraftFeedback] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const hydratedRemote = useRef(false);
  const active = sessions.find((session) => session.id === activeId) ?? null;
  const remoteAvailable = remoteSessions.isSuccess;
  const targets = useMemo(() => [
    ...(artifacts.data?.items ?? []).map((item) => ({ key: `artifact:${item.id}`, label: `${artifactAliasNames(item)[0] ? `${artifactAliasNames(item)[0]} · ` : ""}${item.model_name || prettyKind(item.kind)}`, value: item.path, backend: item.format === "mlx" ? "mlx" : undefined })),
    ...(catalog.data?.items ?? []).map((item) => ({ key: `catalog:${item.id}`, label: item.label || item.id, value: item.mlx_variant ?? item.id, backend: item.mlx_variant ? "mlx" : undefined })),
  ], [artifacts.data?.items, catalog.data?.items]);

  useEffect(() => {
    if (!remoteSessions.isSuccess || hydratedRemote.current) return;
    hydratedRemote.current = true;
    const remoteItems = remoteSessions.data?.items ?? [];
    setSessions(remoteItems);
    setActiveId((current) => remoteItems.some((session) => session.id === current) ? current : remoteItems[0]?.id ?? "");
  }, [remoteSessions.data?.items, remoteSessions.isSuccess]);
  useEffect(() => {
    if (remoteSessions.isLoading || remoteSessions.isSuccess || sessions.length) return;
    const session = newSession(1);
    setSessions([session]);
    setActiveId(session.id);
  }, [remoteSessions.isLoading, remoteSessions.isSuccess, sessions.length]);
  useEffect(() => { try { window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions)); } catch { /* private storage */ } }, [sessions]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.messages.length]);
  useEffect(() => { setSelectedMessages([]); setReviewNote(""); setDraftFeedback(null); if (!active) return; const storedTarget = typeof active.settings?.target === "string" ? active.settings.target : ""; const storedCompare = typeof active.settings?.compare_target === "string" ? active.settings.compare_target : ""; if (storedTarget) setTarget(storedTarget); setCompareTarget(storedCompare); setSeed(active.seed ?? 42); setTemperature(asFiniteNumber(active.generation_settings?.temperature, 0.7)); setMaxTokens(asFiniteNumber(active.generation_settings?.max_tokens, 256)); }, [active?.id]);
  useEffect(() => {
    if (!requestedArtifact || target) return;
    const key = `artifact:${requestedArtifact}`;
    if (targets.some((item) => item.key === key)) setTarget(key);
  }, [requestedArtifact, target, targets]);
  useEffect(() => {
    if (target || !targets.length) return;
    const serving = targets.find((item) => item.value === status.data?.model);
    setTarget(serving?.key ?? targets[0].key);
  }, [status.data?.model, target, targets]);

  const chat = useMutation({
    mutationFn: async ({ messages }: { messages: PlaygroundMessage[] }) => api.playgroundChat({ messages, model: status.data?.model ?? undefined, serve_url: status.data?.url, temperature, max_tokens: maxTokens }),
  });

  function updateActive(updater: (session: PlaygroundSession) => PlaygroundSession) {
    setSessions((values) => values.map((session) => session.id === activeId ? { ...updater(session), updated_at: new Date().toISOString() } : session));
  }
  function replaceSession(next: PlaygroundSession) { setSessions((values) => values.some((session) => session.id === next.id) ? values.map((session) => session.id === next.id ? next : session) : [next, ...values]); }
  async function addSession() {
    const draft = newSession(sessions.length + 1);
    if (remoteAvailable) {
      try {
        const created = await api.createPlaygroundSession({ id: draft.id, name: draft.name, seed: draft.seed, generation_settings: draft.generation_settings, settings: {} });
        replaceSession(created);
        setActiveId(created.id);
        return;
      } catch (error) {
        if (!isUnavailablePlaygroundEndpoint(error)) setDraftFeedback(error instanceof Error ? error.message : "Session could not be saved to the workstation service.");
      }
    }
    setSessions((values) => [draft, ...values]);
    setActiveId(draft.id);
  }
  async function removeSession(id: string) {
    if (remoteAvailable) {
      try { await api.archivePlaygroundSession(id); } catch (error) { if (!isUnavailablePlaygroundEndpoint(error)) { setDraftFeedback(error instanceof Error ? error.message : "Session could not be archived."); return; } }
    }
    setSessions((values) => values.filter((session) => session.id !== id));
    if (id === activeId) setActiveId(sessions.find((session) => session.id !== id)?.id ?? "");
  }
  async function persistActivePatch(patch: Partial<Pick<PlaygroundSession, "name" | "artifact_id" | "compare_artifact_id" | "endpoint" | "seed" | "generation_settings" | "settings">>) {
    if (!active || !remoteAvailable) return;
    try { replaceSession(await api.updatePlaygroundSession(active.id, patch)); } catch (error) { if (!isUnavailablePlaygroundEndpoint(error)) setDraftFeedback(error instanceof Error ? error.message : "Session metadata could not be saved."); }
  }
  function startTarget(key = target) {
    const selected = targets.find((item) => item.key === key);
    if (!selected) return;
    start.mutate({ model: selected.value, backend: selected.backend }, { onSuccess: () => { const patch = { artifact_id: artifactOccurrenceId(key), compare_artifact_id: artifactOccurrenceId(compareTarget), endpoint: status.data?.url ?? null, seed, generation_settings: { temperature, max_tokens: maxTokens }, settings: { ...(active?.settings ?? {}), target: key, compare_target: compareTarget || null } }; updateActive((session) => ({ ...session, ...patch })); void persistActivePatch(patch); }, onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.serve }) });
  }
  async function send() {
    const content = input.trim();
    if (!active || !content || chat.isPending || !status.data?.running) return;
    const userMessage: PlaygroundSessionMessage = { id: localMessageId(), role: "user", content, artifact_id: artifactOccurrenceId(target), generation: { seed, temperature, max_tokens: maxTokens, model: status.data?.model } };
    let next = [...active.messages, userMessage];
    updateActive((session) => ({ ...session, messages: next, seed, generation_settings: { temperature, max_tokens: maxTokens, model: status.data?.model } }));
    setInput("");
    try {
      if (remoteAvailable) {
        try { const saved = await api.appendPlaygroundMessage(active.id, userMessage); replaceSession(saved); next = saved.messages; } catch (error) { if (!isUnavailablePlaygroundEndpoint(error)) throw error; }
      }
      const response = await chat.mutateAsync({ messages: next });
      const returned = response.choices?.[0]?.message ?? { role: "assistant" as const, content: response.message || "No response returned." };
      const assistant: PlaygroundSessionMessage = { ...returned, id: localMessageId(), artifact_id: artifactOccurrenceId(target), generation: { seed, temperature, max_tokens: maxTokens, model: status.data?.model }, evidence: { finish_reason: response.choices?.[0]?.finish_reason ?? null } };
      if (remoteAvailable) { try { replaceSession(await api.appendPlaygroundMessage(active.id, assistant)); } catch (error) { if (!isUnavailablePlaygroundEndpoint(error)) throw error; else updateActive((session) => ({ ...session, messages: [...session.messages, assistant] })); } } else updateActive((session) => ({ ...session, messages: [...session.messages, assistant] }));
    } catch (error) {
      const failed: PlaygroundSessionMessage = { id: localMessageId(), role: "assistant", content: error instanceof Error ? `Error: ${error.message}` : "The endpoint did not respond.", evidence: { error: true } };
      updateActive((session) => ({ ...session, messages: [...session.messages, failed] }));
    }
  }
  async function createReviewedDraft(kind: "benchmark_suite" | "dataset_source") {
    if (!active || !selectedMessages.length || !reviewNote.trim()) return;
    const selected = active.messages.filter((message, index) => selectedMessages.includes(messageKey(message, index)));
    const messageIds = selected.flatMap((message) => message.id ? [message.id] : []);
    if (remoteAvailable && messageIds.length === selected.length) {
      try {
        const result = await api.reviewPlaygroundSession(active.id, { kind, message_ids: messageIds, review_note: reviewNote.trim() });
        const identity = result.benchmark_suite_revision_id ?? result.dataset_source_draft_id ?? result.id;
        setDraftFeedback(`${kind === "benchmark_suite" ? "Benchmark-suite" : "Dataset Lab source"} draft created${identity ? ` · ${identity}` : ""}. It will not start training automatically.`);
        setSelectedMessages([]);
        setReviewNote("");
        return;
      } catch (error) {
        if (!isUnavailablePlaygroundEndpoint(error)) {
          setDraftFeedback(error instanceof Error ? error.message : "The reviewed draft could not be created.");
          return;
        }
      }
    }
    const draft = { id: globalThis.crypto?.randomUUID?.() ?? `draft-${Date.now()}`, kind, status: "reviewed_draft", session_id: active.id, session_name: active.name, artifact_id: active.artifact_id ?? null, review_note: reviewNote.trim(), messages: selected, created_at: new Date().toISOString() };
    try { const current = JSON.parse(window.localStorage.getItem(PLAYGROUND_DRAFT_STORAGE_KEY) ?? "[]"); window.localStorage.setItem(PLAYGROUND_DRAFT_STORAGE_KEY, JSON.stringify([draft, ...(Array.isArray(current) ? current : [])])); setDraftFeedback(`${kind === "benchmark_suite" ? "Benchmark-suite" : "Dataset Lab source"} draft saved for review. It will not start training automatically.`); setSelectedMessages([]); setReviewNote(""); } catch { setDraftFeedback("This browser could not store the reviewed draft."); }
  }

  return (
    <div className="grid min-h-[calc(100vh-154px)] xl:grid-cols-[230px_minmax(420px,1fr)_300px]">
      <aside className="border-b border-border bg-bg-subtle/25 xl:border-b-0 xl:border-r">
        <div className="flex items-center justify-between border-b border-border-subtle px-3 py-3"><div><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Sessions</div><div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-fg-subtle"><span className={cn("h-1.5 w-1.5 rounded-full", remoteAvailable ? "bg-success" : remoteSessions.isLoading ? "animate-pulse bg-warning" : "bg-fg-disabled")} />{remoteAvailable ? "Durable workstation history" : remoteSessions.isLoading ? "Checking session service" : "Local fallback active"}</div></div><Button size="icon" variant="ghost" onClick={() => void addSession()} title="New session"><Plus /></Button></div>
        <div className="divide-y divide-border-subtle">{sessions.map((session) => <div key={session.id} className={cn("group relative flex items-center", session.id === activeId && "bg-accent-bg/55")}><button type="button" onClick={() => setActiveId(session.id)} className="min-w-0 flex-1 px-3 py-3 text-left"><div className="truncate text-[11.5px] font-medium text-fg">{session.name}</div><div className="mt-1 text-[9.5px] text-fg-disabled">{session.messages.length} messages · {session.updated_at ? relativeTime(session.updated_at) : "new"}</div></button><button type="button" onClick={() => void removeSession(session.id)} className="mr-2 grid h-7 w-7 place-items-center rounded-sm text-fg-disabled opacity-0 hover:bg-surface hover:text-danger group-hover:opacity-100 focus:opacity-100" aria-label={`Archive ${session.name}`}><Trash2 className="h-3 w-3" /></button>{session.id === activeId ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}</div>)}</div>
      </aside>

      <main className="flex min-h-[560px] min-w-0 flex-col border-b border-border xl:border-b-0 xl:border-r">
        <div className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-4 py-3">
          {active ? <Input value={active.name} onChange={(event) => updateActive((session) => ({ ...session, name: event.target.value }))} onBlur={() => void persistActivePatch({ name: active.name })} aria-label="Session name" className="h-8 min-w-[160px] max-w-[260px] border-transparent bg-transparent px-1 text-[13px] font-medium hover:border-border focus:border-accent" /> : null}
          <div className="ml-auto flex items-center gap-2"><Badge tone={status.data?.model_ready ? "success" : status.data?.running ? "warning" : "neutral"} dot size="sm">{status.data?.model_ready ? "chat ready" : status.data?.running ? "loading" : "not serving"}</Badge>{active?.compare_artifact_id ? <Badge tone="info" size="sm">sequential compare</Badge> : null}</div>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {active?.messages.length ? <div className="mx-auto max-w-3xl space-y-4">{active.messages.map((message, index) => { const key = messageKey(message, index); return <ChatMessage key={key} message={message} selected={selectedMessages.includes(key)} onToggle={() => setSelectedMessages((values) => values.includes(key) ? values.filter((value) => value !== key) : [...values, key])} />; })}{chat.isPending ? <div className="flex items-center gap-2 text-[11.5px] text-fg-muted"><Loader2 className="h-3.5 w-3.5 animate-spin text-accent" /> Generating response…</div> : null}<div ref={endRef} /></div> : <div className="grid min-h-[360px] place-items-center text-center"><div className="max-w-sm"><div className="mx-auto grid h-10 w-10 place-items-center rounded-full border border-border-subtle bg-surface text-accent"><MessageSquare className="h-4 w-4" /></div><h2 className="mt-4 text-[15px] font-medium text-fg">Test the active local model</h2><p className="mt-2 text-[11.5px] leading-relaxed text-fg-subtle">Sessions preserve prompts, generation settings, seed metadata, and the selected artifact. Start a target from the inspector, then send a message.</p></div></div>}
        </div>
        <div className="border-t border-border bg-bg-subtle/45 p-3"><div className="mx-auto flex max-w-3xl items-end gap-2"><textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void send(); } }} placeholder={status.data?.running ? "Message the active model…" : "Start a model from the inspector first"} rows={2} disabled={!status.data?.running} className="min-h-12 flex-1 resize-none rounded-md border border-border bg-bg px-3 py-2 text-[12px] leading-relaxed text-fg outline-none placeholder:text-fg-disabled focus:border-accent focus:ring-2 focus:ring-accent/25 disabled:opacity-50" /><Button size="icon" onClick={() => void send()} disabled={!input.trim() || !status.data?.running || chat.isPending} title="Send (⌘ Enter)">{chat.isPending ? <Loader2 className="animate-spin" /> : <Send />}</Button></div><div className="mx-auto mt-1.5 flex max-w-3xl justify-between text-[9.5px] text-fg-disabled"><span>⌘ Enter to send</span><span>seed {seed} · t={temperature.toFixed(2)} · max {maxTokens}</span></div></div>
      </main>

      <aside className="bg-bg-subtle/20">
        <div className="border-b border-border-subtle px-4 py-4"><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Serving target</div><div className="mt-3 space-y-2"><NativeSelect value={target} onChange={setTarget} ariaLabel="Serving target"><option value="">Choose a model or artifact</option><optgroup label="Trained artifacts">{targets.filter((item) => item.key.startsWith("artifact:")).map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</optgroup><optgroup label="Catalog">{targets.filter((item) => item.key.startsWith("catalog:")).map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</optgroup></NativeSelect>{status.data?.running ? <Button size="sm" variant="secondary" className="w-full" onClick={() => stop.mutate()} disabled={stop.isPending}>{stop.isPending ? <Loader2 className="animate-spin" /> : <Square />} Stop {status.data.model}</Button> : <Button size="sm" className="w-full" onClick={() => startTarget()} disabled={!target || start.isPending}>{start.isPending ? <Loader2 className="animate-spin" /> : <Server />} Start local service</Button>}</div>{start.error ? <InlineNotice tone="danger">{start.error.message}</InlineNotice> : null}{status.data?.message ? <p className="mt-3 text-[10.5px] leading-relaxed text-fg-subtle">{status.data.message}</p> : null}</div>
        <InspectorSection title="Sequential comparison"><p className="mb-2 text-[10.5px] leading-relaxed text-fg-subtle">Choose a candidate, finish the base conversation, then switch the managed service while keeping this session.</p><NativeSelect value={compareTarget} onChange={(value) => { setCompareTarget(value); const patch = { compare_artifact_id: artifactOccurrenceId(value), settings: { ...(active?.settings ?? {}), target, compare_target: value || null } }; updateActive((session) => ({ ...session, ...patch })); void persistActivePatch(patch); }} ariaLabel="Comparison target"><option value="">No comparison target</option>{targets.filter((item) => item.key !== target).map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</NativeSelect>{compareTarget ? <Button size="sm" variant="ghost" className="mt-2 w-full" onClick={() => { setTarget(compareTarget); startTarget(compareTarget); }}><RefreshCw /> Switch to candidate</Button> : null}</InspectorSection>
        <InspectorSection title="Generation settings"><FormField label="Seed metadata"><Input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} mono className="h-8 text-[11px]" /></FormField><FormField label={`Temperature · ${temperature.toFixed(2)}`} className="mt-3"><input type="range" min={0} max={2} step={0.05} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} className="w-full accent-[var(--color-accent)]" /></FormField><FormField label="Maximum output tokens" className="mt-3"><Input type="number" min={1} max={8192} value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} mono className="h-8 text-[11px]" /></FormField></InspectorSection>
        <InspectorSection title="Review selected turns"><p className="mb-2 text-[10.5px] leading-relaxed text-fg-subtle">Select complete turns in the conversation, add a review note, then create a draft. Drafts never launch training.</p>{active ? <Button size="sm" variant="ghost" asChild className="mb-2 w-full"><Link to="/datasets/review" search={{ new: "1", source: "playground_session", sourceRef: active.id, baseRef: undefined }}><CheckCircle2 />Create review proposal</Link></Button> : null}<div className="mb-2 font-mono text-[9.5px] text-fg-disabled">{selectedMessages.length} message{selectedMessages.length === 1 ? "" : "s"} selected</div><textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} rows={2} placeholder="Required review note" className="w-full resize-none rounded-md border border-border bg-bg px-2.5 py-2 text-[10.5px] text-fg outline-none focus:border-accent" /><div className="mt-2 grid gap-1.5"><Button size="sm" variant="secondary" onClick={() => void createReviewedDraft("benchmark_suite")} disabled={!selectedMessages.length || !reviewNote.trim()}><ShieldCheck /> Benchmark draft</Button><Button size="sm" variant="ghost" onClick={() => void createReviewedDraft("dataset_source")} disabled={!selectedMessages.length || !reviewNote.trim()}><Boxes /> Data source draft</Button></div>{draftFeedback ? <InlineNotice tone={draftFeedback.includes("could not") ? "danger" : "success"}>{draftFeedback}</InlineNotice> : null}</InspectorSection>
        <PlaygroundPreferenceReview session={active} durable={remoteAvailable} />
        <details className="border-b border-border-subtle px-4 py-3"><summary className="cursor-pointer text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Endpoint details</summary><div className="mt-3"><KeyValue label="State" value={status.data?.ready_state ?? status.data?.state ?? "idle"} /><KeyValue label="URL" value={status.data?.url ?? "Not running"} mono /><KeyValue label="Backend" value={status.data?.backend ?? "Automatic"} /></div></details>
      </aside>
    </div>
  );
}

function ChatMessage({ message, selected, onToggle }: { message: PlaygroundSessionMessage; selected: boolean; onToggle: () => void }) {
  return <div className={cn("group flex items-start gap-2", message.role === "user" ? "justify-end" : "justify-start")}><label className={cn("mt-2 flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-sm border transition-opacity", message.role === "user" && "order-2", selected ? "border-accent bg-accent-bg text-accent opacity-100" : "border-border-subtle text-fg-disabled opacity-30 group-hover:opacity-100 focus-within:opacity-100")} title="Select for reviewed draft"><input type="checkbox" checked={selected} onChange={onToggle} className="sr-only" />{selected ? <Check className="h-3 w-3" /> : <CircleDashed className="h-3 w-3" />}</label><div className={cn("max-w-[85%] rounded-lg px-3 py-2.5 text-[12px] leading-relaxed", message.role === "user" ? "bg-accent text-accent-fg" : message.content.startsWith("Error:") ? "border border-danger/30 bg-danger-bg text-danger" : "border border-border-subtle bg-surface text-fg-muted")}><div className={cn("mb-1 flex items-center justify-between gap-3 text-[8.5px] font-medium uppercase tracking-[0.13em]", message.role === "user" ? "text-accent-fg/65" : "text-fg-disabled")}><span>{message.role}</span>{message.generation && typeof message.generation.seed === "number" ? <span className="font-mono normal-case tracking-normal">seed {message.generation.seed}</span> : null}</div><div className="whitespace-pre-wrap">{message.content}</div></div></div>;
}

type PreferenceSchemaOption = { id: string; label: string; taskType: "pairwise" | "ranking" };
type PersistedMessageOption = { id: string; label: string; message: PlaygroundSessionMessage; index: number };

function PlaygroundPreferenceReview({ session, durable }: { session: PlaygroundSession | null; durable: boolean }) {
  const queryClient = useQueryClient();
  const [schemaRevisionId, setSchemaRevisionId] = useState("");
  const [promptMessageId, setPromptMessageId] = useState("");
  const [baseMessageId, setBaseMessageId] = useState("");
  const [candidateMessageId, setCandidateMessageId] = useState("");
  const [reviewNote, setReviewNote] = useState("");
  const [createdQueueId, setCreatedQueueId] = useState<string | null>(null);
  const schemas = useQuery({
    queryKey: ["playground-preference-schema-revisions"],
    queryFn: async () => {
      const catalog = await api.listAnnotationSchemas({ limit: 200 });
      const revisions = await Promise.all(catalog.items.filter((schema) => !schema.archived).map(async (schema) => ({ schema, items: (await api.listAnnotationSchemaRevisions(schema.id, { limit: 200 })).items })));
      return revisions.flatMap(({ schema, items }) => items.filter((revision) => revision.task_type === "pairwise" || revision.task_type === "ranking").map((revision): PreferenceSchemaOption => ({ id: revision.id, label: `${schema.name} · r${revision.revision_number} · ${prettyKind(revision.task_type)}`, taskType: revision.task_type as "pairwise" | "ranking" })));
    },
    staleTime: 30_000,
  });
  const userMessages = useMemo(() => persistedPlaygroundMessages(session, "user", durable), [durable, session]);
  const assistantMessages = useMemo(() => persistedPlaygroundMessages(session, "assistant", durable), [durable, session]);
  const prompt = userMessages.find((option) => option.id === promptMessageId);
  const base = assistantMessages.find((option) => option.id === baseMessageId);
  const candidate = assistantMessages.find((option) => option.id === candidateMessageId);
  const selectedSchema = schemas.data?.find((schema) => schema.id === schemaRevisionId);
  const validation = preferencePairingValidation({ durable, session, selectedSchema, prompt, base, candidate, reviewNote });

  useEffect(() => {
    setSchemaRevisionId("");
    setPromptMessageId("");
    setBaseMessageId("");
    setCandidateMessageId("");
    setReviewNote("");
    setCreatedQueueId(null);
  }, [session?.id]);
  useEffect(() => {
    if (!session || !durable) return;
    setPromptMessageId((current) => userMessages.some((option) => option.id === current) ? current : suggestedPrompt(userMessages, assistantMessages[0])?.id ?? userMessages.at(-1)?.id ?? "");
    setBaseMessageId((current) => assistantMessages.some((option) => option.id === current) ? current : suggestedBaseResponse(session, assistantMessages)?.id ?? assistantMessages[0]?.id ?? "");
    setCandidateMessageId((current) => assistantMessages.some((option) => option.id === current) && current !== baseMessageId ? current : suggestedCandidateResponse(session, assistantMessages, baseMessageId)?.id ?? assistantMessages.find((option) => option.id !== baseMessageId)?.id ?? "");
  }, [assistantMessages, baseMessageId, durable, session, userMessages]);

  const create = useMutation({
    mutationFn: () => {
      if (!session || !selectedSchema || !prompt || !base || !candidate || validation) throw new Error(validation || "Complete the response pairing.");
      const pairing: PlaygroundReviewPairing = { prompt_message_id: prompt.id, base_message_id: base.id, candidate_message_id: candidate.id };
      return api.reviewPlaygroundSession(session.id, { kind: "review_queue", schema_revision_id: selectedSchema.id, review_note: reviewNote.trim(), name: `${session.name} · ${prettyKind(selectedSchema.taskType)} review`, pairings: [pairing] });
    },
    onSuccess: (result) => {
      setCreatedQueueId(result.review_queue_id ?? result.id ?? null);
      queryClient.invalidateQueries({ queryKey: ["review-queues"] });
    },
  });

  return <InspectorSection title="Compare base and candidate">
    <p className="mb-3 text-[10.5px] leading-relaxed text-fg-subtle">Bind one persisted user prompt to two real assistant responses. Halo Forge will create a preference queue without inferring or fabricating either candidate.</p>
    {!durable ? <InlineNotice tone="warning">Preference pairings require a durable workstation session. Reconnect the session service before creating this queue.</InlineNotice> : null}
    {durable ? <div className="space-y-3">
      <FormField label="Pairwise or ranking schema"><NativeSelect className="w-full" value={schemaRevisionId} onChange={setSchemaRevisionId} ariaLabel="Preference annotation schema"><option value="">Choose an immutable schema</option>{(schemas.data ?? []).map((schema) => <option key={schema.id} value={schema.id}>{schema.label}</option>)}</NativeSelect></FormField>
      <FormField label="Persisted user prompt"><NativeSelect className="w-full" value={promptMessageId} onChange={setPromptMessageId} ariaLabel="Persisted user prompt"><option value="">Choose the prompt</option>{userMessages.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</NativeSelect></FormField>
      <FormField label="Base assistant response"><NativeSelect className="w-full" value={baseMessageId} onChange={(value) => { setBaseMessageId(value); if (value === candidateMessageId) setCandidateMessageId(assistantMessages.find((option) => option.id !== value)?.id ?? ""); }} ariaLabel="Base assistant response"><option value="">Choose the base response</option>{assistantMessages.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</NativeSelect></FormField>
      <FormField label="Candidate assistant response"><NativeSelect className="w-full" value={candidateMessageId} onChange={setCandidateMessageId} ariaLabel="Candidate assistant response"><option value="">Choose the candidate response</option>{assistantMessages.map((option) => <option key={option.id} value={option.id} disabled={option.id === baseMessageId}>{option.label}</option>)}</NativeSelect></FormField>
      {base && candidate ? <div className="grid gap-2"><PairingPreview label="Base" message={base.message} /><PairingPreview label="Candidate" message={candidate.message} /></div> : null}
      <FormField label="Review note · required"><textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} rows={2} placeholder="What should the reviewer compare?" className="w-full resize-none rounded-md border border-border bg-bg px-2.5 py-2 text-[10.5px] text-fg outline-none focus:border-accent" /></FormField>
      {validation && schemaRevisionId && promptMessageId && baseMessageId && candidateMessageId ? <p role="alert" className="text-[9.5px] leading-4 text-danger">{validation}</p> : null}
      <Button size="sm" className="w-full" onClick={() => create.mutate()} disabled={Boolean(validation) || create.isPending}>{create.isPending ? <Loader2 className="animate-spin" /> : <CheckCircle2 />}Create preference review queue</Button>
      {create.error ? <InlineNotice tone="danger">{create.error.message}</InlineNotice> : null}
      {createdQueueId ? <div className="rounded-md border border-success/25 bg-success-bg px-3 py-2.5"><div className="text-[10.5px] font-medium text-success">Preference review queue created</div><p className="mt-1 text-[9.5px] leading-4 text-fg-subtle">The base and candidate remain unlabelled until you review the queue.</p><Button size="sm" variant="ghost" className="mt-2 w-full" asChild><Link to="/datasets/review/$queueId" params={{ queueId: createdQueueId }}>Open review queue<ArrowRight /></Link></Button></div> : null}
    </div> : null}
  </InspectorSection>;
}

function PairingPreview({ label, message }: { label: string; message: PlaygroundSessionMessage }) { return <div className="rounded-md border border-border-subtle bg-surface/45 px-3 py-2"><div className="flex items-center justify-between gap-2 text-[8.5px] font-medium uppercase tracking-wider text-fg-disabled"><span>{label}</span><span className="truncate font-mono normal-case tracking-normal">{String(message.generation?.model ?? message.artifact_id ?? "persisted response")}</span></div><p className="mt-1 line-clamp-3 whitespace-pre-wrap text-[10.5px] leading-4 text-fg-muted">{message.content}</p></div>; }

/* ----------------------------------------------------------------------
 * Shared primitives
 * ------------------------------------------------------------------- */

function WorkspaceToolbar({ title, detail, query, onQuery, placeholder, children }: { title: string; detail: string; query?: string; onQuery?: (value: string) => void; placeholder?: string; children?: ReactNode }) {
  return <header className="border-b border-border bg-bg-subtle/35 px-4 py-3"><div className="flex items-start justify-between gap-3"><div><h2 className="text-[12.5px] font-medium text-fg">{title}</h2><p className="mt-0.5 text-[9.5px] text-fg-disabled">{detail}</p></div></div>{onQuery ? <label className="relative mt-3 block"><Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-disabled" /><Input value={query} onChange={(event) => onQuery(event.target.value)} placeholder={placeholder} className="h-8 pl-8 text-[11px]" /></label> : null}{children ? <div className="mt-2 flex flex-wrap items-center gap-2">{children}</div> : null}</header>;
}

function InspectorHeader({ eyebrow, title, subtitle, badges }: { eyebrow: string; title: string; subtitle?: string; badges?: ReactNode }) {
  return <header className="border-b border-border px-5 py-5"><div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-accent">{eyebrow}</div><h2 className="mt-2 break-words text-[19px] font-medium leading-tight text-fg">{title}</h2>{subtitle ? <div className="mt-1 break-all font-mono text-[9.5px] text-fg-disabled">{subtitle}</div> : null}{badges ? <div className="mt-3 flex flex-wrap gap-1.5">{badges}</div> : null}</header>;
}

function InspectorSection({ title, children }: { title: string; children: ReactNode }) {
  return <section className="border-b border-border-subtle px-5 py-4"><h3 className="mb-2.5 text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">{title}</h3>{children}</section>;
}

function KeyValue({ label, value, mono, copyable }: { label: string; value: string; mono?: boolean; copyable?: boolean }) {
  const [copied, setCopied] = useState(false);
  return <div className="flex items-start justify-between gap-4 border-b border-border-subtle/70 py-1.5 last:border-0"><span className="shrink-0 text-[10.5px] text-fg-subtle">{label}</span><span className="flex min-w-0 items-start gap-1"><span className={cn("break-all text-right text-[10.5px] text-fg-muted", mono && "font-mono")}>{value}</span>{copyable ? <button type="button" onClick={async () => { try { await navigator.clipboard.writeText(value); setCopied(true); window.setTimeout(() => setCopied(false), 1200); } catch { /* clipboard may be unavailable */ } }} className="mt-0.5 shrink-0 text-fg-disabled hover:text-accent" aria-label={`Copy ${label}`}>{copied ? <Check className="h-3 w-3 text-success" /> : <Copy className="h-3 w-3" />}</button> : null}</span></div>;
}

function WorkspaceMessage({ icon: Icon, label, detail, spin, tone, action }: { icon: typeof Search; label: string; detail?: string; spin?: boolean; tone?: "danger"; action?: ReactNode }) {
  return <div className="grid min-h-52 place-items-center px-6 py-10 text-center"><div><Icon className={cn("mx-auto h-4 w-4 text-fg-disabled", spin && "animate-spin", tone === "danger" && "text-danger")} /><p className={cn("mt-2 text-[12px] text-fg-muted", tone === "danger" && "text-danger")}>{label}</p>{detail ? <p className="mx-auto mt-1 max-w-[34ch] text-[10.5px] leading-relaxed text-fg-disabled">{detail}</p> : null}{action ? <div className="mt-3">{action}</div> : null}</div></div>;
}

function NativeSelect({ value, onChange, children, ariaLabel, className }: { value: string; onChange: (value: string) => void; children: ReactNode; ariaLabel: string; className?: string }) {
  return <select value={value} onChange={(event) => onChange(event.target.value)} aria-label={ariaLabel} className={cn("h-8 min-w-0 rounded-md border border-border bg-bg-subtle px-2 text-[10.5px] text-fg outline-none hover:border-border-strong focus:border-accent focus:ring-2 focus:ring-accent/25", className)}>{children}</select>;
}

function FormField({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return <label className={cn("block", className)}><span className="mb-1.5 block text-[9.5px] font-medium uppercase tracking-[0.1em] text-fg-disabled">{label}</span>{children}</label>;
}

function SmallChip({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "success" | "accent" }) {
  return <span className={cn("rounded-sm border px-1.5 py-0.5 text-[9.5px]", tone === "success" ? "border-success/25 bg-success-bg text-success" : tone === "accent" ? "border-accent/25 bg-accent-bg text-accent" : "border-border-subtle text-fg-subtle")}>{children}</span>;
}

function InlineNotice({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "success" | "warning" | "danger" }) {
  return <div className={cn("mt-3 rounded-sm border px-3 py-2 text-[10.5px] leading-relaxed", tone === "success" ? "border-success/25 bg-success-bg text-success" : tone === "warning" ? "border-warning/25 bg-warning-bg text-warning" : tone === "danger" ? "border-danger/25 bg-danger-bg text-danger" : "border-border-subtle bg-surface text-fg-subtle")}>{children}</div>;
}

function InlineNoticeBlock({ children }: { children: ReactNode }) { return <div className="mx-5 my-4 rounded-sm border border-border-subtle bg-surface/45 px-3 py-2.5 text-[10.5px] leading-relaxed text-fg-subtle">{children}</div>; }

function StorageReadout({ label, value }: { label: string; value: string }) { return <div><div className="text-[8.5px] uppercase tracking-wider text-fg-disabled">{label}</div><div className="mt-0.5 font-mono text-[10.5px] text-fg-muted">{value}</div></div>; }

function ArtifactKindIcon({ kind }: { kind: string }) { const Icon = kind === "adapter" ? GitBranch : kind === "merged" ? Merge : kind === "converted" || kind === "quantized" ? Archive : kind === "export_bundle" ? FileArchive : Box; return <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-md border border-border-subtle bg-surface text-fg-subtle"><Icon className="h-3.5 w-3.5" /></div>; }

function artifactAliasNames(artifact: ModelArtifactOccurrence): string[] { return (artifact.aliases ?? []).map((alias) => typeof alias === "string" ? alias : alias.alias); }
function artifactName(artifact?: ModelArtifactOccurrence): string { return artifact?.model_name || artifact?.path.split(/[\\/]/).filter(Boolean).pop() || artifact?.id || "Unknown artifact"; }
function shortHash(hash: string | null | undefined): string { if (!hash) return "hash pending"; const value = hash.replace(/^sha256:/, ""); return `sha256:${value.slice(0, 10)}…${value.slice(-6)}`; }
function prettyKind(value: string): string { return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()); }
function operationTitle(action: Exclude<StudioAction, null>): string { return action === "bake" ? "Bake adapter into base" : action === "merge" ? "Combine adapter lineage" : action === "convert" ? "Convert and publish a new format" : action === "quantize" ? "Create a post-training quantized variant" : action === "qualify" ? "Run quality and performance gates" : "Build a portable local bundle"; }
function preferredTrainMode(model: ModelCatalogEntry): import("@/lib/api").TrainingMode { const preferred = ["sft", "raft", "dpo", "orpo", "rm", "grpo", "vlm", "audio", "reasoning", "agentic"].find((mode) => model.trainer_support.includes(mode)); return (preferred ?? "sft") as import("@/lib/api").TrainingMode; }
function isModelTab(value: unknown): value is ModelTab { return ["catalog", "artifacts", "cached", "serve"].includes(String(value)); }
function formatBytes(bytes: number | null | undefined): string { if (bytes == null || !Number.isFinite(bytes)) return "Not measured"; const units = ["B", "KB", "MB", "GB", "TB"]; let value = Math.max(0, bytes); let unit = 0; while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; } return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`; }
function optionalNumber(value: string): number | null { if (!value.trim()) return null; const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; }
function suitesForPurpose(items: BenchmarkSuite[], purpose: "development" | "operational" | "holdout"): BenchmarkSuite[] { return items.filter((suite) => suite.purpose === purpose); }

type QualificationAxis = { label: string; delta: string; detail: string; tone: "good" | "bad" | "neutral" };

function buildQualificationAxes(comparison: QualificationComparison | undefined, baseline: ArtifactQualification | undefined, candidate: ArtifactQualification | undefined, baseArtifact: ModelArtifactOccurrence | undefined, candidateArtifact: ModelArtifactOccurrence): QualificationAxis[] {
  const qualityDeltas = (comparison?.deltas ?? []).filter((item) => item.stage !== "operational" && typeof item.favorable_delta === "number").map((item) => item.favorable_delta as number);
  const quality = qualityDeltas.length ? qualityDeltas.reduce((sum, value) => sum + value, 0) / qualityDeltas.length : null;
  const baseSpeed = qualificationMetric(baseline, ["output_tokens_per_second", "tokens_per_second"]);
  const candidateSpeed = qualificationMetric(candidate, ["output_tokens_per_second", "tokens_per_second"]);
  const baseMemory = qualificationMetric(baseline, ["peak_device_memory_bytes", "peak_process_memory_bytes", "peak_system_memory_bytes", "peak_memory_bytes"]);
  const candidateMemory = qualificationMetric(candidate, ["peak_device_memory_bytes", "peak_process_memory_bytes", "peak_system_memory_bytes", "peak_memory_bytes"]);
  const baseSize = baseArtifact?.size_bytes ?? baseArtifact?.blob?.size_bytes ?? null;
  const candidateSize = candidateArtifact.size_bytes ?? candidateArtifact.blob?.size_bytes ?? null;
  return [
    axisFromDelta("Quality", quality, quality == null ? "No comparable quality delta" : `${qualityDeltas.length} favorable gate delta${qualityDeltas.length === 1 ? "" : "s"}`),
    relativeAxis("Speed", baseSpeed, candidateSpeed, false, "tok/s"),
    relativeAxis("Memory", baseMemory, candidateMemory, true, "bytes", true),
    relativeAxis("Size", baseSize, candidateSize, true, "bytes", true),
  ];
}

function qualificationMetric(qualification: ArtifactQualification | undefined, keys: string[]): number | null {
  if (!qualification) return null;
  for (const key of keys) {
    const performance = qualification.performance?.[key];
    if (typeof performance === "number" && Number.isFinite(performance)) return performance;
    const metric = qualification.metrics?.[key];
    if (typeof metric === "number" && Number.isFinite(metric)) return metric;
  }
  return null;
}

function axisFromDelta(label: string, delta: number | null, detail: string): QualificationAxis {
  if (delta == null) return { label, delta: "Not measured", detail, tone: "neutral" };
  return { label, delta: `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`, detail, tone: delta > 0 ? "good" : delta < 0 ? "bad" : "neutral" };
}

function relativeAxis(label: string, baseline: number | null, candidate: number | null, lowerIsBetter: boolean, unit: string, bytes = false): QualificationAxis {
  if (baseline == null || candidate == null || baseline === 0) return { label, delta: "Not measured", detail: "Baseline and candidate evidence required", tone: "neutral" };
  const percent = (candidate - baseline) / Math.abs(baseline) * 100;
  const favorable = lowerIsBetter ? -percent : percent;
  const format = (value: number) => bytes ? formatBytes(value) : `${value.toFixed(1)} ${unit}`;
  return { label, delta: `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}%`, detail: `${format(baseline)} → ${format(candidate)}`, tone: favorable > 0 ? "good" : favorable < 0 ? "bad" : "neutral" };
}
function artifactOccurrenceId(targetKey: string): string | null { return targetKey.startsWith("artifact:") ? targetKey.slice("artifact:".length) || null : null; }
function asFiniteNumber(value: unknown, fallback: number): number { return typeof value === "number" && Number.isFinite(value) ? value : fallback; }
function localMessageId(): string { return globalThis.crypto?.randomUUID?.() ?? `message-${Date.now()}-${Math.random().toString(16).slice(2)}`; }
function messageKey(message: PlaygroundSessionMessage, index: number): string { return message.id || `${index}:${message.role}:${message.created_at ?? "local"}`; }
function persistedPlaygroundMessages(session: PlaygroundSession | null, role: "user" | "assistant", durable: boolean): PersistedMessageOption[] { if (!session || !durable) return []; return session.messages.flatMap((message, index) => message.role === role && message.id && message.content.trim() ? [{ id: message.id, label: playgroundMessageOptionLabel(message, index), message, index }] : []); }
function playgroundMessageOptionLabel(message: PlaygroundSessionMessage, index: number) { const identity = String(message.generation?.model ?? message.artifact_id ?? message.role); const content = message.content.replace(/\s+/g, " ").trim(); return `${index + 1} · ${identity} · ${content.length > 62 ? `${content.slice(0, 61)}…` : content}`; }
function suggestedBaseResponse(session: PlaygroundSession, responses: PersistedMessageOption[]) { return responses.find((option) => session.artifact_id && option.message.artifact_id === session.artifact_id) ?? responses[0]; }
function suggestedCandidateResponse(session: PlaygroundSession, responses: PersistedMessageOption[], baseId: string) { return responses.find((option) => option.id !== baseId && session.compare_artifact_id && option.message.artifact_id === session.compare_artifact_id) ?? responses.find((option) => option.id !== baseId); }
function suggestedPrompt(prompts: PersistedMessageOption[], response?: PersistedMessageOption) { if (!response) return prompts.at(-1); return [...prompts].reverse().find((prompt) => prompt.index < response.index) ?? prompts.at(-1); }
function preferencePairingValidation({ durable, session, selectedSchema, prompt, base, candidate, reviewNote }: { durable: boolean; session: PlaygroundSession | null; selectedSchema?: PreferenceSchemaOption; prompt?: PersistedMessageOption; base?: PersistedMessageOption; candidate?: PersistedMessageOption; reviewNote: string }): string | null { if (!durable) return "Reconnect the durable session service before creating a preference queue."; if (!session) return "Choose a Playground session."; if (!selectedSchema) return "Choose an immutable pairwise or ranking schema."; if (!prompt) return "Choose a persisted user prompt."; if (!base) return "Choose a persisted base assistant response."; if (!candidate) return "Choose a persisted candidate assistant response."; if (base.id === candidate.id) return "Base and candidate must be different persisted responses."; if (base.message.content === candidate.message.content) return "Base and candidate response text must differ for preference review."; if (!reviewNote.trim()) return "Add the required review note."; return null; }
function isUnavailablePlaygroundEndpoint(error: unknown): boolean { return error instanceof ApiError && [404, 405, 501].includes(error.status); }
function newSession(index: number): PlaygroundSession { const now = new Date().toISOString(); return { id: globalThis.crypto?.randomUUID?.() ?? `session-${Date.now()}-${index}`, name: `Session ${index}`, messages: [], seed: 42, generation_settings: { temperature: 0.7, max_tokens: 256 }, created_at: now, updated_at: now }; }
function loadSessions(): PlaygroundSession[] { if (typeof window === "undefined") return []; try { const value = JSON.parse(window.localStorage.getItem(SESSION_STORAGE_KEY) ?? "[]"); return Array.isArray(value) ? value.filter((item) => item && typeof item.id === "string" && Array.isArray(item.messages)) : []; } catch { return []; } }
