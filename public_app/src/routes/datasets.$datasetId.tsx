import { createFileRoute, Link, Outlet, useNavigate, useRouterState } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  CheckCircle2,
  Clock3,
  Database,
  FileSearch,
  GripVertical,
  Hammer,
  Loader2,
  RefreshCcw,
  RotateCcw,
  Rows3,
  Square,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Topbar } from "@/components/shell";
import { DataSectionTabs } from "@/components/data/data-section-tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  api,
  type DatasetJob,
  type DatasetPreview,
  type DatasetRecipe,
  type DatasetRecord,
  type DatasetSource,
  type DatasetVersion,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type DatasetTab = "overview" | "preview" | "build" | "versions";
type RecipeDraftStep = { id: string; kind: string; params: Record<string, unknown> };

export const Route = createFileRoute("/datasets/$datasetId")({
  component: DatasetDetailRoute,
  validateSearch: (search): { tab?: DatasetTab; recipeFrom?: string } => ({
    tab: isDatasetTab(search.tab) ? search.tab : "overview",
    recipeFrom: typeof search.recipeFrom === "string" ? search.recipeFrom : undefined,
  }),
});

function DatasetDetailRoute() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  if (pathname.includes("/versions/")) return <Outlet />;
  return <DatasetWorkspace />;
}

function DatasetWorkspace() {
  const { datasetId } = Route.useParams();
  const search = Route.useSearch();
  const activeTab = search.tab ?? "overview";
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const dataset = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: () => api.datasetDetail(datasetId),
    refetchInterval: 20_000,
  });
  const versions = useQuery({
    queryKey: ["datasets", datasetId, "versions"],
    queryFn: () => api.datasetVersions(datasetId),
    refetchInterval: activeTab === "versions" ? 10_000 : false,
  });
  const jobs = useQuery({
    queryKey: ["dataset-jobs", datasetId],
    queryFn: () => api.listDatasetJobs({ datasetId }),
    refetchInterval: (query) =>
      (query.state.data?.items ?? []).some((job) => isActiveJob(job.status)) ? 1_500 : 10_000,
    refetchIntervalInBackground: false,
  });
  const clonedVersion = useQuery({
    queryKey: ["dataset-versions", search.recipeFrom],
    queryFn: () => api.datasetVersion(search.recipeFrom!),
    enabled: Boolean(search.recipeFrom),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["datasets", datasetId] });
    queryClient.invalidateQueries({ queryKey: ["datasets", datasetId, "versions"] });
    queryClient.invalidateQueries({ queryKey: ["dataset-jobs", datasetId] });
  };

  if (dataset.isLoading) return <DetailLoading />;
  if (dataset.isError || !dataset.data) {
    return <DetailError message={(dataset.error as Error)?.message || "Dataset was not found."} onRetry={() => dataset.refetch()} />;
  }

  const record = dataset.data;
  const activeJob = chooseVisibleJob(jobs.data?.items ?? [], record.active_job);

  return (
    <>
      <Topbar
        eyebrow="Dataset Lab"
        title={record.name}
        subtitle={record.description || sourceSummary(record)}
        actions={
          <>
            <Button variant="ghost" size="icon" asChild aria-label="Back to datasets">
              <Link to="/datasets"><ArrowLeft /></Link>
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={() => navigate({ to: "/datasets/$datasetId", params: { datasetId }, search: { tab: "build", recipeFrom: undefined } })}
            >
              <Hammer />Build version
            </Button>
          </>
        }
        statusBar={
          <>
            <span>{record.modality || "text"}</span>
            <span className="text-fg-disabled">•</span>
            <span>{record.sources.length} source{record.sources.length === 1 ? "" : "s"}</span>
            <span className="text-fg-disabled">•</span>
            <span>{versions.data?.items.length ?? 0} versions</span>
            {record.latest_version ? (
              <>
                <span className="text-fg-disabled">•</span>
                <span>{formatInteger(record.latest_version.row_count)} rows</span>
              </>
            ) : null}
          </>
        }
        live={Boolean(activeJob && isActiveJob(activeJob.status))}
      />

      <DataSectionTabs />
      <DatasetTabs datasetId={datasetId} active={activeTab} />
      {activeJob ? <JobStrip job={activeJob} onChanged={invalidate} /> : null}

      {activeTab === "overview" ? (
        <OverviewTab dataset={record} versions={versions.data?.items ?? []} />
      ) : activeTab === "preview" ? (
        <PreviewTab datasetId={datasetId} />
      ) : activeTab === "build" ? (
        <BuildTab
          dataset={record}
          clonedRecipe={clonedVersion.data?.recipe}
          cloneLoading={clonedVersion.isLoading}
          onBuilt={invalidate}
        />
      ) : (
        <VersionsTab dataset={record} versions={versions.data?.items ?? []} loading={versions.isLoading} />
      )}
    </>
  );
}

function DatasetTabs({ datasetId, active }: { datasetId: string; active: DatasetTab }) {
  const tabs: Array<{ key: DatasetTab; label: string; icon: typeof Database }> = [
    { key: "overview", label: "Overview", icon: Database },
    { key: "preview", label: "Preview", icon: Rows3 },
    { key: "build", label: "Build", icon: Hammer },
    { key: "versions", label: "Versions", icon: Clock3 },
  ];
  return (
    <nav className="flex items-center gap-1 border-b border-border bg-bg-subtle px-5" aria-label="Dataset sections">
      {tabs.map((tab) => (
        <Link
          key={tab.key}
          to="/datasets/$datasetId"
          params={{ datasetId }}
          search={{ tab: tab.key, recipeFrom: undefined }}
          className={cn(
            "relative flex h-10 items-center gap-2 px-3 text-xs transition-colors",
            active === tab.key ? "text-accent" : "text-fg-muted hover:text-fg",
          )}
        >
          <tab.icon className="h-3.5 w-3.5" />
          {tab.label}
          {active === tab.key ? <span className="absolute inset-x-1 bottom-0 h-0.5 bg-accent" /> : null}
        </Link>
      ))}
    </nav>
  );
}

function OverviewTab({ dataset, versions }: { dataset: DatasetRecord; versions: DatasetVersion[] }) {
  const statistics = useQuery({
    queryKey: ["datasets", dataset.id, "statistics"],
    queryFn: () => api.datasetStatistics(dataset.id),
    retry: false,
  });
  const latest = dataset.latest_version || versions[0];
  return (
    <div className="grid min-h-[520px] grid-cols-1 divide-y divide-border-subtle lg:grid-cols-[minmax(0,1fr)_320px] lg:divide-x lg:divide-y-0">
      <div className="min-w-0">
        <SectionHeader title="Sources" detail="Registered inputs remain immutable references; builds create versioned outputs." />
        <div className="divide-y divide-border-subtle border-b border-border-subtle">
          {dataset.sources.map((source, index) => <DatasetSourceRow key={source.id || `${source.uri}-${index}`} datasetId={dataset.id} source={source} />)}
        </div>

        <SectionHeader title="Canonical schema" detail="Fields expected by downstream recipe steps and trainers." />
        <KeyValueTable value={schemaValue(dataset.canonical_schema)} />

        <SectionHeader title="Latest statistics" detail={latest ? `Version ${versionLabel(latest)}` : "Build a version to profile records."} />
        {statistics.isLoading ? (
          <InlineLoading label="Profiling statistics" />
        ) : statistics.isError ? (
          <InlineEmpty label="Statistics are not available until the first build completes." />
        ) : (
          <KeyValueTable value={statistics.data || latest?.statistics || {}} />
        )}
      </div>

      <aside className="bg-bg-subtle/45">
        <SectionHeader title="Current state" detail="Dataset-level build and storage summary." />
        <dl className="divide-y divide-border-subtle border-y border-border-subtle">
          <DefinitionRow label="Dataset ID" value={dataset.id} mono />
          <DefinitionRow label="Versions" value={String(versions.length)} mono />
          <DefinitionRow label="Latest rows" value={formatInteger(latest?.row_count)} mono />
          <DefinitionRow label="Latest size" value={formatBytes(latest?.size_bytes)} mono />
          <DefinitionRow label="Assets" value={latest?.assets_materialized ? "Materialized" : "Referenced"} />
          <DefinitionRow label="Created" value={formatDate(dataset.created_at)} />
          <DefinitionRow label="Updated" value={formatDate(dataset.updated_at)} />
        </dl>
        {latest ? (
          <div className="px-5 py-4">
            <Button variant="secondary" size="sm" className="w-full" asChild>
              <Link to="/datasets/$datasetId/versions/$versionId" params={{ datasetId: dataset.id, versionId: latest.id }} search={{ split: defaultSplit(latest) }}>
                Inspect latest version<ArrowRight />
              </Link>
            </Button>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function DatasetSourceRow({ datasetId, source }: { datasetId: string; source: DatasetSource }) {
  const queryClient = useQueryClient();
  const refresh = useMutation({
    mutationFn: () => {
      if (!source.id) throw new Error("This legacy source has no refresh identity.");
      return api.refreshDatasetSource(source.id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["dataset-jobs", datasetId] });
    },
  });
  return (
    <div className="grid gap-3 px-5 py-4 sm:grid-cols-[110px_minmax(0,1fr)_220px]">
      <Badge tone={source.kind === "huggingface" ? "info" : "neutral"} size="sm" className="w-fit">{source.kind}</Badge>
      <div className="min-w-0">
        <div className="truncate font-mono text-xs text-fg">{source.uri}</div>
        <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-fg-subtle">
          {source.config ? <span>config {source.config}</span> : null}
          {source.split ? <span>split {source.split}</span> : null}
          {source.revision ? <span>revision {source.revision}</span> : null}
        </div>
        {refresh.isSuccess ? <div className="mt-2 text-[10px] text-success">Refresh requested. Changed content will become a new source revision.</div> : null}
        {refresh.isError ? <div className="mt-2 text-[10px] text-danger">{(refresh.error as Error).message}</div> : null}
      </div>
      <div className="flex items-start justify-end gap-2">
        <span className="max-w-[120px] truncate pt-1.5 text-right font-mono text-[10px] text-fg-disabled" title={source.fingerprint || undefined}>{source.fingerprint || "fingerprint after build"}</span>
        <Button variant="ghost" size="sm" disabled={!source.id || refresh.isPending} onClick={() => refresh.mutate()} title="Reinspect the source. Changed content creates a new revision; existing versions never change.">
          <RefreshCcw className={cn(refresh.isPending && "animate-spin")} />Refresh source
        </Button>
      </div>
    </div>
  );
}

function PreviewTab({ datasetId }: { datasetId: string }) {
  const [offset, setOffset] = useState(0);
  const limit = 25;
  const preview = useQuery({
    queryKey: ["datasets", datasetId, "preview", offset],
    queryFn: () => api.datasetPreview(datasetId, { offset, limit }),
  });
  return (
    <div>
      <div className="flex items-center justify-between border-b border-border-subtle px-5 py-3">
        <div>
          <h2 className="text-xs font-medium text-fg">Source preview</h2>
          <p className="mt-0.5 text-[11px] text-fg-subtle">Raw records before recipe transforms.</p>
        </div>
        <Pagination offset={offset} limit={limit} total={preview.data?.total ?? 0} onChange={setOffset} />
      </div>
      {preview.isLoading ? <InlineLoading label="Loading source records" /> : preview.isError ? (
        <InlineError label={(preview.error as Error).message} onRetry={() => preview.refetch()} />
      ) : preview.data?.items.length ? (
        <PreviewTable preview={preview.data} />
      ) : <InlineEmpty label="No records were returned by this source." />}
    </div>
  );
}

function BuildTab({
  dataset,
  clonedRecipe,
  cloneLoading,
  onBuilt,
}: {
  dataset: DatasetRecord;
  clonedRecipe?: DatasetVersion["recipe"];
  cloneLoading: boolean;
  onBuilt: () => void;
}) {
  const [steps, setSteps] = useState<RecipeDraftStep[]>(() => [makeDraftStep("normalize"), makeDraftStep("validate"), makeDraftStep("split")]);
  const [selectedId, setSelectedId] = useState(steps[0]?.id ?? "");
  const [recipeName, setRecipeName] = useState("training-ready");
  const [seed, setSeed] = useState(42);
  const [jsonText, setJsonText] = useState(() => JSON.stringify(steps[0]?.params ?? {}, null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);
  const build = useMutation({
    mutationFn: (recipe: DatasetRecipe) => api.buildDataset(dataset.id, { recipe }),
    onSuccess: () => onBuilt(),
  });

  useEffect(() => {
    const recipe = normalizeRecipe(clonedRecipe);
    if (!recipe?.steps.length) return;
    const cloned = recipe.steps.map((step) => ({
      id: crypto.randomUUID(),
      kind: String(step.kind || "normalize"),
      params: Object.fromEntries(Object.entries(step).filter(([key]) => !["id", "kind", "type", "label", "enabled"].includes(key))),
    }));
    setSteps(cloned);
    setSelectedId(cloned[0]?.id ?? "");
    setJsonText(JSON.stringify(cloned[0]?.params ?? {}, null, 2));
    if (recipe.name) setRecipeName(recipe.name);
    if (typeof recipe.seed === "number") setSeed(recipe.seed);
  }, [clonedRecipe]);

  const selected = steps.find((step) => step.id === selectedId) || steps[0];

  function selectStep(step: RecipeDraftStep) {
    persistParams();
    setSelectedId(step.id);
    setJsonText(JSON.stringify(step.params, null, 2));
    setJsonError(null);
  }

  function persistParams(): boolean {
    if (!selected) return true;
    try {
      const value = JSON.parse(jsonText) as unknown;
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Parameters must be a JSON object.");
      setSteps((current) => current.map((step) => step.id === selected.id ? { ...step, params: value as Record<string, unknown> } : step));
      setJsonError(null);
      return true;
    } catch (error) {
      setJsonError((error as Error).message);
      return false;
    }
  }

  function addStep(kind: string) {
    persistParams();
    const step = makeDraftStep(kind);
    setSteps((current) => [...current, step]);
    setSelectedId(step.id);
    setJsonText(JSON.stringify(step.params, null, 2));
    setJsonError(null);
  }

  function removeStep(id: string) {
    const index = steps.findIndex((step) => step.id === id);
    const next = steps.filter((step) => step.id !== id);
    setSteps(next);
    const replacement = next[Math.min(index, Math.max(0, next.length - 1))];
    setSelectedId(replacement?.id ?? "");
    setJsonText(JSON.stringify(replacement?.params ?? {}, null, 2));
  }

  function moveStep(id: string, direction: -1 | 1) {
    setSteps((current) => {
      const index = current.findIndex((step) => step.id === id);
      const target = index + direction;
      if (index < 0 || target < 0 || target >= current.length) return current;
      const next = [...current];
      [next[index], next[target]] = [next[target]!, next[index]!];
      return next;
    });
  }

  function startBuild() {
    let resolvedSteps = steps;
    if (selected) {
      try {
        const value = JSON.parse(jsonText) as unknown;
        if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Parameters must be a JSON object.");
        resolvedSteps = steps.map((step) => step.id === selected.id ? { ...step, params: value as Record<string, unknown> } : step);
        setSteps(resolvedSteps);
        setJsonError(null);
      } catch (error) {
        setJsonError((error as Error).message);
        return;
      }
    }
    const recipeSteps = resolvedSteps.map((step) => ({ kind: step.kind, ...step.params }));
    build.mutate({
      name: recipeName.trim() || undefined,
      schema: typeof dataset.canonical_schema === "string" ? dataset.canonical_schema : undefined,
      seed,
      steps: recipeSteps,
    });
  }

  if (cloneLoading) return <InlineLoading label="Loading recipe" />;

  return (
    <div className="grid min-h-[620px] grid-cols-1 lg:grid-cols-[minmax(360px,0.9fr)_minmax(420px,1.1fr)]">
      <section className="border-b border-border-subtle lg:border-b-0 lg:border-r">
        <SectionHeader title="Recipe" detail="Steps run top to bottom and are recorded in version provenance." />
        <div className="grid grid-cols-[1fr_110px] gap-3 border-y border-border-subtle bg-bg-subtle/45 px-5 py-3">
          <div>
            <Label className="text-[10px] uppercase tracking-[0.1em] text-fg-disabled">Recipe name</Label>
            <Input value={recipeName} onChange={(event) => setRecipeName(event.target.value)} className="mt-1 h-8 text-xs" />
          </div>
          <div>
            <Label className="text-[10px] uppercase tracking-[0.1em] text-fg-disabled">Seed</Label>
            <Input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} mono className="mt-1 h-8 text-xs" />
          </div>
        </div>

        <ol className="divide-y divide-border-subtle">
          {steps.map((step, index) => (
            <li
              key={step.id}
              className={cn(
                "group flex items-center transition-colors",
                selected?.id === step.id ? "bg-accent-bg" : "hover:bg-surface/55",
              )}
            >
              <button
                type="button"
                onClick={() => selectStep(step)}
                className={cn(
                  "flex min-w-0 flex-1 items-center gap-3 py-3 pl-4 text-left",
                )}
              >
                <GripVertical className="h-4 w-4 text-fg-disabled" />
                <span className={cn("flex h-6 w-6 items-center justify-center rounded-sm border font-mono text-[10px]", selected?.id === step.id ? "border-accent text-accent" : "border-border text-fg-subtle")}>{index + 1}</span>
                <span className="min-w-0 flex-1">
                  <span className={cn("block text-xs font-medium", selected?.id === step.id ? "text-accent" : "text-fg")}>{stepLabel(step.kind)}</span>
                  <span className="mt-0.5 block truncate font-mono text-[10px] text-fg-subtle">{summarizeParams(step.params)}</span>
                </span>
              </button>
              <span className="mr-3 flex opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                <IconButton label="Move up" disabled={index === 0} onClick={() => moveStep(step.id, -1)}><ArrowUp /></IconButton>
                <IconButton label="Move down" disabled={index === steps.length - 1} onClick={() => moveStep(step.id, 1)}><ArrowDown /></IconButton>
                <IconButton label="Remove step" onClick={() => removeStep(step.id)}><Trash2 /></IconButton>
              </span>
            </li>
          ))}
        </ol>

        <div className="border-t border-border-subtle px-4 py-3">
          <select
            defaultValue=""
            onChange={(event) => { if (event.target.value) { addStep(event.target.value); event.target.value = ""; } }}
            className="h-8 w-full rounded-md border border-dashed border-border-strong bg-bg-subtle px-3 text-xs text-fg-muted hover:border-accent focus:border-accent focus:outline-none"
          >
            <option value="" disabled>+ Add recipe step</option>
            {RECIPE_KINDS.map((kind) => <option key={kind} value={kind}>{stepLabel(kind)}</option>)}
          </select>
        </div>
      </section>

      <section className="flex min-h-[520px] flex-col bg-bg-subtle/25">
        {selected ? (
          <>
            <div className="flex items-center justify-between border-b border-border-subtle px-5 py-4">
              <div>
                <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Step parameters</div>
                <h2 className="mt-0.5 text-sm font-medium text-fg">{stepLabel(selected.kind)}</h2>
              </div>
              <select
                value={selected.kind}
                onChange={(event) => {
                  const kind = event.target.value;
                  const params = defaultParams(kind);
                  setSteps((current) => current.map((step) => step.id === selected.id ? { ...step, kind, params } : step));
                  setJsonText(JSON.stringify(params, null, 2));
                }}
                className="h-8 rounded-md border border-border bg-bg px-2 text-xs text-fg"
              >
                {RECIPE_KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
              </select>
            </div>
            <div className="flex-1 px-5 py-4">
              <p className="mb-3 text-xs leading-5 text-fg-muted">{stepDescription(selected.kind)}</p>
              <textarea
                value={jsonText}
                onChange={(event) => setJsonText(event.target.value)}
                onBlur={() => persistParams()}
                spellCheck={false}
                aria-label={`${selected.kind} step parameters`}
                className={cn(
                  "min-h-72 w-full resize-y rounded-md border bg-bg p-3 font-mono text-xs leading-5 text-fg focus:outline-none focus:ring-2",
                  jsonError ? "border-danger focus:ring-danger/25" : "border-border focus:border-accent focus:ring-accent/25",
                )}
              />
              {jsonError ? <p className="mt-2 text-xs text-danger">{jsonError}</p> : null}
            </div>
          </>
        ) : (
          <InlineEmpty label="Add a step to begin the recipe." />
        )}
        <div className="flex items-center justify-between border-t border-border bg-bg px-5 py-3">
          <div className="text-[11px] text-fg-subtle">{steps.length} ordered step{steps.length === 1 ? "" : "s"} · source {dataset.sources[0]?.kind || "unknown"}</div>
          <Button variant="primary" size="sm" disabled={build.isPending || steps.length === 0 || Boolean(jsonError)} onClick={startBuild}>
            {build.isPending ? <Loader2 className="animate-spin" /> : <Hammer />}
            Build version
          </Button>
        </div>
        {build.isError ? <div className="border-t border-danger/30 bg-danger-bg px-5 py-2 text-xs text-danger">{(build.error as Error).message}</div> : null}
      </section>
    </div>
  );
}

function VersionsTab({ dataset, versions, loading }: { dataset: DatasetRecord; versions: DatasetVersion[]; loading: boolean }) {
  if (loading) return <InlineLoading label="Loading versions" />;
  if (!versions.length) return <InlineEmpty label="No versions yet. Build a recipe to create the first immutable output." />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[820px] text-left text-[12px]">
        <thead className="bg-bg-subtle text-[10px] uppercase tracking-[0.12em] text-fg-disabled">
          <tr className="border-b border-border">
            <th className="px-5 py-2 font-medium">Version</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 text-right font-medium">Rows</th>
            <th className="px-3 py-2 text-right font-medium">Size</th>
            <th className="px-3 py-2 font-medium">Splits</th>
            <th className="px-3 py-2 font-medium">Content hash</th>
            <th className="px-3 py-2 font-medium">Created</th>
            <th className="w-12 px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {versions.map((version) => (
            <tr key={version.id} className="group border-b border-border-subtle hover:bg-surface/55">
              <td className="px-5 py-3">
                <Link to="/datasets/$datasetId/versions/$versionId" params={{ datasetId: dataset.id, versionId: version.id }} search={{ split: defaultSplit(version) }} className="font-medium text-fg group-hover:text-accent">
                  {versionLabel(version)}
                </Link>
                <div className="mt-0.5 font-mono text-[10px] text-fg-disabled">{version.id}</div>
              </td>
              <td className="px-3 py-3"><Badge tone={versionTone(version.status)} dot size="sm">{version.status}</Badge></td>
              <td className="px-3 py-3 text-right font-mono text-fg">{formatInteger(version.row_count)}</td>
              <td className="px-3 py-3 text-right font-mono text-fg-muted">{formatBytes(version.size_bytes)}</td>
              <td className="px-3 py-3 font-mono text-[10px] text-fg-muted">{formatSplits(version.split_counts)}</td>
              <td className="max-w-40 truncate px-3 py-3 font-mono text-[10px] text-fg-subtle">{version.content_hash || "—"}</td>
              <td className="px-3 py-3 text-[11px] text-fg-subtle">{formatDate(version.created_at)}</td>
              <td className="px-3 py-3"><Button variant="ghost" size="icon" className="h-7 w-7" asChild><Link to="/datasets/$datasetId/versions/$versionId" params={{ datasetId: dataset.id, versionId: version.id }} search={{ split: defaultSplit(version) }} aria-label={`Open ${versionLabel(version)}`}><ArrowRight /></Link></Button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JobStrip({ job, onChanged }: { job: DatasetJob; onChanged: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const cancel = useMutation({ mutationFn: () => api.cancelDatasetJob(job.id), onSuccess: onChanged });
  const retry = useMutation({ mutationFn: () => api.retryDatasetJob(job.id), onSuccess: onChanged });
  const progress = jobProgress(job);
  return (
    <section className={cn("border-b", job.status === "failed" ? "border-danger/40 bg-danger-bg/45" : "border-border bg-bg-subtle/65")}>
      <div className="flex flex-wrap items-center gap-3 px-5 py-2.5">
        {isActiveJob(job.status) ? <Loader2 className="h-4 w-4 animate-spin text-accent" /> : job.status === "failed" ? <AlertTriangle className="h-4 w-4 text-danger" /> : <CheckCircle2 className="h-4 w-4 text-success" />}
        <div className="min-w-36">
          <div className="text-xs font-medium text-fg">{job.stage || job.job_type || job.kind || "Dataset job"}</div>
          <div className="font-mono text-[10px] text-fg-subtle">{job.id}</div>
        </div>
        <div className="min-w-48 flex-1">
          <div className="h-1.5 overflow-hidden rounded-full bg-surface">
            <div className={cn("h-full transition-[width] duration-500", job.status === "failed" ? "bg-danger" : "bg-accent")} style={{ width: `${progress}%` }} />
          </div>
        </div>
        <span className="w-12 text-right font-mono text-[11px] text-fg-muted">{Math.round(progress)}%</span>
        <Badge tone={jobTone(job.status)} dot size="sm">{job.status}</Badge>
        <Button variant="ghost" size="sm" onClick={() => setExpanded((value) => !value)}>{expanded ? "Hide logs" : "Logs"}</Button>
        {isActiveJob(job.status) ? <Button variant="danger" size="sm" disabled={cancel.isPending} onClick={() => cancel.mutate()}><Square />Cancel</Button> : null}
        {job.status === "failed" || job.status === "cancelled" ? <Button variant="secondary" size="sm" disabled={retry.isPending} onClick={() => retry.mutate()}><RotateCcw />Retry</Button> : null}
      </div>
      {expanded ? (
        <pre className="max-h-52 overflow-auto border-t border-border-subtle bg-bg px-5 py-3 font-mono text-[10.5px] leading-5 text-fg-muted">
          {(job.logs?.length ? job.logs : [job.error || "No log lines yet."]).join("\n")}
        </pre>
      ) : null}
    </section>
  );
}

export function PreviewTable({ preview }: { preview: DatasetPreview }) {
  const columns = useMemo(() => collectColumns(preview.items), [preview.items]);
  return (
    <div className="overflow-auto">
      <table className="w-full min-w-max text-left text-[11px]">
        <thead className="sticky top-0 z-10 bg-bg-subtle text-[10px] uppercase tracking-[0.1em] text-fg-disabled">
          <tr className="border-b border-border">
            <th className="w-14 px-3 py-2 text-right font-medium">#</th>
            {columns.map((column) => <th key={column} className="min-w-40 max-w-80 px-3 py-2 font-medium">{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {preview.items.map((row, index) => (
            <tr key={preview.offset + index} className="border-b border-border-subtle align-top hover:bg-surface/45">
              <td className="px-3 py-2 text-right font-mono text-fg-disabled">{preview.offset + index + 1}</td>
              {columns.map((column) => <td key={column} className="max-w-80 px-3 py-2"><PreviewCell name={column} value={row[column]} /></td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PreviewCell({ name, value }: { name: string; value: unknown }) {
  const mediaUrl = typeof value === "string" ? value : null;
  if (mediaUrl && isImageValue(name, mediaUrl)) {
    return <a href={mediaUrl} target="_blank" rel="noreferrer" className="block w-36"><img src={mediaUrl} alt={`${name} preview`} loading="lazy" className="h-20 w-36 rounded-sm border border-border object-cover" /></a>;
  }
  if (mediaUrl && isAudioValue(name, mediaUrl)) {
    return <audio controls preload="none" src={mediaUrl} className="h-8 w-64" aria-label={`${name} audio preview`} />;
  }
  if (value === null || value === undefined) return <span className="text-fg-disabled">null</span>;
  if (typeof value === "boolean") return <Badge tone={value ? "success" : "neutral"} size="sm">{String(value)}</Badge>;
  if (typeof value === "number") return <span className="font-mono text-fg">{String(value)}</span>;
  if (typeof value === "object") return <code className="line-clamp-4 break-all font-mono text-[10px] leading-4 text-fg-muted">{JSON.stringify(value)}</code>;
  return <span className="line-clamp-5 whitespace-pre-wrap break-words text-fg-muted">{String(value)}</span>;
}

function Pagination({ offset, limit, total, onChange }: { offset: number; limit: number; total: number; onChange: (offset: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[10px] text-fg-subtle">{total ? `${offset + 1}–${Math.min(offset + limit, total)} of ${total}` : "0 rows"}</span>
      <Button variant="ghost" size="icon" className="h-7 w-7" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))} aria-label="Previous records"><ArrowLeft /></Button>
      <Button variant="ghost" size="icon" className="h-7 w-7" disabled={offset + limit >= total} onClick={() => onChange(offset + limit)} aria-label="Next records"><ArrowRight /></Button>
    </div>
  );
}

function SectionHeader({ title, detail }: { title: string; detail?: string }) {
  return <div className="flex items-end justify-between gap-4 px-5 py-4"><div><h2 className="text-xs font-medium text-fg">{title}</h2>{detail ? <p className="mt-0.5 text-[11px] text-fg-subtle">{detail}</p> : null}</div></div>;
}

function DefinitionRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return <div className="flex items-start justify-between gap-4 px-5 py-2.5"><dt className="text-[11px] text-fg-subtle">{label}</dt><dd className={cn("max-w-[190px] break-all text-right text-[11px] text-fg", mono && "font-mono")}>{value}</dd></div>;
}

export function KeyValueTable({ value }: { value: Record<string, unknown> }) {
  const entries = Object.entries(value);
  if (!entries.length) return <InlineEmpty label="No values recorded." />;
  return <dl className="divide-y divide-border-subtle border-y border-border-subtle">{entries.map(([key, item]) => <div key={key} className="grid grid-cols-[180px_minmax(0,1fr)] gap-4 px-5 py-2.5"><dt className="font-mono text-[10.5px] text-fg-subtle">{key}</dt><dd className="break-words text-[11px] text-fg-muted">{formatValue(item)}</dd></div>)}</dl>;
}

function IconButton({ label, disabled, onClick, children }: { label: string; disabled?: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button type="button" disabled={disabled} onClick={onClick} aria-label={label} className="flex h-7 w-7 items-center justify-center rounded-sm text-fg-subtle hover:bg-surface hover:text-fg disabled:opacity-20 [&_svg]:h-3.5 [&_svg]:w-3.5">{children}</button>;
}

function InlineLoading({ label }: { label: string }) { return <div className="flex items-center justify-center gap-2 px-6 py-16 text-xs text-fg-muted"><Loader2 className="h-4 w-4 animate-spin text-accent" />{label}</div>; }
function InlineEmpty({ label }: { label: string }) { return <div className="flex flex-col items-center justify-center px-6 py-16 text-center"><FileSearch className="h-7 w-7 text-fg-disabled" /><p className="mt-3 max-w-md text-xs text-fg-muted">{label}</p></div>; }
function InlineError({ label, onRetry }: { label: string; onRetry: () => void }) { return <div className="flex flex-col items-center justify-center px-6 py-16 text-center"><AlertTriangle className="h-7 w-7 text-danger" /><p className="mt-3 max-w-md text-xs text-fg-muted">{label}</p><Button className="mt-3" size="sm" onClick={onRetry}><RefreshCcw />Retry</Button></div>; }

function DetailLoading() { return <><Topbar eyebrow="Dataset Lab" title="Loading dataset" /><InlineLoading label="Loading dataset workspace" /></>; }
function DetailError({ message, onRetry }: { message: string; onRetry: () => void }) { return <><Topbar eyebrow="Dataset Lab" title="Dataset unavailable" actions={<Button variant="ghost" size="sm" asChild><Link to="/datasets"><ArrowLeft />Datasets</Link></Button>} /><InlineError label={message} onRetry={onRetry} /></>; }

const RECIPE_KINDS = ["map", "normalize", "validate", "filter", "dedup", "score", "sample", "shuffle", "limit", "mix", "split", "contamination", "curriculum", "failure_mining", "synthesize"];

function makeDraftStep(kind: string): RecipeDraftStep { return { id: crypto.randomUUID(), kind, params: defaultParams(kind) }; }
function defaultParams(kind: string): Record<string, unknown> {
  const defaults: Record<string, Record<string, unknown>> = {
    map: { fields: { prompt: "prompt", response: "response" }, preserve_unmapped_metadata: true },
    normalize: { fields: [], trim: true, lowercase: false, collapse_whitespace: true },
    validate: { on_error: "reject" },
    filter: { field: "text", op: "exists", on_reject: "reject" },
    dedup: { method: "exact", field: "text", case_sensitive: false },
    score: { method: "heuristic", threshold: 0.5, score_field: "_quality_score", reject_below: true },
    sample: { count: 1000, seed: 42 },
    shuffle: { seed: 42 },
    limit: { count: 1000 },
    mix: { datasets: [], seed: 42 },
    split: { method: "random", ratios: { train: 0.8, validation: 0.1, test: 0.1 }, seed: 42 },
    contamination: { splits: ["train", "validation", "test"], method: "exact", field: "text", action: "report" },
    curriculum: { field: "difficulty", output_field: "metadata.curriculum", boundaries: [], labels: [] },
    failure_mining: { failure_field: "success", failure_values: [false, 0, "failed"], mode: "append" },
    synthesize: { prompt_field: "prompt", output_field: "response", n_per_record: 1, threshold: 0 },
  };
  return defaults[kind] || {};
}

function stepLabel(kind: string): string { return kind.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" "); }
function stepDescription(kind: string): string {
  const descriptions: Record<string, string> = {
    map: "Map source columns into the canonical dataset schema.", normalize: "Trim and normalize selected text fields.", validate: "Reject, quarantine, or stop on schema-invalid records.", filter: "Keep records matching a field predicate and record rejections.", dedup: "Remove exact, fuzzy, semantic, or perceptual duplicates.", score: "Score quality and optionally reject records below a threshold.", sample: "Take a deterministic count or fraction of records.", shuffle: "Randomize record order with a recorded seed.", limit: "Cap output to a fixed number of records.", mix: "Combine weighted sources into one version.", split: "Create deterministic train, validation, and test partitions.", contamination: "Measure or remove overlap between protected splits.", curriculum: "Assign curriculum bands from a difficulty field.", failure_mining: "Append or replace records with failures from a prior run.", synthesize: "Generate additional responses through a configured teacher endpoint.",
  };
  return descriptions[kind] || "Configure this deterministic recipe transform.";
}

function summarizeParams(params: Record<string, unknown>): string { const values = Object.entries(params).slice(0, 3).map(([key, value]) => `${key}=${typeof value === "object" ? "…" : String(value)}`); return values.join(" · ") || "no parameters"; }
function normalizeRecipe(value: DatasetVersion["recipe"] | undefined): DatasetRecipe | null { if (!value || typeof value !== "object" || !("steps" in value) || !Array.isArray(value.steps)) return null; return value as DatasetRecipe; }
function isDatasetTab(value: unknown): value is DatasetTab { return value === "overview" || value === "preview" || value === "build" || value === "versions"; }
function isActiveJob(status: string): boolean { return status === "queued" || status === "running" || status === "building" || status === "materializing"; }
function chooseVisibleJob(jobs: DatasetJob[], embedded?: DatasetJob | null): DatasetJob | null { return jobs.find((job) => isActiveJob(job.status)) || embedded || jobs[0] || null; }
function jobProgress(job: DatasetJob): number { const total = job.total_records ?? job.total; const completed = job.processed_records ?? job.completed ?? 0; const value = job.progress_percent ?? job.progress ?? (total ? (completed / total) * 100 : 0); return Math.max(0, Math.min(100, value <= 1 && value > 0 ? value * 100 : value)); }
function jobTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" { if (status === "completed") return "success"; if (status === "failed") return "danger"; if (status === "cancelled") return "warning"; if (isActiveJob(status)) return "accent"; return "neutral"; }
function versionTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" { return jobTone(status); }
function sourceSummary(dataset: DatasetRecord): string { const source = dataset.sources[0]; return source ? `${source.kind} · ${source.uri}` : "No source registered"; }
function formatInteger(value?: number | null): string { return typeof value === "number" ? new Intl.NumberFormat().format(value) : "—"; }
function formatBytes(value?: number | null): string { if (typeof value !== "number") return "—"; const units = ["B", "KB", "MB", "GB", "TB"]; let size = value; let index = 0; while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; } return `${size < 10 && index > 0 ? size.toFixed(1) : Math.round(size)} ${units[index]}`; }
function formatDate(value?: string | null): string { if (!value) return "—"; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date); }
function formatSplits(splits?: Record<string, number>): string { if (!splits || !Object.keys(splits).length) return "—"; return Object.entries(splits).map(([key, value]) => `${key} ${formatInteger(value)}`).join(" · "); }
function formatValue(value: unknown): string { if (value === null || value === undefined) return "—"; if (typeof value === "object") return JSON.stringify(value, null, 2); return String(value); }
function schemaValue(value: DatasetRecord["canonical_schema"]): Record<string, unknown> { return typeof value === "string" ? { kind: value } : value || { status: "Inferred at build time" }; }
function versionLabel(version: DatasetVersion): string { return version.label || (version.version !== undefined ? `v${version.version}` : version.id); }
function defaultSplit(version: DatasetVersion): string { const splits = Object.keys(version.split_counts || {}); return splits.includes("train") ? "train" : splits[0] || "all"; }
function collectColumns(items: Array<Record<string, unknown>>): string[] { const seen = new Set<string>(); items.slice(0, 25).forEach((row) => Object.keys(row).forEach((key) => seen.add(key))); return [...seen].slice(0, 16); }
function isImageValue(name: string, value: string): boolean { return /image|photo|thumbnail|frame/i.test(name) || /\.(png|jpe?g|gif|webp|avif)(\?|$)/i.test(value) || value.startsWith("data:image/"); }
function isAudioValue(name: string, value: string): boolean { return /audio|speech|sound|waveform/i.test(name) || /\.(mp3|wav|ogg|m4a|flac)(\?|$)/i.test(value) || value.startsWith("data:audio/"); }
