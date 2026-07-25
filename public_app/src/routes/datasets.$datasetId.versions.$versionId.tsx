import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Boxes,
  CheckCircle2,
  Copy,
  Download,
  FileSearch,
  Fingerprint,
  GitCompare,
  Loader2,
  ListChecks,
  Package,
  Play,
  RefreshCcw,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Topbar } from "@/components/shell";
import { DataSectionTabs } from "@/components/data/data-section-tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  api,
  type DatasetVersion,
  type DatasetVersionComparison,
  type RunListItem,
  type TrainerCompatibility,
  type TrainingDatasetArtifact,
  type TrainingMode,
} from "@/lib/api";
import { KeyValueTable, PreviewTable } from "@/routes/datasets.$datasetId";
import { cn } from "@/lib/utils";

type DatasetVersionTab = "overview" | "records" | "quality" | "training" | "lineage";

const DATASET_VERSION_TABS: Array<{ id: DatasetVersionTab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "records", label: "Records" },
  { id: "quality", label: "Quality" },
  { id: "training", label: "Training" },
  { id: "lineage", label: "Lineage" },
];

export const Route = createFileRoute("/datasets/$datasetId/versions/$versionId")({
  component: DatasetVersionRoute,
  validateSearch: (search: Record<string, unknown>): { split?: string; view?: DatasetVersionTab } => ({
    split: typeof search.split === "string" && search.split ? search.split : undefined,
    view: isDatasetVersionTab(search.view) ? search.view : undefined,
  }),
});

function DatasetVersionRoute() {
  const { datasetId, versionId } = Route.useParams();
  const search = Route.useSearch();
  const split = search.split ?? "train";
  const activeTab = search.view ?? "overview";
  const queryClient = useQueryClient();
  const [offset, setOffset] = useState(0);
  const [format, setFormat] = useState("jsonl");
  const [comparisonVersionId, setComparisonVersionId] = useState("");
  const [artifactMode, setArtifactMode] = useState<TrainingMode>("sft");
  const [artifactAdapter, setArtifactAdapter] = useState("");
  const [artifactModel, setArtifactModel] = useState("");
  const limit = 25;
  const version = useQuery({
    queryKey: ["dataset-versions", versionId],
    queryFn: () => api.datasetVersion(versionId),
  });
  const dataset = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: () => api.datasetDetail(datasetId),
  });
  const preview = useQuery({
    queryKey: ["dataset-versions", versionId, "preview", split, offset],
    queryFn: () => api.datasetVersionPreview(versionId, { split: split === "all" ? undefined : split, offset, limit }),
    enabled: version.data?.status === "ready" || version.data?.status === "completed",
  });
  const statistics = useQuery({
    queryKey: ["dataset-versions", versionId, "statistics"],
    queryFn: () => api.datasetVersionStatistics(versionId),
    enabled: Boolean(version.data),
    retry: false,
  });
  const versions = useQuery({
    queryKey: ["datasets", datasetId, "versions"],
    queryFn: () => api.datasetVersions(datasetId),
  });
  const artifacts = useQuery({
    queryKey: ["dataset-versions", versionId, "training-artifacts"],
    queryFn: () => api.listTrainingArtifacts(versionId),
    enabled: Boolean(version.data),
    refetchInterval: (query) =>
      query.state.data?.items.some((artifact) => ["queued", "rendering", "running"].includes(artifact.status))
        ? 2_000
        : false,
    retry: false,
  });
  const linkedRuns = useQuery({
    queryKey: ["dataset-versions", versionId, "runs"],
    queryFn: () => api.datasetVersionRuns(versionId),
    enabled: Boolean(version.data),
    retry: false,
  });
  const comparison = useQuery({
    queryKey: ["dataset-versions", versionId, "compare", comparisonVersionId],
    queryFn: () => api.compareDatasetVersions(versionId, comparisonVersionId),
    enabled: Boolean(comparisonVersionId),
    retry: false,
  });
  const exportVersion = useMutation({
    mutationFn: () => api.exportDatasetVersion(versionId, { format, split: split === "all" ? undefined : split }),
  });
  const materialize = useMutation({
    mutationFn: () => api.materializeDatasetVersion(versionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset-versions", versionId] });
      queryClient.invalidateQueries({ queryKey: ["dataset-jobs", datasetId] });
    },
  });
  const renderArtifact = useMutation({
    mutationFn: () =>
      api.renderTrainingArtifact(versionId, {
        adapter_id:
          artifactAdapter ||
          compatibilityForVersion(version.data).find(
            (entry) => entry.compatible && entry.trainer_mode === artifactMode,
          )?.adapter_id ||
          `${artifactMode}.canonical`,
        trainer_mode: artifactMode,
        model: artifactModel || undefined,
        bindings: [{
          role: "train",
          dataset_version_id: versionId,
          split:
            activeSplit === "all"
              ? Object.keys(version.data?.split_counts ?? {})[0] || "train"
              : activeSplit,
        }],
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset-versions", versionId, "training-artifacts"] });
    },
  });

  useEffect(() => {
    const compatible = compatibilityForVersion(version.data).filter((entry) => entry.compatible);
    if (!compatible.length) return;
    const selected = compatible.find((entry) => entry.trainer_mode === artifactMode) ?? compatible[0];
    if (selected.trainer_mode !== artifactMode) setArtifactMode(selected.trainer_mode as TrainingMode);
    if (!artifactAdapter) setArtifactAdapter(selected.adapter_id);
  }, [artifactAdapter, artifactMode, version.data]);

  if (version.isLoading) return <><Topbar eyebrow="Dataset Lab" title="Loading version" /><Loading label="Loading version evidence" /></>;
  if (version.isError || !version.data) return <VersionError message={(version.error as Error)?.message || "Version was not found."} datasetId={datasetId} onRetry={() => version.refetch()} />;

  const item = version.data;
  const splits = Object.keys(item.split_counts || {});
  const activeSplit = split === "all" || splits.includes(split) ? split : splits[0] || "all";
  const trainingSplit = activeSplit === "all" ? splits[0] || "train" : activeSplit;
  const exportPath = exportVersion.data
    ? typeof exportVersion.data.path === "string"
      ? exportVersion.data.path
      : typeof exportVersion.data.output === "string"
        ? exportVersion.data.output
        : null
    : null;
  const rejectionEvidence = item.rejections ?? evidenceFromStatistics(item.statistics, "rejections", "rejection_reasons");
  const contaminationEvidence = item.contamination ?? evidenceFromStatistics(item.statistics, "contamination", "contamination_report");
  const compatibility = compatibilityForVersion(item);
  const compatible = compatibility.filter((entry) => entry.compatible);
  const preferredTrainer = compatible[0];
  const tokenStats = exactTokenStats(artifacts.data?.items ?? [], statistics.data ?? item.statistics);

  return (
    <>
      <Topbar
        eyebrow={dataset.data?.name || "Dataset version"}
        title={versionLabel(item)}
        subtitle={`Immutable build ${item.content_hash ? `· ${item.content_hash.slice(0, 12)}` : ""}`}
        actions={
          <>
            <Button variant="ghost" size="icon" asChild aria-label="Back to dataset">
              <Link to="/datasets/$datasetId" params={{ datasetId }} search={{ tab: "versions", recipeFrom: undefined }}><ArrowLeft /></Link>
            </Button>
            <Button variant="secondary" size="sm" asChild>
              <Link to="/datasets/$datasetId" params={{ datasetId }} search={{ tab: "build", recipeFrom: versionId }}><Copy />Clone recipe</Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/datasets/review" search={{ new: "1", source: "dataset_version", sourceRef: versionId, baseRef: undefined }}><ListChecks />Review</Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/datasets/ground" search={{ sourceVersion: versionId }}><FileSearch />Create examples from documents</Link>
            </Button>
            {preferredTrainer ? (
              <>
                <Button variant="secondary" size="sm" asChild>
                  <Link
                    to="/sweeps"
                    search={{
                      group: undefined,
                      new: "1",
                      policy: undefined,
                      datasetVersion: versionId,
                      trainerMode: preferredTrainer.trainer_mode as TrainingMode,
                      kind: "repeat",
                    }}
                  >
                    <RefreshCcw />Start repeat
                  </Link>
                </Button>
                <Button variant="primary" size="sm" asChild>
                  <Link
                    to="/train"
                    search={{
                      template: undefined,
                      model: undefined,
                      mode: preferredTrainer.trainer_mode,
                      datasetVersion: versionId,
                      datasetSplit: trainingSplit,
                      parentRun: undefined,
                    }}
                  >
                    <Play />Train single
                  </Link>
                </Button>
              </>
            ) : (
              <Button variant="primary" size="sm" disabled title="No registered trainer adapter accepts this version">
                <Play />No compatible trainer
              </Button>
            )}
          </>
        }
        statusBar={
          <>
            <Badge tone={versionTone(item.status)} dot size="sm">{item.status}</Badge>
            <span>{formatInteger(item.row_count)} rows</span>
            <span className="text-fg-disabled">•</span>
            <span>{formatBytes(item.size_bytes)}</span>
            <span className="text-fg-disabled">•</span>
            <span>{item.assets_materialized ? "assets materialized" : "assets referenced"}</span>
          </>
        }
      />

      <DataSectionTabs />
      <DatasetVersionTabs datasetId={datasetId} versionId={versionId} split={activeSplit} active={activeTab} />

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px]">
        <main className="min-w-0 border-b border-border-subtle lg:border-b-0 lg:border-r">
          {activeTab === "overview" ? (
            <section className="border-b border-border-subtle">
              <SectionTitle title="Immutable dataset build" detail="A concise view of the exact data identity available to downstream work." />
              <div className="grid gap-px border-y border-border-subtle bg-border-subtle sm:grid-cols-3">
                <OverviewReadout label="Records" value={formatInteger(item.row_count)} />
                <OverviewReadout label="Splits" value={String(splits.length || 1)} />
                <OverviewReadout label="Storage" value={formatBytes(item.size_bytes)} />
              </div>
              <div className="px-5 py-4"><KeyValueTable value={{ split_counts: item.split_counts ?? {}, status: item.status, assets: item.assets_materialized ? "materialized" : "referenced" }} /></div>
            </section>
          ) : null}

          {activeTab === "records" ? <section className="border-b border-border-subtle">
            <SectionTitle title="Splits" detail="Inspect the exact records that will be handed to training." />
            <div className="flex flex-wrap items-center gap-1 border-t border-border-subtle bg-bg-subtle/55 px-5 py-2">
              {(splits.length ? splits : ["train"]).map((name) => (
                <Link
                  key={name}
                  to="/datasets/$datasetId/versions/$versionId"
                  params={{ datasetId, versionId }}
                  search={{ split: name, view: "records" }}
                  onClick={() => setOffset(0)}
                  className={cn(
                    "rounded-sm px-2.5 py-1 text-xs transition-colors",
                    activeSplit === name ? "bg-accent-bg text-accent" : "text-fg-muted hover:bg-surface hover:text-fg",
                  )}
                >
                  {name}
                  {item.split_counts?.[name] !== undefined ? <span className="ml-2 font-mono text-[10px] opacity-70">{formatInteger(item.split_counts[name])}</span> : null}
                </Link>
              ))}
              <div className="ml-auto flex items-center gap-2">
                <span className="font-mono text-[10px] text-fg-subtle">{preview.data?.total ? `${offset + 1}–${Math.min(offset + limit, preview.data.total)} of ${preview.data.total}` : "0 rows"}</span>
                <Button variant="ghost" size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>Previous</Button>
                <Button variant="ghost" size="sm" disabled={!preview.data || offset + limit >= preview.data.total} onClick={() => setOffset(offset + limit)}>Next</Button>
              </div>
            </div>
            {preview.isLoading ? <Loading label="Loading version records" /> : preview.isError ? <InlineError label={(preview.error as Error).message} onRetry={() => preview.refetch()} /> : preview.data?.items.length ? <PreviewTable preview={preview.data} /> : <Empty label={item.status === "ready" ? "This split has no records." : "Records are available after the build completes."} />}
          </section> : null}

          {activeTab === "quality" ? <section className="border-b border-border-subtle">
            <SectionTitle title="Statistics" detail="Stored profile for this immutable version." />
            {statistics.isLoading ? <Loading label="Loading statistics" /> : <KeyValueTable value={statistics.data || item.statistics || {}} />}
          </section> : null}

          {activeTab === "training" ? <section className="border-b border-border-subtle">
            <SectionTitle title="Trainer compatibility" detail="Reported by the versioned adapter registry, not inferred from the schema name." />
            <CompatibilityTable items={compatibility} />
          </section> : null}

          {activeTab === "training" ? <section className="border-b border-border-subtle">
            <SectionTitle
              title={tokenStats?.exact === false ? "Estimated token profile" : "Exact token profile"}
              detail={tokenStats?.exact === false ? "Tokenizer unavailable; these counts use the recorded fallback estimate." : "Model and chat-template-aware counts from rendered training artifacts."}
            />
            {tokenStats ? <KeyValueTable value={tokenStats} /> : <Empty label="Render a training artifact to compute a token profile; provide a local model/tokenizer for exact counts." compact />}
          </section> : null}

          {activeTab === "training" ? <section className="border-b border-border-subtle">
            <details>
              <summary className="cursor-pointer list-none px-5 py-4 hover:bg-surface/35">
                <div className="flex items-start justify-between gap-3">
                  <div><div className="text-[12px] font-medium text-fg">Advanced · Training artifacts workbench</div><div className="mt-1 text-[11px] leading-5 text-fg-muted">Normal Train launches prepare the compatible artifact automatically. Open this only to inspect or render a specific adapter manually.</div></div>
                  <Badge tone="neutral" size="sm">{artifacts.data?.items.length ?? 0} artifacts</Badge>
                </div>
              </summary>
              <div className="border-t border-border-subtle">
                <ArtifactWorkbench
                  versionId={versionId}
                  activeSplit={trainingSplit}
                  compatibility={compatibility}
                  artifacts={artifacts.data?.items ?? []}
                  loading={artifacts.isLoading}
                  mode={artifactMode}
                  adapter={artifactAdapter || preferredTrainer?.adapter_id || ""}
                  model={artifactModel}
                  onMode={setArtifactMode}
                  onAdapter={setArtifactAdapter}
                  onModel={setArtifactModel}
                  onRender={() => renderArtifact.mutate()}
                  rendering={renderArtifact.isPending}
                  error={renderArtifact.isError ? (renderArtifact.error as Error).message : null}
                />
              </div>
            </details>
          </section> : null}

          {activeTab === "quality" ? <section className="border-b border-border-subtle">
            <SectionTitle title="Compare versions" detail="Identity-aware record, split, recipe, profile, and source contribution changes." />
            <VersionComparison
              versions={(versions.data?.items ?? []).filter((candidate) => candidate.id !== versionId)}
              selected={comparisonVersionId}
              onSelect={setComparisonVersionId}
              data={comparison.data}
              loading={comparison.isLoading}
              error={comparison.isError ? (comparison.error as Error).message : null}
            />
          </section> : null}

          {activeTab === "quality" ? <section className="border-b border-border-subtle">
            <SectionTitle title="Rejections" detail="Records excluded or quarantined by recipe steps." />
            <EvidenceValue value={rejectionEvidence} empty="No rejection summary was recorded." />
          </section> : null}

          {activeTab === "quality" ? <section>
            <SectionTitle title="Contamination" detail="Overlap findings across train, validation, test, and protected sets." />
            <EvidenceValue value={contaminationEvidence} empty="No contamination check was recorded for this version." />
          </section> : null}

          {activeTab === "lineage" ? <section>
            <SectionTitle title="Ordered provenance" detail="Every source and transform that produced this immutable content hash." />
            <Provenance value={item.provenance} />
          </section> : null}
        </main>

        <aside className="bg-bg-subtle/35">
          <SectionTitle title="Version evidence" detail="Hashes, recipe lineage, and portable output actions." />
          <dl className="divide-y divide-border-subtle border-y border-border-subtle">
            <Definition label="Version ID" value={item.id} mono />
            <Definition label="Content hash" value={item.content_hash || "—"} mono />
            <Definition label="Recipe hash" value={item.recipe_hash || "—"} mono />
            <Definition label="Storage" value={item.storage_path || "—"} mono />
            <Definition label="Created" value={formatDate(item.created_at)} />
          </dl>

          {activeTab === "overview" || activeTab === "records" ? <div className="border-b border-border-subtle px-5 py-4">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium text-fg"><Download className="h-3.5 w-3.5 text-fg-subtle" />Export split</div>
            <div className="flex gap-2">
              <select value={format} onChange={(event) => setFormat(event.target.value)} className="h-8 flex-1 rounded-md border border-border bg-bg px-2 text-xs text-fg">
                <option value="jsonl">JSONL</option>
                <option value="parquet">Parquet</option>
                <option value="csv">CSV</option>
              </select>
              <Button variant="secondary" size="sm" disabled={exportVersion.isPending} onClick={() => exportVersion.mutate()}>
                {exportVersion.isPending ? <Loader2 className="animate-spin" /> : <Download />}Export
              </Button>
            </div>
            {exportPath ? <div className="mt-2 break-all font-mono text-[10px] text-success">{exportPath}</div> : null}
            {exportVersion.isError ? <div className="mt-2 text-[10px] text-danger">{(exportVersion.error as Error).message}</div> : null}
          </div> : null}

          {activeTab === "training" || activeTab === "lineage" ? <div className="border-b border-border-subtle">
            <SectionTitle title="Runs using this version" detail="Training lineage bound to this exact content hash." />
            <LinkedRuns items={linkedRuns.data?.items ?? []} loading={linkedRuns.isLoading} />
          </div> : null}

          {activeTab === "overview" || activeTab === "records" ? <div className="border-b border-border-subtle px-5 py-4">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium text-fg"><Boxes className="h-3.5 w-3.5 text-fg-subtle" />Media assets</div>
            <p className="mb-3 text-[11px] leading-4 text-fg-subtle">Copy referenced image or audio assets into the version store for portable replay.</p>
            <Button variant="secondary" size="sm" className="w-full" disabled={materialize.isPending || item.assets_materialized} onClick={() => materialize.mutate()}>
              {materialize.isPending ? <Loader2 className="animate-spin" /> : item.assets_materialized ? <CheckCircle2 /> : <Boxes />}
              {item.assets_materialized ? "Materialized" : "Materialize assets"}
            </Button>
            {materialize.isSuccess ? <div className="mt-2 font-mono text-[10px] text-success">Job {materialize.data.job_id || materialize.data.id} queued</div> : null}
          </div> : null}

          {activeTab === "lineage" ? <div>
            <SectionTitle title="Provenance" detail="Ordered transforms with input, output, and rejection counts." />
            <Provenance value={item.provenance} />
          </div> : null}
        </aside>
      </div>
    </>
  );
}

function DatasetVersionTabs({ datasetId, versionId, split, active }: { datasetId: string; versionId: string; split: string; active: DatasetVersionTab }) {
  return (
    <nav aria-label="Dataset version detail" className="sticky top-[49px] z-10 flex overflow-x-auto border-b border-border bg-bg-subtle/95 px-3 backdrop-blur md:px-5">
      {DATASET_VERSION_TABS.map((tab) => (
        <Link key={tab.id} to="/datasets/$datasetId/versions/$versionId" params={{ datasetId, versionId }} search={{ split, view: tab.id }} aria-current={active === tab.id ? "page" : undefined} className={cn("relative flex h-11 shrink-0 items-center px-3 text-[12px] transition-colors", active === tab.id ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>
          {tab.label}
          {active === tab.id ? <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" /> : null}
        </Link>
      ))}
    </nav>
  );
}

function OverviewReadout({ label, value }: { label: string; value: string }) {
  return <div className="bg-bg px-5 py-4"><div className="font-mono text-xl text-fg">{value}</div><div className="mt-1 text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">{label}</div></div>;
}

function isDatasetVersionTab(value: unknown): value is DatasetVersionTab {
  return DATASET_VERSION_TABS.some((tab) => tab.id === value);
}

function CompatibilityTable({ items }: { items: TrainerCompatibility[] }) {
  if (!items.length) {
    return (
      <Empty
        label="No adapter-registry compatibility evidence was returned. Training stays disabled until an adapter validates this version."
        compact
      />
    );
  }
  return (
    <div className="overflow-x-auto border-y border-border-subtle">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-border-subtle bg-bg-subtle/45 text-left text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">
            <th className="px-5 py-2 font-medium">Trainer</th>
            <th className="px-3 py-2 font-medium">Adapter</th>
            <th className="px-3 py-2 font-medium">Version</th>
            <th className="px-3 py-2 font-medium">Result</th>
            <th className="px-3 py-2 font-medium">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={`${item.adapter_id}-${item.trainer_mode}`} className="border-b border-border-subtle last:border-0">
              <td className="px-5 py-2.5 font-medium text-fg">{item.trainer_mode}</td>
              <td className="px-3 py-2.5 font-mono text-fg-muted">{item.adapter_id}</td>
              <td className="px-3 py-2.5 font-mono text-fg-subtle">{item.adapter_version || "—"}</td>
              <td className="px-3 py-2.5">
                <Badge tone={item.compatible ? "success" : "danger"} dot size="sm">
                  {item.compatible ? "compatible" : "blocked"}
                </Badge>
              </td>
              <td className="max-w-[40ch] px-3 py-2.5 text-fg-subtle">{item.reason || "Adapter validation passed."}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactWorkbench({
  versionId,
  activeSplit,
  compatibility,
  artifacts,
  loading,
  mode,
  adapter,
  model,
  onMode,
  onAdapter,
  onModel,
  onRender,
  rendering,
  error,
}: {
  versionId: string;
  activeSplit: string;
  compatibility: TrainerCompatibility[];
  artifacts: TrainingDatasetArtifact[];
  loading: boolean;
  mode: TrainingMode;
  adapter: string;
  model: string;
  onMode: (mode: TrainingMode) => void;
  onAdapter: (adapter: string) => void;
  onModel: (model: string) => void;
  onRender: () => void;
  rendering: boolean;
  error: string | null;
}) {
  const compatible = compatibility.filter((item) => item.compatible);
  return (
    <div className="border-y border-border-subtle">
      <div className="grid gap-3 bg-bg-subtle/35 px-5 py-4 md:grid-cols-[150px_minmax(170px,1fr)_minmax(200px,1fr)_auto] md:items-end">
        <label className="space-y-1">
          <span className="block text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">Trainer mode</span>
          <select
            value={mode}
            onChange={(event) => {
              const next = event.target.value as TrainingMode;
              onMode(next);
              const match = compatible.find((item) => item.trainer_mode === next);
              onAdapter(match?.adapter_id || "");
            }}
            className="h-9 w-full rounded-md border border-border bg-bg px-2 text-xs text-fg"
          >
            {(["sft", "raft", "dpo", "orpo", "rm", "grpo", "vlm", "audio", "reasoning", "agentic"] as TrainingMode[]).map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">Adapter ID</span>
          <input
            value={adapter}
            onChange={(event) => onAdapter(event.target.value)}
            placeholder={`${mode}.canonical`}
            className="h-9 w-full rounded-md border border-border bg-bg px-2 font-mono text-[11px] text-fg"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">Model / tokenizer</span>
          <input
            value={model}
            onChange={(event) => onModel(event.target.value)}
            placeholder="Optional model revision for exact token counts"
            className="h-9 w-full rounded-md border border-border bg-bg px-2 font-mono text-[11px] text-fg"
          />
        </label>
        <Button
          variant="primary"
          size="sm"
          disabled={rendering || !compatible.some((item) => item.trainer_mode === mode)}
          onClick={onRender}
          title={compatible.some((item) => item.trainer_mode === mode) ? "Render immutable trainer inputs" : "This mode has not passed adapter validation"}
        >
          {rendering ? <Loader2 className="animate-spin" /> : <Package />}
          Render
        </Button>
      </div>
      <div className="border-t border-border-subtle px-5 py-2 font-mono text-[10px] text-fg-subtle">
        train = {versionId}:{activeSplit} · held-out splits are never added implicitly
      </div>
      {error ? <div className="border-t border-border-subtle px-5 py-2 text-[11px] text-danger">{error}</div> : null}
      {loading ? (
        <Loading label="Loading training artifacts" />
      ) : artifacts.length ? (
        <div className="overflow-x-auto border-t border-border-subtle">
          <table className="w-full text-[11px]">
            <thead><tr className="border-b border-border-subtle text-left text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled"><th className="px-5 py-2 font-medium">Artifact</th><th className="px-3 py-2 font-medium">Adapter</th><th className="px-3 py-2 font-medium">Rows</th><th className="px-3 py-2 font-medium">Tokens</th><th className="px-3 py-2 font-medium">Status</th></tr></thead>
            <tbody>
              {artifacts.map((artifact) => (
                <tr key={artifact.id} className="border-b border-border-subtle last:border-0">
                  <td className="px-5 py-2.5"><div className="font-mono text-fg">{artifact.id}</div><div className="mt-0.5 font-mono text-[9.5px] text-fg-disabled">{artifact.artifact_hash?.slice(0, 16) || "hash pending"}</div></td>
                  <td className="px-3 py-2.5"><div className="font-medium text-fg">{artifact.trainer_mode}</div><div className="font-mono text-[9.5px] text-fg-subtle">{artifact.adapter_id}@{artifact.adapter_version}</div></td>
                  <td className="px-3 py-2.5 font-mono text-fg-muted">{formatInteger(sumValues(artifact.row_counts))}</td>
                  <td className="px-3 py-2.5 font-mono text-fg-muted">{formatInteger(tokenTotal(artifact.token_statistics))}</td>
                  <td className="px-3 py-2.5"><Badge tone={versionTone(artifact.status)} dot size="sm">{artifact.status}</Badge>{artifact.derived_validation ? <div className="mt-1 text-[9.5px] text-warning">derived validation</div> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty label="No trainer artifact has been rendered for this version." compact />
      )}
    </div>
  );
}

function VersionComparison({
  versions,
  selected,
  onSelect,
  data,
  loading,
  error,
}: {
  versions: DatasetVersion[];
  selected: string;
  onSelect: (id: string) => void;
  data?: DatasetVersionComparison;
  loading: boolean;
  error: string | null;
}) {
  return (
    <div className="border-y border-border-subtle">
      <div className="flex flex-wrap items-center gap-3 bg-bg-subtle/35 px-5 py-3">
        <GitCompare className="h-4 w-4 text-fg-subtle" />
        <span className="text-[11px] text-fg-muted">Compare this build with</span>
        <select value={selected} onChange={(event) => onSelect(event.target.value)} className="h-8 min-w-56 rounded-md border border-border bg-bg px-2 font-mono text-[11px] text-fg">
          <option value="">Choose a version</option>
          {versions.map((version) => <option key={version.id} value={version.id}>{versionLabel(version)} · {version.id}</option>)}
        </select>
      </div>
      {loading ? <Loading label="Comparing record identities" /> : error ? <div className="px-5 py-5 text-[11px] text-danger">{error}</div> : data ? (
        <>
          <div className="grid grid-cols-2 gap-px border-t border-border-subtle bg-border-subtle sm:grid-cols-3 lg:grid-cols-6">
            {comparisonCounts(data).map(([label, value]) => (
              <div key={label} className="bg-bg px-4 py-3"><div className="font-mono text-lg text-fg">{formatInteger(value)}</div><div className="text-[9.5px] uppercase tracking-[0.1em] text-fg-disabled">{label}</div></div>
            ))}
          </div>
          <div className="grid border-t border-border-subtle lg:grid-cols-3 lg:divide-x lg:divide-border-subtle">
            <ComparisonEvidence title="Recipe" value={data.recipe_diff} />
            <ComparisonEvidence title="Statistics" value={data.statistics_diff} />
            <ComparisonEvidence title="Source contribution" value={data.source_contribution_diff} />
          </div>
          <VersionRecordChanges data={data} />
        </>
      ) : <Empty label="Choose another version to inspect identity-preserving changes." compact />}
    </div>
  );
}

function VersionRecordChanges({ data }: { data: DatasetVersionComparison }) {
  const groups: Array<[string, Array<Record<string, unknown>> | undefined]> = [
    ["added", data.added],
    ["removed", data.removed],
    ["content changed", data.changed],
    ["repeated", data.repeated],
    ["split moved", data.split_moved],
  ];
  const populated = groups.filter(([, items]) => items?.length);
  if (!populated.length) return null;
  return <div className="border-t border-border-subtle"><div className="px-5 py-3 text-[10px] uppercase tracking-[0.12em] text-fg-disabled">Record evidence</div><div className="grid lg:grid-cols-2">{populated.map(([label, items]) => <div key={label} className="min-w-0 border-t border-border-subtle px-5 py-3 odd:lg:border-r"><div className="mb-2 flex items-center justify-between"><span className="text-[11px] font-medium text-fg">{label}</span><Badge tone={label === "removed" || label === "content changed" ? "warning" : "neutral"} size="sm">{items?.length ?? 0}</Badge></div><div className="space-y-1">{items?.slice(0, 5).map((item, index) => <div key={`${label}-${index}`} className="truncate font-mono text-[9.5px] text-fg-muted" title={JSON.stringify(item)}>{String(item.record_id ?? item.instance_id ?? item.content_hash ?? JSON.stringify(item))}</div>)}</div></div>)}</div></div>;
}

function ComparisonEvidence({ title, value }: { title: string; value?: Record<string, unknown> }) {
  return <div className="min-w-0 px-5 py-4"><div className="mb-2 text-[10px] uppercase tracking-[0.12em] text-fg-disabled">{title}</div>{value && Object.keys(value).length ? <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-4 text-fg-muted">{JSON.stringify(value, null, 2)}</pre> : <span className="text-[11px] text-fg-disabled">No change recorded.</span>}</div>;
}

function LinkedRuns({ items, loading }: { items: RunListItem[]; loading: boolean }) {
  if (loading) return <Loading label="Loading linked runs" />;
  if (!items.length) return <Empty label="No run is attached to this dataset version yet." compact />;
  return <div className="divide-y divide-border-subtle border-t border-border-subtle">{items.slice(0, 8).map((run) => <Link key={run.run_id} to="/runs/$runId" params={{ runId: run.run_id }} className="flex items-center justify-between gap-3 px-5 py-2.5 hover:bg-surface"><span className="min-w-0 truncate font-mono text-[10px] text-accent">{run.run_id}</span><span className="shrink-0 text-[10px] text-fg-subtle">{run.modality} · {run.status || "unknown"}</span></Link>)}</div>;
}

function Provenance({ value }: { value?: Record<string, unknown> }) {
  const rawSteps = value?.steps ?? value?.step_summaries ?? value;
  const steps = Array.isArray(rawSteps) ? rawSteps.filter((step): step is Record<string, unknown> => Boolean(step) && typeof step === "object") : [];
  if (!steps.length) return <Empty label="No ordered provenance steps were recorded." compact />;
  return (
    <ol className="divide-y divide-border-subtle border-y border-border-subtle">
      {steps.map((step, index) => (
        <li key={`${String(step.kind)}-${index}`} className="px-5 py-3">
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-sm border border-border font-mono text-[9px] text-fg-subtle">{index + 1}</span>
            <span className="text-xs font-medium text-fg">{String(step.kind || "step")}</span>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[9.5px] text-fg-subtle">
            <span>in {formatInteger(asNumber(step.input_count))}</span>
            <span>out {formatInteger(asNumber(step.output_count))}</span>
            <span>reject {formatInteger(asNumber(step.rejected_count))}</span>
          </div>
        </li>
      ))}
    </ol>
  );
}

function EvidenceValue({ value, empty }: { value: DatasetVersion["rejections"] | DatasetVersion["contamination"]; empty: string }) {
  if (!value || (Array.isArray(value) && !value.length) || (!Array.isArray(value) && !Object.keys(value).length)) return <Empty label={empty} compact />;
  if (Array.isArray(value)) return <div className="overflow-x-auto"><table className="w-full text-[11px]"><tbody>{value.map((row, index) => <tr key={index} className="border-t border-border-subtle"><td className="w-12 px-5 py-2 font-mono text-fg-disabled">{index + 1}</td><td className="px-3 py-2 font-mono text-fg-muted">{JSON.stringify(row)}</td></tr>)}</tbody></table></div>;
  return <KeyValueTable value={value} />;
}

function SectionTitle({ title, detail }: { title: string; detail?: string }) { return <div className="px-5 py-4"><h2 className="text-xs font-medium text-fg">{title}</h2>{detail ? <p className="mt-0.5 text-[11px] text-fg-subtle">{detail}</p> : null}</div>; }
function Definition({ label, value, mono }: { label: string; value: string; mono?: boolean }) { return <div className="px-5 py-2.5"><dt className="text-[10px] uppercase tracking-[0.1em] text-fg-disabled">{label}</dt><dd className={cn("mt-1 break-all text-[11px] text-fg-muted", mono && "font-mono")}>{value}</dd></div>; }
function Loading({ label }: { label: string }) { return <div className="flex items-center justify-center gap-2 px-6 py-16 text-xs text-fg-muted"><Loader2 className="h-4 w-4 animate-spin text-accent" />{label}</div>; }
function Empty({ label, compact }: { label: string; compact?: boolean }) { return <div className={cn("flex flex-col items-center justify-center px-6 text-center", compact ? "py-8" : "py-16")}><FileSearch className="h-6 w-6 text-fg-disabled" /><p className="mt-2 max-w-md text-[11px] text-fg-muted">{label}</p></div>; }
function InlineError({ label, onRetry }: { label: string; onRetry: () => void }) { return <div className="flex flex-col items-center px-6 py-12 text-center"><ShieldCheck className="h-6 w-6 text-danger" /><p className="mt-2 text-xs text-fg-muted">{label}</p><Button className="mt-3" size="sm" onClick={onRetry}><RefreshCcw />Retry</Button></div>; }
function VersionError({ message, datasetId, onRetry }: { message: string; datasetId: string; onRetry: () => void }) { return <><Topbar eyebrow="Dataset Lab" title="Version unavailable" actions={<Button variant="ghost" size="sm" asChild><Link to="/datasets/$datasetId" params={{ datasetId }} search={{ tab: "versions", recipeFrom: undefined }}><ArrowLeft />Dataset</Link></Button>} /><div className="flex flex-col items-center px-6 py-20 text-center"><Fingerprint className="h-8 w-8 text-danger" /><p className="mt-3 text-xs text-fg-muted">{message}</p><Button className="mt-3" size="sm" onClick={onRetry}><RefreshCcw />Retry</Button></div></>; }

function versionLabel(version: DatasetVersion): string { return version.label || (version.version !== undefined ? `v${version.version}` : version.id); }
function versionTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" { if (status === "ready" || status === "completed") return "success"; if (status === "failed") return "danger"; if (status === "cancelled") return "warning"; if (["queued", "running", "building"].includes(status)) return "accent"; return "neutral"; }
function formatInteger(value?: number | null): string { return typeof value === "number" ? new Intl.NumberFormat().format(value) : "—"; }
function formatBytes(value?: number | null): string { if (typeof value !== "number") return "—"; const units = ["B", "KB", "MB", "GB", "TB"]; let size = value; let index = 0; while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; } return `${size < 10 && index > 0 ? size.toFixed(1) : Math.round(size)} ${units[index]}`; }
function formatDate(value?: string | null): string { if (!value) return "—"; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date); }
function asNumber(value: unknown): number | null { return typeof value === "number" ? value : null; }
function evidenceFromStatistics(statistics: Record<string, unknown> | undefined, ...keys: string[]): Record<string, unknown> | Array<Record<string, unknown>> | undefined { for (const key of keys) { const value = statistics?.[key]; if (Array.isArray(value)) return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object"); if (value && typeof value === "object") return value as Record<string, unknown>; } return undefined; }

function compatibilityForVersion(version?: DatasetVersion): TrainerCompatibility[] {
  if (!version) return [];
  const raw = version.compatible_trainers ?? version.trainer_compatibility ?? version.compatibility;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((entry): TrainerCompatibility[] => {
    if (typeof entry === "string") {
      return [{ adapter_id: entry, trainer_mode: entry.split(".")[0], compatible: true }];
    }
    if (!entry || typeof entry !== "object") return [];
    const item = entry as Record<string, unknown>;
    const adapterId = String(item.adapter_id ?? item.adapter ?? "");
    const trainerMode = String(item.trainer_mode ?? item.mode ?? "");
    if (!adapterId || !trainerMode) return [];
    return [{
      adapter_id: adapterId,
      adapter_version: typeof item.adapter_version === "string" ? item.adapter_version : undefined,
      trainer_mode: trainerMode,
      compatible: item.compatible !== false,
      reason: typeof item.reason === "string" ? item.reason : null,
      required_schema: typeof item.required_schema === "string" ? item.required_schema : null,
    }];
  });
}

function exactTokenStats(
  artifacts: TrainingDatasetArtifact[],
  fallback?: Record<string, unknown>,
): Record<string, unknown> | null {
  const ready = artifacts.find((artifact) => artifact.status === "ready" && artifact.token_statistics);
  if (ready?.token_statistics) {
    return {
      artifact_id: ready.id,
      model: ready.model ?? "default tokenizer",
      tokenizer_revision: ready.tokenizer_revision ?? "—",
      chat_template_hash: ready.chat_template_hash ?? "—",
      ...ready.token_statistics,
    };
  }
  const candidate = fallback?.exact_token_statistics ?? fallback?.token_statistics_exact;
  return candidate && typeof candidate === "object" ? candidate as Record<string, unknown> : null;
}

function sumValues(values?: Record<string, number>): number | null {
  if (!values) return null;
  return Object.values(values).reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
}

function tokenTotal(values?: Record<string, unknown>): number | null {
  if (!values) return null;
  const direct = values.total_tokens ?? values.tokens;
  if (typeof direct === "number") return direct;
  const overall = values.overall;
  if (overall && typeof overall === "object" && typeof (overall as Record<string, unknown>).total === "number") {
    return (overall as Record<string, number>).total;
  }
  const splits = values.splits;
  if (splits && typeof splits === "object") {
    const totals = Object.values(splits as Record<string, unknown>).flatMap((entry) => {
      if (!entry || typeof entry !== "object") return [];
      const total = (entry as Record<string, unknown>).total;
      return typeof total === "number" ? [total] : [];
    });
    if (totals.length) return totals.reduce((sum, value) => sum + value, 0);
  }
  const splitTotals = Object.entries(values)
    .filter(([key, value]) => key.endsWith("_tokens") && typeof value === "number")
    .map(([, value]) => value as number);
  return splitTotals.length ? splitTotals.reduce((sum, value) => sum + value, 0) : null;
}

function comparisonCounts(data: DatasetVersionComparison): Array<[string, number | null]> {
  const fromSummary = (key: string): number | null => {
    const value = data.summary?.[key];
    return typeof value === "number" ? value : null;
  };
  return [
    ["added", data.added?.length ?? fromSummary("added")],
    ["removed", data.removed?.length ?? fromSummary("removed")],
    ["changed", data.changed?.length ?? fromSummary("changed")],
    ["repeated", data.repeated?.length ?? fromSummary("repeated")],
    ["split moved", data.split_moved?.length ?? fromSummary("split_moved")],
    ["same", fromSummary("unchanged")],
  ];
}
