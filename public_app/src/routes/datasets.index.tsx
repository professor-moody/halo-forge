import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  Cloud,
  Database,
  FolderOpen,
  Loader2,
  Plus,
  RefreshCcw,
  Search,
  X,
} from "lucide-react";
import { useMemo, useState, type FormEvent } from "react";
import { Topbar } from "@/components/shell";
import { DataSectionTabs } from "@/components/data/data-section-tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, type DatasetCreatePayload, type DatasetRecord } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/datasets/")({
  component: DatasetIndexRoute,
});

function DatasetIndexRoute() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const datasets = useQuery({
    queryKey: ["datasets"],
    queryFn: api.listDatasets,
    refetchInterval: 20_000,
    refetchIntervalInBackground: false,
  });
  const create = useMutation({
    mutationFn: api.createDataset,
    onSuccess: (dataset) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setCreateOpen(false);
      navigate({
        to: "/datasets/$datasetId",
        params: { datasetId: dataset.id },
        search: { tab: "overview", recipeFrom: undefined },
      });
    },
  });

  const items = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return datasets.data?.items ?? [];
    return (datasets.data?.items ?? []).filter((dataset) =>
      [dataset.name, dataset.description, dataset.modality, ...dataset.sources.map((source) => source.uri)]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle)),
    );
  }, [datasets.data?.items, query]);

  return (
    <>
      <Topbar
        eyebrow="Data"
        title="Dataset Lab"
        subtitle="Register sources, build reproducible versions, and inspect the evidence before training."
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setCreateOpen(true)}>
              <Plus />
              Advanced register
            </Button>
            <Button variant="primary" size="sm" asChild>
              <Link to="/datasets/new" search={{ example: undefined }}>
                <ArrowRight />
                Train on your data
              </Link>
            </Button>
          </>
        }
      />
      <DataSectionTabs />

      <div className="min-h-full">
        <div className="flex flex-wrap items-center gap-3 border-b border-border-subtle px-5 py-3">
          <div className="relative min-w-64 max-w-md flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-disabled" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search name, source, or modality"
              className="h-8 pl-8 text-xs"
              aria-label="Search datasets"
            />
          </div>
          <span className="font-mono text-[11px] text-fg-subtle">
            {items.length} of {datasets.data?.items.length ?? 0} datasets
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => datasets.refetch()}
            disabled={datasets.isFetching}
            aria-label="Refresh datasets"
          >
            <RefreshCcw className={cn(datasets.isFetching && "animate-spin")} />
            Refresh
          </Button>
        </div>

        {datasets.isLoading ? (
          <DatasetLoading />
        ) : datasets.isError ? (
          <DatasetError message={(datasets.error as Error).message} onRetry={() => datasets.refetch()} />
        ) : items.length === 0 ? (
          <DatasetEmpty filtered={Boolean(query.trim())} onCreate={() => setCreateOpen(true)} />
        ) : (
          <DatasetTable items={items} />
        )}
      </div>

      {createOpen ? (
        <CreateDatasetPanel
          pending={create.isPending}
          error={create.error instanceof Error ? create.error.message : null}
          onClose={() => setCreateOpen(false)}
          onSubmit={(payload) => create.mutate(payload)}
        />
      ) : null}
    </>
  );
}

function DatasetTable({ items }: { items: DatasetRecord[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[880px] text-left text-[12.5px]">
        <thead className="sticky top-0 z-10 bg-bg-subtle text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">
          <tr className="border-b border-border">
            <th className="px-5 py-2 font-medium">Dataset</th>
            <th className="px-3 py-2 font-medium">Source</th>
            <th className="px-3 py-2 font-medium">Modality</th>
            <th className="px-3 py-2 text-right font-medium">Rows</th>
            <th className="px-3 py-2 font-medium">Latest version</th>
            <th className="px-3 py-2 font-medium">Updated</th>
            <th className="w-12 px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {items.map((dataset) => {
            const source = dataset.sources[0];
            const version = dataset.latest_version;
            const activeJob = dataset.active_job;
            return (
              <tr
                key={dataset.id}
                className="group border-b border-border-subtle transition-colors hover:bg-surface/55"
              >
                <td className="px-5 py-3">
                  <Link
                    to="/datasets/$datasetId"
                    params={{ datasetId: dataset.id }}
                    search={{ tab: "overview" }}
                    className="block min-w-0"
                  >
                    <div className="font-medium text-fg group-hover:text-accent">{dataset.name}</div>
                    <div className="mt-0.5 max-w-[42ch] truncate text-[11px] text-fg-subtle">
                      {dataset.description || dataset.id}
                    </div>
                  </Link>
                </td>
                <td className="max-w-[260px] px-3 py-3">
                  <div className="flex items-center gap-2">
                    {source?.kind === "huggingface" ? (
                      <Cloud className="h-3.5 w-3.5 text-fg-disabled" />
                    ) : (
                      <FolderOpen className="h-3.5 w-3.5 text-fg-disabled" />
                    )}
                    <span className="truncate font-mono text-[11px] text-fg-muted">
                      {source?.uri ?? "No source"}
                    </span>
                  </div>
                </td>
                <td className="px-3 py-3 text-fg-muted">{dataset.modality || "text"}</td>
                <td className="px-3 py-3 text-right font-mono text-fg">
                  {formatInteger(version?.row_count ?? dataset.row_count)}
                </td>
                <td className="px-3 py-3">
                  {activeJob ? (
                    <Badge tone="accent" dot size="sm">
                      {activeJob.stage || activeJob.status}
                    </Badge>
                  ) : version ? (
                    <div className="flex items-center gap-2">
                      <Badge tone={versionTone(version.status)} dot size="sm">
                        {version.status}
                      </Badge>
                      <span className="font-mono text-[10px] text-fg-disabled">
                        {shortId(version.id)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-[11px] text-fg-disabled">Not built</span>
                  )}
                </td>
                <td className="px-3 py-3 text-[11px] text-fg-subtle">
                  {relativeDate(dataset.updated_at || dataset.created_at)}
                </td>
                <td className="px-3 py-3 text-right">
                  <Button variant="ghost" size="icon" className="h-7 w-7" asChild>
                    <Link
                      to="/datasets/$datasetId"
                      params={{ datasetId: dataset.id }}
                      search={{ tab: "overview" }}
                      aria-label={`Open ${dataset.name}`}
                    >
                      <ArrowRight />
                    </Link>
                  </Button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CreateDatasetPanel({
  pending,
  error,
  onClose,
  onSubmit,
}: {
  pending: boolean;
  error: string | null;
  onClose: () => void;
  onSubmit: (payload: DatasetCreatePayload) => void;
}) {
  const [sourceKind, setSourceKind] = useState<"local" | "huggingface">("local");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modality, setModality] = useState("text");
  const [uri, setUri] = useState("");
  const [config, setConfig] = useState("");
  const [split, setSplit] = useState("");
  const [revision, setRevision] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!uri.trim()) return;
    onSubmit({
      name: name.trim() || undefined,
      description: description.trim() || undefined,
      modality,
      source: {
        kind: sourceKind,
        uri: uri.trim(),
        config: config.trim() || undefined,
        split: split.trim() || undefined,
        revision: revision.trim() || undefined,
      },
    });
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/55" role="dialog" aria-modal="true" aria-label="Create dataset">
      <button className="flex-1 cursor-default" onClick={onClose} aria-label="Close create dataset panel" />
      <form
        onSubmit={submit}
        className="flex h-full w-full max-w-lg flex-col border-l border-border bg-bg shadow-2xl shadow-black/40"
      >
        <div className="flex items-start justify-between border-b border-border px-5 py-4">
          <div>
            <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Dataset source</div>
            <h2 className="mt-0.5 text-base font-semibold text-fg">Register data</h2>
            <p className="mt-1 text-xs text-fg-muted">Create a stable Dataset Lab record before building versions.</p>
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <X />
          </Button>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
          <div className="grid grid-cols-2 gap-2">
            <SourceChoice
              active={sourceKind === "local"}
              icon={FolderOpen}
              label="Local files"
              detail="JSONL, CSV, Parquet, or a directory"
              onClick={() => setSourceKind("local")}
            />
            <SourceChoice
              active={sourceKind === "huggingface"}
              icon={Cloud}
              label="Hugging Face"
              detail="Dataset repository and revision"
              onClick={() => setSourceKind("huggingface")}
            />
          </div>

          <Field label={sourceKind === "local" ? "Path" : "Repository ID"} required>
            <Input
              value={uri}
              onChange={(event) => setUri(event.target.value)}
              placeholder={sourceKind === "local" ? "/data/project/train.jsonl" : "organization/dataset"}
              mono
              required
              autoFocus
            />
          </Field>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Name">
              <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Inferred when blank" />
            </Field>
            <Field label="Modality">
              <select
                value={modality}
                onChange={(event) => setModality(event.target.value)}
                className="h-9 w-full rounded-md border border-border bg-bg-subtle px-3 text-sm text-fg focus:border-accent focus:outline-none"
              >
                <option value="text">Text</option>
                <option value="code">Code</option>
                <option value="vision">Vision</option>
                <option value="audio">Audio</option>
                <option value="multimodal">Multimodal</option>
                <option value="preference">Preference</option>
              </select>
            </Field>
          </div>

          <Field label="Description">
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              rows={3}
              placeholder="Purpose, scope, or collection notes"
              className="w-full resize-none rounded-md border border-border bg-bg-subtle px-3 py-2 text-sm text-fg placeholder:text-fg-disabled focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30"
            />
          </Field>

          {sourceKind === "huggingface" ? (
            <div className="grid grid-cols-3 gap-3 border-t border-border-subtle pt-5">
              <Field label="Config">
                <Input value={config} onChange={(event) => setConfig(event.target.value)} placeholder="default" mono />
              </Field>
              <Field label="Split">
                <Input value={split} onChange={(event) => setSplit(event.target.value)} placeholder="train" mono />
              </Field>
              <Field label="Pinned revision" required>
                <Input value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="commit SHA" mono required />
              </Field>
            </div>
          ) : null}

          {error ? (
            <div className="border-l-2 border-danger bg-danger-bg px-3 py-2 text-xs text-danger">{error}</div>
          ) : null}
        </div>

        <div className="flex items-center justify-between border-t border-border px-5 py-3">
          <span className="text-[11px] text-fg-subtle">The source is indexed; records are not copied until build.</span>
          <div className="flex gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
            <Button type="submit" variant="primary" size="sm" disabled={pending || !uri.trim() || (sourceKind === "huggingface" && !revision.trim())}>
              {pending ? <Loader2 className="animate-spin" /> : <Plus />}
              Register
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
}

function SourceChoice({ active, icon: Icon, label, detail, onClick }: { active: boolean; icon: typeof Cloud; label: string; detail: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-start gap-3 rounded-md border px-3 py-3 text-left transition-colors",
        active ? "border-accent bg-accent-bg" : "border-border bg-bg-subtle hover:border-border-strong hover:bg-surface",
      )}
    >
      <Icon className={cn("mt-0.5 h-4 w-4", active ? "text-accent" : "text-fg-subtle")} />
      <span>
        <span className={cn("block text-xs font-medium", active ? "text-accent" : "text-fg")}>{label}</span>
        <span className="mt-0.5 block text-[10px] leading-4 text-fg-subtle">{detail}</span>
      </span>
    </button>
  );
}

function Field({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[11px] text-fg-muted">
        {label}{required ? <span className="ml-1 text-accent">*</span> : null}
      </Label>
      {children}
    </div>
  );
}

function DatasetLoading() {
  return (
    <div className="divide-y divide-border-subtle" aria-label="Loading datasets">
      {[0, 1, 2, 3, 4].map((item) => (
        <div key={item} className="flex h-16 animate-pulse items-center gap-4 px-5">
          <div className="h-3 w-40 rounded-sm bg-surface" />
          <div className="h-3 w-64 rounded-sm bg-surface" />
          <div className="ml-auto h-5 w-20 rounded-sm bg-surface" />
        </div>
      ))}
    </div>
  );
}

function DatasetError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-20 text-center">
      <Database className="h-8 w-8 text-danger" />
      <h2 className="mt-4 text-sm font-medium text-fg">Dataset index unavailable</h2>
      <p className="mt-1 max-w-md text-xs text-fg-muted">{message}</p>
      <Button className="mt-4" size="sm" onClick={onRetry}><RefreshCcw />Retry</Button>
    </div>
  );
}

function DatasetEmpty({ filtered, onCreate }: { filtered: boolean; onCreate: () => void }) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-20 text-center">
      <Database className="h-8 w-8 text-fg-disabled" />
      <h2 className="mt-4 text-sm font-medium text-fg">{filtered ? "No matching datasets" : "No datasets registered"}</h2>
      <p className="mt-1 max-w-md text-xs leading-5 text-fg-muted">
        {filtered ? "Change the search query to return to the full index." : "Choose what the model should learn, then inspect and map your source before publishing a reproducible version."}
      </p>
      {!filtered ? <div className="mt-4 flex flex-wrap justify-center gap-2"><Button variant="primary" size="sm" asChild><Link to="/datasets/new" search={{ example: undefined }}><ArrowRight />Train on your data</Link></Button><Button variant="secondary" size="sm" asChild><Link to="/datasets/new" search={{ example: "1" }}>Try a working example</Link></Button><Button variant="ghost" size="sm" onClick={onCreate}>Advanced register</Button></div> : null}
    </div>
  );
}

function formatInteger(value: number | null | undefined): string {
  return typeof value === "number" ? new Intl.NumberFormat().format(value) : "—";
}

function shortId(value: string): string {
  return value.length > 10 ? value.slice(0, 8) : value;
}

function relativeDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const seconds = Math.round((date.getTime() - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (Math.abs(seconds) < 60) return formatter.format(seconds, "second");
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return formatter.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return formatter.format(hours, "hour");
  return formatter.format(Math.round(hours / 24), "day");
}

function versionTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" {
  if (status === "ready" || status === "completed") return "success";
  if (status === "failed") return "danger";
  if (status === "building" || status === "running" || status === "queued") return "accent";
  if (status === "cancelled") return "warning";
  return "neutral";
}
