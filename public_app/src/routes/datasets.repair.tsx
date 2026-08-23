import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, ArrowRight, CheckCircle2, Loader2, RefreshCw, Wrench } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, type DatasetIssue, type DatasetRepairAction } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

type RepairSearch = { session?: string; inspection?: string; source?: string };

export const Route = createFileRoute("/datasets/repair")({
  validateSearch: (value: Record<string, unknown>): RepairSearch => ({
    session: typeof value.session === "string" ? value.session : undefined,
    inspection: typeof value.inspection === "string" ? value.inspection : undefined,
    source: typeof value.source === "string" ? value.source : undefined,
  }),
  component: DatasetRepairRoute,
});

function DatasetRepairRoute() {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: "/datasets/repair" });
  const [source, setSource] = useState(search.source ?? "");
  const [actions, setActions] = useState<DatasetRepairAction[]>([]);
  const [selected, setSelected] = useState<DatasetIssue | null>(null);
  const [editField, setEditField] = useState("");
  const [editValue, setEditValue] = useState("");
  const [mediaRoot, setMediaRoot] = useState("");
  const [aliasFrom, setAliasFrom] = useState("");
  const [aliasTo, setAliasTo] = useState("");
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [published, setPublished] = useState<Record<string, unknown> | null>(null);
  const [issueOffset, setIssueOffset] = useState(0);
  const issueLimit = 100;

  const create = useMutation({
    mutationFn: () => api.createDatasetRepair(search.inspection ? { inspection_id: search.inspection } : { source_uri: source }),
    onSuccess: (value) => navigate({ search: { session: value.id, inspection: undefined, source: undefined }, replace: true }),
  });
  const session = useQuery({
    queryKey: ["dataset-repair", search.session],
    queryFn: () => api.datasetRepair(search.session!),
    enabled: Boolean(search.session),
    refetchInterval: (query) => ["scanning", "previewing"].includes(query.state.data?.status ?? "") ? 1_500 : false,
  });
  const issues = useQuery({
    queryKey: ["dataset-repair-issues", search.session, issueOffset],
    queryFn: () => api.datasetRepairIssues(search.session!, { limit: issueLimit, offset: issueOffset }),
    enabled: Boolean(search.session && session.data && session.data.status !== "scanning"),
  });
  const savedPlan = useQuery({
    queryKey: ["dataset-repair-plan", session.data?.latest_plan_revision_id],
    queryFn: () => api.datasetRepairPlan(session.data!.latest_plan_revision_id!),
    enabled: Boolean(session.data?.latest_plan_revision_id),
  });
  const savedRevision = useQuery({
    queryKey: ["dataset-repair-revision", session.data?.published_repair_revision_id],
    queryFn: () => api.datasetRepairRevision(session.data!.published_repair_revision_id!),
    enabled: Boolean(session.data?.published_repair_revision_id),
  });
  useEffect(() => {
    if (savedPlan.data) {
      setActions((current) => current.length ? current : savedPlan.data.actions);
    }
  }, [savedPlan.data]);
  const runPreview = useMutation({
    mutationFn: async () => {
      if (!search.session) throw new Error("Repair session is missing");
      const plan = await api.createDatasetRepairPlan(search.session, actions);
      return api.createDatasetRepairPreview(search.session, plan.id);
    },
    onSuccess: (value) => setPreviewId(value.id),
  });
  const activePreviewId = previewId ?? session.data?.latest_preview_id ?? null;
  const preview = useQuery({
    queryKey: ["dataset-repair-preview", activePreviewId],
    queryFn: () => api.datasetRepairPreview(activePreviewId!),
    enabled: Boolean(activePreviewId),
    refetchInterval: (query) => ["queued", "running"].includes(query.state.data?.status ?? "") ? 1_500 : false,
  });
  const publish = useMutation({
    mutationFn: () => api.publishDatasetRepair(previewId!),
    onSuccess: setPublished,
  });
  const issueItems = issues.data?.items ?? [];
  const selectedIds = useMemo(() => new Set(actions.map((item) => `${item.action_kind}:${item.record_id ?? "all"}:${item.source_index ?? "all"}:${item.field_path ?? ""}`)), [actions]);

  function addAction(action: DatasetRepairAction) {
    const key = `${action.action_kind}:${action.record_id ?? "all"}:${action.source_index ?? "all"}:${action.field_path ?? ""}`;
    setActions((current) => selectedIds.has(key) ? current.filter((item) => `${item.action_kind}:${item.record_id ?? "all"}:${item.source_index ?? "all"}:${item.field_path ?? ""}` !== key) : [...current, action]);
  }

  if (!search.session) {
    return (
      <>
        <Topbar eyebrow="Data" title="Fix data" subtitle="Find safe, deterministic fixes without changing the original source." />
        <main className="mx-auto max-w-3xl px-5 py-6">
          <Card><CardHeader><CardTitle>{search.inspection ? "Review the inspected source" : "Choose the source to inspect"}</CardTitle></CardHeader><CardContent className="space-y-4">
            {!search.inspection ? <><label className="text-xs font-medium text-fg-muted" htmlFor="repair-source">Workstation file or manifest</label><Input id="repair-source" value={source} onChange={(event) => setSource(event.target.value)} placeholder="/path/to/data.jsonl" /></> : <p className="text-sm text-fg-muted">Halo Forge will scan the complete source behind this inspection and preserve its fingerprint.</p>}
            <Button variant="primary" onClick={() => create.mutate()} disabled={create.isPending || (!source && !search.inspection)}>{create.isPending ? <Loader2 className="animate-spin" /> : <Wrench />}Find problems</Button>
            {create.isError ? <p className="text-xs text-danger">{String(create.error)}</p> : null}
          </CardContent></Card>
        </main>
      </>
    );
  }

  const currentSession = session.data;
  const exactPreview = preview.data;
  const activePublished = published ?? savedRevision.data ?? null;
  const publishedRevisionId = typeof activePublished?.id === "string"
    ? activePublished.id
    : typeof activePublished?.revision_id === "string"
      ? activePublished.revision_id
      : currentSession?.published_repair_revision_id;
  return (
    <>
      <Topbar
        eyebrow="Data · Own data"
        title="Fix data"
        subtitle={currentSession?.status === "scanning" ? "Scanning the complete source…" : "Review each change before publishing an immutable repair."}
        actions={<Button asChild variant="ghost" size="sm"><Link to="/datasets">Back to datasets</Link></Button>}
      />
      <main className="grid min-h-[calc(100vh-64px)] lg:grid-cols-[320px_minmax(0,1fr)_300px]">
        <aside className="border-r border-border-subtle">
          <div className="border-b border-border-subtle px-4 py-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-disabled">Detected problems</p>
            <p className="mt-1 text-xs text-fg-muted">{issues.data?.total ?? 0} exact issues · original source unchanged</p>
          </div>
          {session.isLoading || currentSession?.status === "scanning" ? <div className="flex items-center gap-2 p-4 text-sm text-fg-muted"><Loader2 className="animate-spin" />Scanning records…</div> : (
            <div className="max-h-[calc(100vh-150px)] overflow-auto">
              {issueItems.map((issue) => <button key={issue.id} onClick={() => setSelected(issue)} className={`w-full border-b border-border-subtle px-4 py-3 text-left hover:bg-bg-subtle ${selected?.id === issue.id ? "bg-bg-subtle" : ""}`}><div className="flex items-center justify-between gap-2"><span className="text-xs font-medium text-fg">{issue.category.replaceAll("_", " ")}</span><Badge tone={issue.severity === "error" ? "warning" : "neutral"}>{issue.severity}</Badge></div><p className="mt-1 line-clamp-2 text-[11px] text-fg-muted">{issue.message}</p></button>)}
              {!issueItems.length ? <p className="p-4 text-sm text-fg-muted">No repairable issues were found.</p> : null}
              {(issues.data?.total ?? 0) > issueLimit ? <div className="flex items-center justify-between border-t border-border-subtle p-3"><Button variant="ghost" size="sm" onClick={() => setIssueOffset((value) => Math.max(0, value - issueLimit))} disabled={issueOffset === 0}>Previous</Button><span className="text-[11px] text-fg-muted">{issueOffset + 1}–{Math.min(issueOffset + issueLimit, issues.data?.total ?? 0)} of {issues.data?.total}</span><Button variant="ghost" size="sm" onClick={() => setIssueOffset((value) => value + issueLimit)} disabled={issueOffset + issueLimit >= (issues.data?.total ?? 0)}>Next</Button></div> : null}
            </div>
          )}
        </aside>

        <section className="min-w-0 space-y-5 px-5 py-5">
          {selected ? <>
            <div><div className="flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-warning" /><h2 className="text-base font-semibold text-fg">{selected.message}</h2></div><p className="mt-2 font-mono text-[11px] text-fg-disabled">Record {selected.source_index != null ? selected.source_index + 1 : "—"} · {selected.field_path ?? "whole record"}</p></div>
            <div className="grid gap-3 md:grid-cols-2"><RecordPanel label="Detected evidence" value={selected.evidence} /><RecordPanel label="Planned result" value={{ action: actions.find((item) => item.record_id === selected.record_id) ?? "No change selected yet", original_preserved: true }} /></div>
            <div className="flex flex-wrap gap-2"><Button variant="secondary" onClick={() => addAction({ action_kind: "quarantine", record_id: selected.record_id, source_index: selected.source_index, issue_code: selected.code, reason: "Reviewed unresolved record" })}>Quarantine record</Button><Button variant="ghost" onClick={() => addAction({ action_kind: "exclude", record_id: selected.record_id, source_index: selected.source_index, issue_code: selected.code, reason: "Reviewed explicit exclusion" })}>Exclude record</Button></div>
            <div className="rounded border border-border-subtle bg-surface/30 p-4"><p className="mb-3 text-xs font-medium text-fg">Edit one field</p><div className="grid gap-2 md:grid-cols-[1fr_1fr_auto]"><Input value={editField} onChange={(event) => setEditField(event.target.value)} placeholder="field path" /><Input value={editValue} onChange={(event) => setEditValue(event.target.value)} placeholder="replacement value" /><Button variant="secondary" onClick={() => addAction({ action_kind: "edit", record_id: selected.record_id, source_index: selected.source_index, field_path: editField, value: editValue, issue_code: selected.code, reason: "Operator-reviewed field correction" })} disabled={!editField}>Add edit</Button></div></div>
          </> : <Card><CardContent className="py-8 text-sm text-fg-muted">Select a detected problem to compare the evidence and choose a fix.</CardContent></Card>}

          {exactPreview?.sample?.length ? <section><h2 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-muted">Exact before and after preview</h2><div className="space-y-3">{exactPreview.sample.slice(0, 5).map((item, index) => <div key={index} className="grid gap-3 md:grid-cols-2"><RecordPanel label="Original" value={item.before ?? item.original} /><RecordPanel label="Repaired" value={item.after ?? item.repaired ?? item} /></div>)}</div></section> : null}
        </section>

        <aside className="border-l border-border-subtle bg-bg-subtle/20 p-4">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-muted">Repair plan</h2>
          <p className="mt-2 text-sm text-fg">{actions.length} reviewed {actions.length === 1 ? "change" : "changes"}</p>
          <p className="mt-1 text-xs text-fg-muted">The original file and binary media will not be edited.</p>
          <div className="mt-4 space-y-2 border-t border-border-subtle pt-4"><p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-disabled">Safe source-wide fixes</p><Button className="w-full" variant="secondary" size="sm" onClick={() => addAction({ action_kind: "trim", reason: "Reviewed whitespace and empty-turn cleanup" })}>Trim whitespace and empty turns</Button><Button className="w-full" variant="secondary" size="sm" onClick={() => addAction({ action_kind: "normalize_roles", reason: "Reviewed standard chat-role mapping" })}>Normalize chat roles</Button><div className="flex gap-1"><Input value={mediaRoot} onChange={(event) => setMediaRoot(event.target.value)} placeholder="Correct media folder" /><Button variant="ghost" size="sm" onClick={() => addAction({ action_kind: "media_root", value: mediaRoot, reason: "Reviewed relative media root" })} disabled={!mediaRoot}>Add</Button></div><div className="grid grid-cols-2 gap-1"><Input value={aliasFrom} onChange={(event) => setAliasFrom(event.target.value)} placeholder="Label alias" /><Input value={aliasTo} onChange={(event) => setAliasTo(event.target.value)} placeholder="Canonical label" /></div><Button className="w-full" variant="ghost" size="sm" onClick={() => addAction({ action_kind: "label_alias", field_path: "label", value: { [aliasFrom]: aliasTo }, reason: "Reviewed label alias" })} disabled={!aliasFrom || !aliasTo}>Map label alias</Button></div>
          <div className="mt-4 space-y-2"><Button className="w-full" variant="primary" onClick={() => runPreview.mutate()} disabled={!actions.length || runPreview.isPending}>{runPreview.isPending ? <Loader2 className="animate-spin" /> : <RefreshCw />}Preview exact changes</Button></div>
          {exactPreview ? <div className="mt-5 space-y-2 border-t border-border-subtle pt-4"><p className="text-xs font-medium text-fg">{exactPreview.exact ? "Exact full-source counts" : "Estimate"}</p>{Object.entries(exactPreview.counts).map(([key, value]) => <div key={key} className="flex justify-between text-xs"><span className="text-fg-muted">{key.replaceAll("_", " ")}</span><span className="font-mono text-fg">{value}</span></div>)}{exactPreview.status === "completed" ? <Button className="mt-3 w-full" variant="primary" onClick={() => publish.mutate()} disabled={publish.isPending || Boolean(published)}><CheckCircle2 />Publish repair</Button> : <p className="flex items-center gap-2 text-xs text-fg-muted"><Loader2 className="h-3 w-3 animate-spin" />Completing full scan…</p>}</div> : null}
          {activePublished ? <div className="mt-4 rounded border border-success/40 p-3"><p className="text-sm font-medium text-success">Repair published</p><p className="mt-1 text-xs text-fg-muted">Continue through the normal mapping and version preview. Publishing the repair did not publish a dataset or start training.</p><Button asChild className="mt-3 w-full" variant="primary"><Link to="/datasets/new" search={{ source: currentSession?.source_uri, scenario: currentSession?.scenario_revision_id ?? undefined, repairRevision: publishedRevisionId ?? undefined }}>Continue to dataset <ArrowRight /></Link></Button><Button asChild className="mt-2 w-full" variant="ghost"><Link to="/datasets">Open dataset library</Link></Button></div> : null}
        </aside>
      </main>
    </>
  );
}

function RecordPanel({ label, value }: { label: string; value: unknown }) {
  return <div className="min-w-0 rounded border border-border-subtle bg-bg-subtle p-3"><p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-disabled">{label}</p><div className="max-h-72 space-y-2 overflow-auto text-xs text-fg-muted"><ReadableValue value={value} /></div></div>;
}

function ReadableValue({ value }: { value: unknown }) {
  if (value == null) return <p className="italic text-fg-disabled">No value</p>;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return <p className="whitespace-pre-wrap break-words">{String(value)}</p>;
  if (Array.isArray(value)) return <div className="space-y-2">{value.map((item, index) => <div key={index} className="rounded border border-border-subtle bg-surface/40 p-2"><span className="mb-1 block text-[10px] font-medium text-fg-disabled">Item {index + 1}</span><ReadableValue value={item} /></div>)}</div>;
  if (typeof value === "object") return <dl className="space-y-2">{Object.entries(value as Record<string, unknown>).map(([key, item]) => <div key={key}><dt className="text-[10px] font-semibold uppercase tracking-[0.08em] text-fg-disabled">{key.replaceAll("_", " ")}</dt><dd className="mt-0.5"><ReadableValue value={item} /></dd></div>)}</dl>;
  return <p>{String(value)}</p>;
}
