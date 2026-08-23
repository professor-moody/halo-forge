import { useEffect, useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Ban,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  Clock3,
  Cpu,
  HardDrive,
  Loader2,
  Play,
  RefreshCw,
  RotateCcw,
  ServerCog,
  X,
} from "lucide-react";
import { api, type ActivityItem } from "@/lib/api";
import { useEventSource } from "@/lib/event-source";
import { queryKeys, useActivity } from "@/lib/hooks";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn, relativeTime } from "@/lib/utils";

type ActivityFilter = "active" | "attention" | "history";

export function ActivityCenter({ open, onClose }: { open: boolean; onClose: () => void }) {
  const queryClient = useQueryClient();
  const activity = useActivity(150);
  const [selectedId, setSelectedId] = useState("");
  const [filter, setFilter] = useState<ActivityFilter>("active");
  const [retryReason, setRetryReason] = useState("");
  const [reviewReason, setReviewReason] = useState("");
  const [streamEnabled, setStreamEnabled] = useState(true);
  const stream = useEventSource<Record<string, unknown>>(
    open && streamEnabled ? api.activityEventsUrl : null,
  );

  useEffect(() => {
    if (!stream.data) return;
    queryClient.invalidateQueries({ queryKey: queryKeys.activity });
  }, [queryClient, stream.data]);

  useEffect(() => {
    if (stream.status !== "error" || !stream.error) return;
    if (/404|405|not found/i.test(stream.error)) setStreamEnabled(false);
  }, [stream.error, stream.status]);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, open]);

  const items = activity.data?.items ?? [];
  const visible = useMemo(() => {
    return items.filter((item) => {
      if (filter === "active") return ["queued", "running", "blocked", "preparing", "waiting_for_accelerator", "needs_reconciliation", "awaiting_review"].includes(item.status);
      if (filter === "attention") return ["failed", "interrupted", "blocked", "needs_reconciliation", "warn", "awaiting_review"].includes(item.status);
      return !["queued", "running", "blocked", "preparing", "waiting_for_accelerator", "awaiting_review"].includes(item.status);
    });
  }, [filter, items]);

  useEffect(() => {
    if (selectedId && items.some((item) => item.id === selectedId)) return;
    const first = visible[0] ?? items[0];
    setSelectedId(first?.id ?? "");
  }, [items, selectedId, visible]);

  const selected = items.find((item) => item.id === selectedId) ?? null;
  const worker = activity.data?.worker ?? activity.data?.workers?.[0] ?? null;
  const active = items.filter((item) => ["queued", "running", "blocked", "preparing", "waiting_for_accelerator", "awaiting_review"].includes(item.status));
  const attention = items.filter((item) => ["failed", "interrupted", "needs_reconciliation", "awaiting_review"].includes(item.status));

  const retry = useMutation<unknown, Error, { id: string; domainId?: string | null; domainType?: string | null; kind: string; reason: string }>({
    mutationFn: ({ id, domainId, domainType, kind, reason }: { id: string; domainId?: string | null; domainType?: string | null; kind: string; reason: string }) => {
      if (domainId && ((domainType ?? "").includes("reward_integrity_audit") || kind.includes("reward_integrity_audit"))) {
        return api.retryRewardIntegrityAudit(domainId, reason);
      }
      return api.retryWorkItem(id, reason);
    },
    onSuccess: () => {
      setRetryReason("");
      queryClient.invalidateQueries({ queryKey: queryKeys.activity });
      queryClient.invalidateQueries({ queryKey: ["reward-integrity-audits"] });
    },
  });
  const cancel = useMutation({
    mutationFn: (id: string) => api.cancelWorkItem(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.activity }),
  });
  const reviewGate = useMutation({
    mutationFn: async ({ id, domainType, action, reason }: { id: string; domainType?: string | null; action: "continue" | "stop" | "fork"; reason: string }) => {
      if ((domainType ?? "").includes("reward_integrity")) return api.reviewRewardIntegrityAudit(id, { action, reason });
      return api.reviewGateDecision(id, { action: action === "fork" ? "stop" : action, reason });
    },
    onSuccess: (result, variables) => {
      setReviewReason("");
      queryClient.invalidateQueries({ queryKey: queryKeys.activity });
      queryClient.invalidateQueries({ queryKey: ["run-groups"] });
      if (variables.action === "fork" && result && typeof result === "object" && "href" in result && typeof result.href === "string") {
        window.location.assign(result.href);
      }
    },
  });

  if (!open) return null;

  return (
    <div className="workspace-overlay justify-end" role="presentation" onMouseDown={onClose}>
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="activity-center-title"
        className="activity-panel-enter flex h-full w-full max-w-[720px] flex-col border-l border-border bg-bg shadow-2xl shadow-black/40"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="border-b border-border bg-bg-subtle/90 px-4 py-3 backdrop-blur">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 grid h-8 w-8 place-items-center rounded-md bg-accent-bg text-accent">
              <Activity className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 id="activity-center-title" className="text-[15px] font-semibold tracking-tight text-fg">Activity Center</h2>
                <Badge tone={worker?.status === "online" ? "success" : worker ? "warning" : "neutral"} size="sm" dot>
                  {worker ? `worker ${worker.status}` : "worker unavailable"}
                </Badge>
              </div>
              <p className="mt-0.5 text-[11px] text-fg-muted">
                Queue, resource ownership, retries, and workstation health in one place.
              </p>
            </div>
            <Button variant="ghost" size="icon" onClick={() => activity.refetch()} title="Refresh activity">
              <RefreshCw className={cn("h-4 w-4", activity.isFetching && "animate-spin")} />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose} title="Close Activity Center">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="mt-3 grid grid-cols-3 divide-x divide-border-subtle rounded-md border border-border-subtle bg-surface/45">
            <HeaderReadout label="ACTIVE" value={String(active.length)} accent={active.length > 0} />
            <HeaderReadout label="ATTENTION" value={String(attention.length)} warning={attention.length > 0} />
            <HeaderReadout
              label="UPDATES"
              value={stream.status === "open" ? "LIVE" : activity.isFetching ? "SYNC" : "POLL"}
              accent={stream.status === "open"}
            />
          </div>
        </header>

        <ResourceStrip
          lease={activity.data?.resource_lease ?? null}
          storage={activity.data?.storage ?? null}
          workerHeartbeat={worker?.heartbeat_at ?? null}
        />

        <div className="grid min-h-0 flex-1 sm:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="flex min-h-0 flex-col border-b border-border sm:border-b-0 sm:border-r">
            <div className="flex items-center gap-1 border-b border-border-subtle px-2 py-2">
              <FilterButton label="Active" count={active.length} active={filter === "active"} onClick={() => setFilter("active")} />
              <FilterButton label="Attention" count={attention.length} active={filter === "attention"} onClick={() => setFilter("attention")} />
              <FilterButton label="History" count={Math.max(0, items.length - active.length)} active={filter === "history"} onClick={() => setFilter("history")} />
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto divide-y divide-border-subtle">
              {activity.isLoading ? (
                <PanelMessage icon={Loader2} spin label="Loading workstation activity" />
              ) : activity.isError ? (
                <PanelMessage icon={AlertTriangle} label="Activity is unavailable" detail="The dashboard will reconnect automatically." tone="danger" />
              ) : visible.length ? (
                visible.map((item) => (
                  <ActivityRow key={item.id} item={item} selected={item.id === selectedId} onSelect={() => setSelectedId(item.id)} />
                ))
              ) : (
                <PanelMessage
                  icon={filter === "active" ? CheckCircle2 : CircleDashed}
                  label={filter === "active" ? "Workstation is clear" : "Nothing in this view"}
                  detail={filter === "active" ? "Queued operations will appear here." : "Try another activity filter."}
                />
              )}
            </div>
          </aside>

          <main className="min-h-0 overflow-y-auto">
            {selected ? (
              <ActivityInspector
                item={selected}
                retryReason={retryReason}
                onRetryReason={setRetryReason}
                reviewReason={reviewReason}
                onReviewReason={setReviewReason}
                onRetry={() => retry.mutate({ id: selected.work_item_id ?? selected.id, domainId: selected.domain_id, domainType: selected.domain_type, kind: selected.kind, reason: retryReason.trim() })}
                onCancel={() => cancel.mutate(selected.work_item_id ?? selected.id)}
                retryPending={retry.isPending}
                cancelPending={cancel.isPending}
                onReview={(action) => selected.domain_id && reviewGate.mutate({ id: selected.domain_id, domainType: selected.domain_type, action, reason: reviewReason.trim() })}
                reviewPending={reviewGate.isPending}
                actionError={(retry.error ?? cancel.error ?? reviewGate.error) instanceof Error ? (retry.error ?? cancel.error ?? reviewGate.error)?.message ?? null : null}
              />
            ) : (
              <PanelMessage icon={Activity} label="Select an activity" detail="Attempts, events, telemetry, and next actions appear here." />
            )}
          </main>
        </div>
      </section>
    </div>
  );
}

function ResourceStrip({
  lease,
  storage,
  workerHeartbeat,
}: {
  lease: Record<string, unknown> | null;
  storage: import("@/lib/api").StorageInventory | null;
  workerHeartbeat: string | null;
}) {
  const free = storage?.projected_free_bytes ?? storage?.free_bytes ?? null;
  const owner = String(lease?.owner ?? lease?.work_item_id ?? lease?.kind ?? "available");
  return (
    <div className="grid gap-px border-b border-border bg-border-subtle sm:grid-cols-3">
      <ResourceCell icon={Cpu} label="Heavy resource" value={owner} tone={owner === "available" ? "success" : "accent"} />
      <ResourceCell icon={HardDrive} label="Projected free" value={free == null ? "Not measured" : formatBytes(free)} tone={storage?.low_disk ? "warning" : "neutral"} />
      <ResourceCell icon={ServerCog} label="Worker heartbeat" value={workerHeartbeat ? relativeTime(workerHeartbeat) : "Not reported"} tone="neutral" />
    </div>
  );
}

function ResourceCell({ icon: Icon, label, value, tone }: { icon: typeof Cpu; label: string; value: string; tone: "success" | "accent" | "warning" | "neutral" }) {
  return (
    <div className="flex min-w-0 items-center gap-2 bg-bg-subtle/45 px-3 py-2">
      <Icon className={cn("h-3.5 w-3.5", tone === "success" && "text-success", tone === "accent" && "text-accent", tone === "warning" && "text-warning", tone === "neutral" && "text-fg-disabled")} />
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-wider text-fg-disabled">{label}</div>
        <div className="truncate font-mono text-[10.5px] text-fg-muted">{value}</div>
      </div>
    </div>
  );
}

function HeaderReadout({ label, value, accent, warning }: { label: string; value: string; accent?: boolean; warning?: boolean }) {
  return (
    <div className="px-3 py-2 text-center">
      <div className="text-[9px] uppercase tracking-[0.12em] text-fg-disabled">{label}</div>
      <div className={cn("mt-0.5 font-mono text-[12px] text-fg", accent && "text-accent", warning && "text-warning")}>{value}</div>
    </div>
  );
}

function FilterButton({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className={cn("flex h-7 flex-1 items-center justify-center gap-1.5 rounded-sm text-[10.5px] transition-colors", active ? "bg-accent-bg text-accent" : "text-fg-subtle hover:bg-surface hover:text-fg")}>
      {label}<span className="font-mono text-[9px] opacity-70">{count}</span>
    </button>
  );
}

function ActivityRow({ item, selected, onSelect }: { item: ActivityItem; selected: boolean; onSelect: () => void }) {
  const progress = activityProgress(item);
  const pairedCoverage = item.summary_metrics?.paired_coverage;
  return (
    <button type="button" onClick={onSelect} className={cn("group relative w-full px-3 py-3 text-left transition-colors hover:bg-surface/55", selected && "bg-accent-bg/55")}>
      {selected ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}
      <div className="flex items-start gap-2.5">
        <StatusGlyph status={item.status} />
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <div className="truncate text-[12px] font-medium capitalize text-fg">{item.title || item.kind.replace(/[_-]/g, " ")}</div>
            <ChevronRight className={cn("mt-0.5 h-3 w-3 shrink-0 text-fg-disabled transition-transform", selected && "translate-x-0.5 text-accent")} />
          </div>
          <div className="mt-1 flex items-center gap-2 font-mono text-[9.5px] uppercase tracking-wide text-fg-disabled">
            <span>{item.status.replace(/_/g, " ")}</span>
            {item.stage ? <><span>·</span><span className="truncate">{item.stage}</span></> : null}
            {typeof pairedCoverage === "number" ? <><span>·</span><span>{formatPercent(pairedCoverage)} paired</span></> : null}
          </div>
          {progress != null ? (
            <div className="mt-2 h-0.5 overflow-hidden bg-surface-pressed">
              <div className="h-full bg-accent transition-[width] duration-500" style={{ width: `${progress}%` }} />
            </div>
          ) : null}
          <div className="mt-1.5 flex items-center justify-between text-[9.5px] text-fg-disabled">
            <span>{item.queue_position ? `Queue ${item.queue_position}` : item.attempt ? `Attempt ${item.attempt}/${item.max_attempts ?? "–"}` : ""}</span>
            <span>{item.eta_seconds != null ? formatDuration(item.eta_seconds) : item.created_at ? relativeTime(item.created_at) : ""}</span>
          </div>
        </div>
      </div>
    </button>
  );
}

function ActivityInspector({
  item,
  retryReason,
  onRetryReason,
  reviewReason,
  onReviewReason,
  onRetry,
  onCancel,
  retryPending,
  cancelPending,
  onReview,
  reviewPending,
  actionError,
}: {
  item: ActivityItem;
  retryReason: string;
  onRetryReason: (value: string) => void;
  reviewReason: string;
  onReviewReason: (value: string) => void;
  onRetry: () => void;
  onCancel: () => void;
  retryPending: boolean;
  cancelPending: boolean;
  onReview: (action: "continue" | "stop" | "fork") => void;
  reviewPending: boolean;
  actionError: string | null;
}) {
  const retryable = ["failed", "interrupted", "needs_reconciliation", "cancelled"].includes(item.status);
  const cancellable = ["queued", "running", "blocked", "preparing", "waiting_for_accelerator"].includes(item.status);
  const awaitingReview = item.status === "awaiting_review" && Boolean(item.domain_id) && (["gate_decision", "checkpoint_gate_decision", "checkpoint_gate"].includes(item.domain_type ?? "") || (item.domain_type ?? "").includes("reward_integrity_audit") || item.kind.includes("reward_integrity_audit"));
  const progress = activityProgress(item);
  const verifierCalibrationId = item.domain_id && ((item.domain_type ?? "").includes("verifier_calibration") || item.kind.includes("verifier_calibration")) ? item.domain_id : null;
  const actionLinks = [...(item.action_links ?? [])];
  const rewardAuditId = item.domain_id && ((item.domain_type ?? "").includes("reward_integrity_audit") || item.kind.includes("reward_integrity_audit")) ? item.domain_id : null;
  if (rewardAuditId && !actionLinks.some((link) => link.href.includes(`audit=${encodeURIComponent(rewardAuditId)}`))) {
    actionLinks.push({ id: "open-reward-integrity-audit", label: "Open Training Audit", href: `/eval?section=verifiers&verifierView=training-audits&auditView=results&audit=${encodeURIComponent(rewardAuditId)}` });
  }
  if (rewardAuditId && ["completed", "succeeded", "pass", "warn", "fail", "awaiting_review"].includes(item.status) && !actionLinks.some((link) => link.id === "compare-reward-integrity-audit")) {
    actionLinks.push({ id: "compare-reward-integrity-audit", label: "Compare Audit", href: `/eval?section=verifiers&verifierView=training-audits&auditView=compare&auditBase=${encodeURIComponent(rewardAuditId)}` });
  }
  if (verifierCalibrationId && ["completed", "succeeded", "pass", "warn"].includes(item.status) && !actionLinks.some((link) => link.href.includes("source=verifier_calibration"))) {
    actionLinks.push({ id: "open-verifier-review-proposal", label: "Open Review Proposal", href: `/datasets/review?new=1&source=verifier_calibration&sourceRef=${encodeURIComponent(verifierCalibrationId)}` });
  }
  return (
    <div className="min-h-full">
      <div className="border-b border-border-subtle px-4 py-4">
        <div className="flex items-center gap-2">
          <StatusGlyph status={item.status} />
          <div className="text-[10px] uppercase tracking-[0.13em] text-fg-disabled">{item.kind.replace(/[_-]/g, " ")}</div>
        </div>
        <h3 className="mt-2 text-[17px] font-medium leading-tight text-fg">{item.title || item.stage || item.kind}</h3>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <StatusBadge status={item.status} />
          {item.queue_position ? <Badge tone="neutral" size="sm">queue {item.queue_position}</Badge> : null}
          {item.attempt ? <Badge tone="neutral" size="sm">attempt {item.attempt}/{item.max_attempts ?? "–"}</Badge> : null}
        </div>
        {progress != null ? (
          <div className="mt-4">
            <div className="mb-1.5 flex justify-between font-mono text-[10px] text-fg-disabled"><span>{item.stage || "Progress"}</span><span>{Math.round(progress)}%</span></div>
            <div className="h-1 overflow-hidden rounded-full bg-surface-pressed"><div className="h-full bg-accent transition-[width] duration-500" style={{ width: `${progress}%` }} /></div>
          </div>
        ) : null}
      </div>

      <InspectorSection title="Timing">
        <KeyValue label="Created" value={item.created_at ? relativeTime(item.created_at) : "Not reported"} />
        <KeyValue label="Started" value={item.started_at ? relativeTime(item.started_at) : "Waiting"} />
        <KeyValue label="Heartbeat" value={item.heartbeat_at ? relativeTime(item.heartbeat_at) : "Not reported"} />
        <KeyValue label="ETA" value={item.eta_seconds != null ? formatDuration(item.eta_seconds) : "Learning from progress"} />
      </InspectorSection>

      {item.blockers?.length ? (
        <InspectorSection title="Blocked by">
          <ul className="space-y-1.5 text-[11px] text-warning">{item.blockers.map((blocker) => <li key={blocker} className="flex gap-2"><AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />{blocker}</li>)}</ul>
        </InspectorSection>
      ) : null}

      {item.error ? (
        <InspectorSection title="Failure">
          <div className="rounded-sm border border-danger/25 bg-danger-bg px-3 py-2 font-mono text-[10.5px] leading-relaxed text-danger">{item.error}</div>
        </InspectorSection>
      ) : null}

      {item.telemetry_rollup && Object.keys(item.telemetry_rollup).length ? (
        <InspectorSection title="Resource summary">
          {Object.entries(item.telemetry_rollup).slice(0, 8).map(([key, value]) => <KeyValue key={key} label={key.replace(/_/g, " ")} value={value == null ? "Unavailable" : String(value)} />)}
        </InspectorSection>
      ) : null}

      {rewardAuditId ? (
        <InspectorSection title="Audit evidence">
          <KeyValue label="Paired coverage" value={formatPercent(item.summary_metrics?.paired_coverage)} mono />
        </InspectorSection>
      ) : null}

      {item.events?.length ? (
        <InspectorSection title="Recent events">
          <div className="space-y-2">{item.events.slice(-6).reverse().map((event) => <div key={event.id} className="border-l border-border-strong pl-2"><div className="text-[10.5px] text-fg-muted">{event.message || event.type.replace(/_/g, " ")}</div><div className="mt-0.5 font-mono text-[9px] text-fg-disabled">{event.created_at ? relativeTime(event.created_at) : ""}</div></div>)}</div>
        </InspectorSection>
      ) : null}

      {item.logs?.length ? (
        <InspectorSection title="Latest logs">
          <pre className="max-h-44 overflow-auto rounded-sm border border-border-subtle bg-bg-subtle p-2 font-mono text-[9.5px] leading-relaxed text-fg-subtle">{item.logs.slice(-20).join("\n")}</pre>
        </InspectorSection>
      ) : null}

      {actionLinks.length ? (
        <InspectorSection title="Open related work">
          <div className="grid gap-2">
            {actionLinks.map((link) => (
              <Button key={link.id} size="sm" variant="secondary" asChild className="justify-between">
                <a href={link.href}>{link.label}<ArrowUpRight /></a>
              </Button>
            ))}
          </div>
        </InspectorSection>
      ) : null}

      {(retryable || cancellable || awaitingReview) ? (
        <InspectorSection title="Next action">
          {awaitingReview ? (
            <div className="mb-3 space-y-2 border-l-2 border-warning pl-3">
              <p className="text-[10.5px] leading-relaxed text-fg-subtle">Training paused at a verified boundary. Review the checkpoint and reward-integrity evidence before choosing what happens next.</p>
              <Input value={reviewReason} onChange={(event) => onReviewReason(event.target.value)} placeholder="Required review reason" className="h-8 text-[11px]" />
              <div className="flex flex-wrap gap-2"><Button size="sm" onClick={() => onReview("continue")} disabled={!reviewReason.trim() || reviewPending}>{reviewPending ? <Loader2 className="animate-spin" /> : <Play />} Continue</Button><Button size="sm" variant="ghost" onClick={() => onReview("stop")} disabled={!reviewReason.trim() || reviewPending}><Ban /> Stop</Button>{rewardAuditId ? <Button size="sm" variant="ghost" onClick={() => onReview("fork")} disabled={!reviewReason.trim() || reviewPending}><ArrowUpRight /> Fork</Button> : null}</div>
            </div>
          ) : null}
          {retryable ? (
            <div className="space-y-2">
              <p className="text-[10.5px] leading-relaxed text-fg-subtle">A forced retry is recorded with your reason and runs in a fresh attempt directory.</p>
              <Input value={retryReason} onChange={(event) => onRetryReason(event.target.value)} placeholder="Reason for retry" aria-label="Required retry reason" className="h-8 text-[11px]" />
              <Button size="sm" onClick={onRetry} disabled={!retryReason.trim() || retryPending}>
                {retryPending ? <Loader2 className="animate-spin" /> : <RotateCcw />} Retry work
              </Button>
              <Button size="sm" variant="ghost" asChild>
                <a href="/diagnostics">Create support bundle</a>
              </Button>
            </div>
          ) : null}
          {cancellable ? (
            <Button variant="ghost" size="sm" onClick={onCancel} disabled={cancelPending}>
              {cancelPending ? <Loader2 className="animate-spin" /> : <Ban />} Cancel work
            </Button>
          ) : null}
          {actionError ? <p className="mt-2 text-[10.5px] text-danger">{actionError}</p> : null}
        </InspectorSection>
      ) : null}

      <InspectorSection title="Identity">
        <KeyValue label="Work item" value={item.work_item_id ?? item.id} mono />
        {item.domain_type ? <KeyValue label={item.domain_type.replace(/_/g, " ")} value={item.domain_id ?? "–"} mono /> : null}
        {item.worker_id ? <KeyValue label="Worker" value={item.worker_id} mono /> : null}
      </InspectorSection>
    </div>
  );
}

function InspectorSection({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="border-b border-border-subtle px-4 py-3"><h4 className="mb-2 text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">{title}</h4>{children}</section>;
}

function KeyValue({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return <div className="flex items-start justify-between gap-4 border-b border-border-subtle/70 py-1.5 last:border-0"><span className="text-[10.5px] capitalize text-fg-subtle">{label}</span><span className={cn("max-w-[65%] break-all text-right text-[10.5px] text-fg-muted", mono && "font-mono")}>{value}</span></div>;
}

function PanelMessage({ icon: Icon, label, detail, spin, tone }: { icon: typeof Activity; label: string; detail?: string; spin?: boolean; tone?: "danger" }) {
  return <div className="grid min-h-40 place-items-center px-5 py-8 text-center"><div><Icon className={cn("mx-auto h-4 w-4 text-fg-disabled", spin && "animate-spin", tone === "danger" && "text-danger")} /><p className={cn("mt-2 text-[11.5px] text-fg-muted", tone === "danger" && "text-danger")}>{label}</p>{detail ? <p className="mt-1 text-[10px] text-fg-disabled">{detail}</p> : null}</div></div>;
}

function StatusGlyph({ status }: { status: string }) {
  if (["running", "preparing"].includes(status)) return <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />;
  if (["completed", "succeeded", "pass"].includes(status)) return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success" />;
  if (["failed", "interrupted", "needs_reconciliation"].includes(status)) return <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-danger" />;
  if (["blocked", "awaiting_review", "waiting_for_accelerator"].includes(status)) return <Clock3 className="h-3.5 w-3.5 shrink-0 text-warning" />;
  return <CircleDashed className="h-3.5 w-3.5 shrink-0 text-fg-disabled" />;
}

function StatusBadge({ status }: { status: string }) {
  const tone = ["completed", "succeeded", "pass"].includes(status) ? "success" : ["failed", "interrupted", "needs_reconciliation"].includes(status) ? "danger" : ["blocked", "awaiting_review", "waiting_for_accelerator"].includes(status) ? "warning" : ["running", "preparing"].includes(status) ? "accent" : "neutral";
  return <Badge tone={tone} size="sm" dot>{status.replace(/_/g, " ")}</Badge>;
}

function activityProgress(item: ActivityItem): number | null {
  if (typeof item.progress_percent === "number") return Math.max(0, Math.min(100, item.progress_percent));
  if (typeof item.progress_current === "number" && typeof item.progress_total === "number" && item.progress_total > 0) return Math.max(0, Math.min(100, item.progress_current / item.progress_total * 100));
  return null;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(seconds >= 36_000 ? 0 : 1)}h`;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "Unavailable";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function formatPercent(value?: number | null): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Unavailable";
}
