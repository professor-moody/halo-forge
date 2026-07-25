import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  BarChart3,
  CircleDashed,
  Download,
  FileCheck2,
  GitFork,
  Loader2,
  Play,
  Scale,
  ShieldCheck,
  Square,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  api,
  type CheckpointGateDecision,
  type CheckpointTrajectoryPoint,
  type CohortAnalysisSnapshot,
  type ResearchDecisionRecord,
  type RunGroup,
  type RunGroupTrajectory,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type WorkspaceTab = "trajectory" | "evidence" | "decision";

export function AdaptiveExperimentWorkspace({ group }: { group: RunGroup }) {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<WorkspaceTab>("trajectory");
  const [selectedPointId, setSelectedPointId] = useState("");
  const [selectedAnalysisId, setSelectedAnalysisId] = useState("");
  const [reviewReason, setReviewReason] = useState("");
  const [rationale, setRationale] = useState("");
  const [decision, setDecision] = useState<ResearchDecisionRecord | null>(null);
  const [bundleMessage, setBundleMessage] = useState<string | null>(null);

  const trajectory = useQuery({
    queryKey: ["run-groups", group.id, "trajectory"],
    queryFn: () => api.runGroupTrajectory(group.id),
    refetchInterval: ["queued", "running", "awaiting_review"].includes(group.status) ? 4_000 : false,
    retry: false,
  });
  const analyses = useQuery({
    queryKey: ["run-groups", group.id, "analyses"],
    queryFn: () => api.listRunGroupAnalyses(group.id, { limit: 50 }),
    refetchInterval: 5_000,
    retry: false,
  });

  useEffect(() => {
    if (selectedPointId || !trajectory.data?.points.length) return;
    const actionable = trajectory.data.points.find((point) => {
      const gate = trajectory.data?.gate_decisions?.find((item) => item.id === point.gate_decision_id);
      return point.gate_action === "await_review" || point.gate_action === "stop" || gate?.status === "awaiting_review" || gate?.status === "stopped";
    });
    setSelectedPointId(actionable?.id ?? trajectory.data.points.at(-1)?.id ?? "");
  }, [selectedPointId, trajectory.data]);
  useEffect(() => {
    if (selectedAnalysisId || !analyses.data?.items.length) return;
    setSelectedAnalysisId(analyses.data.items[0].id);
  }, [analyses.data?.items, selectedAnalysisId]);

  const review = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "continue" | "stop" }) => api.reviewGateDecision(id, { action, reason: reviewReason.trim() }),
    onSuccess: () => {
      setReviewReason("");
      queryClient.invalidateQueries({ queryKey: ["run-groups", group.id] });
      queryClient.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  const analyze = useMutation({
    mutationFn: () => api.createRunGroupAnalysis(group.id, {
      confidence_level: 0.95,
      bootstrap_resamples: 10_000,
      bootstrap_seed: 42,
      replicate_unit: "seed",
      comparison: "matched_seeds",
    }),
    onSuccess: (created) => {
      setSelectedAnalysisId(created.id);
      setTab("evidence");
      queryClient.invalidateQueries({ queryKey: ["run-groups", group.id, "analyses"] });
    },
  });
  const selectedAnalysis = analyses.data?.items.find((item) => item.id === selectedAnalysisId) ?? analyses.data?.items[0] ?? null;
  const decide = useMutation({
    mutationFn: () => api.createResearchDecision({
      analysis_snapshot_id: selectedAnalysis?.id ?? "",
      selected_subject: { trial_id: group.best_trial_id, run_group_id: group.id },
      rejected_subjects: (group.trials ?? []).filter((trial) => trial.id !== group.best_trial_id).map((trial) => ({ trial_id: trial.id })),
      rationale: rationale.trim(),
      fork_spec: group.best_trial_id ? { run_group_id: group.id, trial_id: group.best_trial_id } : undefined,
    }),
    onSuccess: setDecision,
  });
  const bundle = useMutation({
    mutationFn: () => api.createEvidenceBundle({ analysis_snapshot_id: selectedAnalysis?.id ?? "", research_decision_id: decision?.id, formats: ["markdown", "html", "json", "csv", "svg"] }),
    onSuccess: (created) => setBundleMessage(created.work_item_id ? `Evidence bundle queued · ${shortId(created.work_item_id)}` : `Evidence bundle ${created.status}`),
  });

  const awaiting = trajectory.data?.gate_decisions?.filter((gate) => gate.action === "await_review" || gate.status === "awaiting_review").length ?? group.awaiting_review_count ?? 0;

  return (
    <section className="border-b border-border-subtle">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border-subtle px-5 py-2.5">
        <nav aria-label="Research evidence" className="flex min-w-0 overflow-x-auto">
          <WorkspaceTabButton active={tab === "trajectory"} onClick={() => setTab("trajectory")} icon={GitFork} label="Checkpoint trajectory" count={trajectory.data?.points.length} />
          <WorkspaceTabButton active={tab === "evidence"} onClick={() => setTab("evidence")} icon={BarChart3} label="Evidence" count={analyses.data?.items.length} />
          <WorkspaceTabButton active={tab === "decision"} onClick={() => setTab("decision")} icon={FileCheck2} label="Decision" />
        </nav>
        <div className="flex items-center gap-2">
          {awaiting ? <Badge tone="warning" size="sm" dot>{awaiting} awaiting review</Badge> : null}
          {group.checkpoint_policy_revision_id ? <Badge tone="neutral" size="sm">adaptive policy</Badge> : <Badge tone="neutral" size="sm">final only</Badge>}
        </div>
      </div>

      {tab === "trajectory" ? (
        <TrajectoryWorkspace
          data={trajectory.data ?? null}
          loading={trajectory.isLoading}
          unavailable={trajectory.isError}
          selectedPointId={selectedPointId}
          onSelectPoint={setSelectedPointId}
          reviewReason={reviewReason}
          onReviewReason={setReviewReason}
          onReview={(gateId, action) => review.mutate({ id: gateId, action })}
          reviewPending={review.isPending}
          reviewError={review.error instanceof Error ? review.error.message : null}
        />
      ) : null}

      {tab === "evidence" ? (
        <EvidenceWorkspace
          items={analyses.data?.items ?? []}
          selectedId={selectedAnalysisId}
          onSelect={setSelectedAnalysisId}
          loading={analyses.isLoading}
          unavailable={analyses.isError}
          onAnalyze={() => analyze.mutate()}
          analyzing={analyze.isPending}
          error={analyze.error instanceof Error ? analyze.error.message : null}
        />
      ) : null}

      {tab === "decision" ? (
        <DecisionWorkspace
          group={group}
          analysis={selectedAnalysis}
          rationale={rationale}
          onRationale={setRationale}
          decision={decision}
          onDecide={() => decide.mutate()}
          deciding={decide.isPending}
          decisionError={decide.error instanceof Error ? decide.error.message : null}
          onBundle={() => bundle.mutate()}
          bundling={bundle.isPending}
          bundleMessage={bundleMessage}
        />
      ) : null}
    </section>
  );
}

export function RunCheckpointTrajectory({ runGroupId, runId }: { runGroupId: string; runId: string }) {
  const trajectory = useQuery({ queryKey: ["run-groups", runGroupId, "trajectory"], queryFn: () => api.runGroupTrajectory(runGroupId), retry: false, refetchInterval: 5_000 });
  const points = trajectory.data?.points.filter((point) => !point.run_id || point.run_id === runId) ?? [];
  if (trajectory.isError || (!trajectory.isLoading && !points.length)) return null;
  return (
    <section className="border border-border-subtle bg-surface/35">
      <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3"><div><div className="text-[9.5px] uppercase tracking-[0.13em] text-fg-disabled">Adaptive training</div><h3 className="mt-1 text-[13px] font-medium text-fg">Checkpoint trajectory</h3></div><Badge tone={points.some((point) => point.gate_action === "await_review") ? "warning" : "neutral"} size="sm">{points.length} boundaries</Badge></div>
      <div className="overflow-x-auto px-4 py-4"><CompactTrajectory points={points} /></div>
    </section>
  );
}

function WorkspaceTabButton({ active, onClick, icon: Icon, label, count }: { active: boolean; onClick: () => void; icon: typeof GitFork; label: string; count?: number }) {
  return <button type="button" onClick={onClick} className={cn("relative flex h-8 shrink-0 items-center gap-1.5 px-2.5 text-[11px] transition-colors", active ? "text-fg" : "text-fg-subtle hover:text-fg")}><Icon className={cn("h-3.5 w-3.5", active && "text-accent")} />{label}{count != null ? <span className="font-mono text-[9px] text-fg-disabled">{count}</span> : null}{active ? <span className="absolute inset-x-2 bottom-0 h-0.5 bg-accent" /> : null}</button>;
}

function TrajectoryWorkspace({ data, loading, unavailable, selectedPointId, onSelectPoint, reviewReason, onReviewReason, onReview, reviewPending, reviewError }: {
  data: RunGroupTrajectory | null; loading: boolean; unavailable: boolean; selectedPointId: string; onSelectPoint: (id: string) => void; reviewReason: string; onReviewReason: (value: string) => void; onReview: (gateId: string, action: "continue" | "stop") => void; reviewPending: boolean; reviewError: string | null;
}) {
  if (loading) return <WorkspaceMessage icon={Loader2} spin title="Loading checkpoint evidence" detail="Published checkpoints and gate decisions will appear here." />;
  if (unavailable || !data) return <WorkspaceMessage icon={CircleDashed} title="Final-only experiment" detail="This group has no adaptive trajectory. New groups can add a periodic or guarded checkpoint policy." />;
  const points = data.points ?? [];
  if (!points.length) return <WorkspaceMessage icon={GitFork} title="Waiting for the first boundary" detail="The resolved plan is pinned; checkpoints appear after verified publication." />;
  const selected = points.find((point) => point.id === selectedPointId) ?? points.at(-1)!;
  const gate = data.gate_decisions?.find((item) => item.id === selected.gate_decision_id);
  return (
    <div className="grid min-h-[300px] lg:grid-cols-[minmax(0,1fr)_260px]">
      <div className="min-w-0 border-b border-border-subtle p-5 lg:border-b-0 lg:border-r">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><h3 className="text-[13px] font-medium text-fg">Verified boundaries</h3><p className="mt-1 text-[10.5px] text-fg-subtle">Each point links a published checkpoint, evaluation evidence, and the continuation decision.</p></div>{data.resolved_plan ? <div className="text-right font-mono text-[9.5px] text-fg-disabled"><div>{data.resolved_plan.boundaries.length} planned · {data.resolved_plan.unit}</div><div>{data.resolved_plan.automatic_actions ? "automatic actions on" : "reviewed actions"}</div></div> : null}</div>
        <div className="overflow-x-auto"><TrajectoryRows points={points} selectedId={selected.id} onSelect={onSelectPoint} /></div>
      </div>
      <aside className="bg-bg-subtle/20 p-4">
        <PointInspector point={selected} gate={gate} />
        {gate ? <GateReviewControls gate={gate} reason={reviewReason} onReason={onReviewReason} onReview={onReview} pending={reviewPending} error={reviewError} /> : null}
      </aside>
    </div>
  );
}

function GateReviewControls({ gate, reason, onReason, onReview, pending, error }: { gate: CheckpointGateDecision; reason: string; onReason: (value: string) => void; onReview: (gateId: string, action: "continue" | "stop") => void; pending: boolean; error: string | null }) {
  const stopped = gate.action === "stop" || gate.status === "stopped";
  const paused = gate.action === "await_review" || gate.status === "awaiting_review";
  if (!stopped && !paused) return null;
  return <div className="mt-4 border-t border-border-subtle pt-4"><div className={cn("text-[9.5px] font-medium uppercase tracking-wider", stopped ? "text-danger" : "text-warning")}>{stopped ? "Stopped boundary" : "Operator review"}</div><p className="mt-2 text-[10.5px] leading-relaxed text-fg-subtle">{stopped ? "This branch is terminal. Continuing requires an explicit operator override with an append-only reason." : "Evidence is incomplete or the policy requested a pause. Record the reason before continuing or stopping."}</p><Input className="mt-3 h-8 text-[11px]" value={reason} onChange={(event) => onReason(event.target.value)} placeholder={stopped ? "Required override reason" : "Review reason"} /><div className="mt-2 flex gap-2"><Button size="sm" onClick={() => onReview(gate.id, "continue")} disabled={!reason.trim() || pending}>{pending ? <Loader2 className="animate-spin" /> : <Play />} {stopped ? "Override & continue" : "Continue"}</Button>{!stopped ? <Button size="sm" variant="ghost" onClick={() => onReview(gate.id, "stop")} disabled={!reason.trim() || pending}><Square /> Stop</Button> : null}</div>{error ? <p className="mt-2 text-[10px] text-danger">{error}</p> : null}</div>;
}

function TrajectoryRows({ points, selectedId, onSelect }: { points: CheckpointTrajectoryPoint[]; selectedId: string; onSelect: (id: string) => void }) {
  const rows = useMemo(() => {
    const groups = new Map<string, CheckpointTrajectoryPoint[]>();
    points.forEach((point) => { const key = point.run_id ?? `seed-${point.seed ?? "–"}`; groups.set(key, [...(groups.get(key) ?? []), point].sort((a, b) => a.boundary_index - b.boundary_index)); });
    return [...groups.entries()];
  }, [points]);
  return <div className="min-w-[340px] divide-y divide-border-subtle border-y border-border-subtle">{rows.map(([key, row]) => <div key={key} className="grid grid-cols-[80px_minmax(0,1fr)] items-center py-3"><div className="pr-3"><div className="truncate font-mono text-[9.5px] text-fg-muted">{shortId(key)}</div><div className="mt-0.5 text-[9px] text-fg-disabled">seed {row[0]?.seed ?? "–"}</div></div><div className="relative flex items-center justify-between gap-2 before:absolute before:inset-x-2 before:top-1/2 before:h-px before:bg-border-strong">{row.map((point) => <TrajectoryPoint key={point.id} point={point} selected={point.id === selectedId} onSelect={() => onSelect(point.id)} />)}</div></div>)}</div>;
}

function CompactTrajectory({ points }: { points: CheckpointTrajectoryPoint[] }) {
  return <div className="relative flex min-w-[340px] items-center justify-between gap-4 before:absolute before:inset-x-2 before:top-2 before:h-px before:bg-border-strong">{points.map((point) => <div key={point.id} className="relative text-center"><span className={cn("mx-auto block h-4 w-4 rounded-full border-2 border-bg", pointTone(point))} /><div className="mt-2 font-mono text-[9px] text-fg-subtle">{point.boundary_value} {point.boundary_unit}</div><div className="mt-0.5 font-mono text-[9px] text-fg-disabled">{point.metric_value == null ? "—" : formatMetric(point.metric_value)}</div></div>)}</div>;
}

function TrajectoryPoint({ point, selected, onSelect }: { point: CheckpointTrajectoryPoint; selected: boolean; onSelect: () => void }) {
  return <button type="button" onClick={onSelect} className="relative z-10 min-w-12 text-center"><span className={cn("mx-auto block h-4 w-4 rounded-full border-2 border-bg transition-transform motion-reduce:transition-none", pointTone(point), selected && "scale-125 ring-2 ring-accent/30")} /><span className={cn("mt-1.5 block font-mono text-[9px]", selected ? "text-accent" : "text-fg-subtle")}>{point.boundary_value}</span><span className="block font-mono text-[8.5px] text-fg-disabled">{point.metric_value == null ? "—" : formatMetric(point.metric_value)}</span></button>;
}

function PointInspector({ point, gate }: { point: CheckpointTrajectoryPoint; gate?: CheckpointGateDecision }) {
  return <div><div className="flex items-center gap-2"><span className={cn("h-2 w-2 rounded-full", pointTone(point))} /><span className="text-[9.5px] uppercase tracking-wider text-fg-disabled">Boundary {point.boundary_index + 1}</span></div><h3 className="mt-2 text-[15px] font-medium text-fg">{point.boundary_value} {point.boundary_unit}</h3><dl className="mt-3 divide-y divide-border-subtle"><InspectorValue label="Status" value={point.status} /><InspectorValue label="Primary metric" value={point.metric_value == null ? "Not measured" : formatMetric(point.metric_value)} mono /><InspectorValue label="Gate" value={gate?.action ?? point.gate_action ?? "pending"} /><InspectorValue label="Checkpoint" value={point.checkpoint_artifact_id ? shortId(point.checkpoint_artifact_id) : "Not published"} mono /></dl>{(gate?.reasons ?? (point.reason ? [point.reason] : [])).length ? <div className="mt-3 text-[10.5px] leading-relaxed text-fg-subtle">{(gate?.reasons ?? [point.reason!]).join(" · ")}</div> : null}</div>;
}

function EvidenceWorkspace({
  items,
  selectedId,
  onSelect,
  loading,
  unavailable,
  onAnalyze,
  analyzing,
  error,
}: {
  items: CohortAnalysisSnapshot[];
  selectedId: string;
  onSelect: (id: string) => void;
  loading: boolean;
  unavailable: boolean;
  onAnalyze: () => void;
  analyzing: boolean;
  error: string | null;
}) {
  const selected = items.find((item) => item.id === selectedId) ?? items[0];
  return (
    <div className="grid min-h-[320px] lg:grid-cols-[190px_minmax(0,1fr)] 2xl:grid-cols-[220px_minmax(0,1fr)_240px]">
      <aside className="border-b border-border-subtle lg:border-b-0 lg:border-r">
        <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
          <span className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
            Analysis snapshots
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={onAnalyze}
            disabled={analyzing}
          >
            {analyzing ? <Loader2 className="animate-spin" /> : <BarChart3 />}{" "}
            Analyze
          </Button>
        </div>
        <div className="divide-y divide-border-subtle">
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              className={cn(
                "w-full px-4 py-3 text-left",
                item.id === selected?.id
                  ? "bg-accent-bg/50"
                  : "hover:bg-surface/45",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-mono text-[9.5px] text-fg-muted">
                  {shortId(item.id)}
                </span>
                <AnalysisBadge value={classification(item)} />
              </div>
              <div className="mt-1 text-[9px] text-fg-disabled">
                {item.completed_at
                  ? new Date(item.completed_at).toLocaleString()
                  : item.status}
              </div>
            </button>
          ))}
          {loading ? (
            <WorkspaceMessage icon={Loader2} spin title="Loading analyses" />
          ) : null}
          {!loading && !items.length ? (
            <div className="px-4 py-8 text-center text-[10.5px] text-fg-disabled">
              No snapshot yet. Complete required evaluations, then analyze the
              cohort.
            </div>
          ) : null}
        </div>
      </aside>
      <main className="min-w-0 border-b border-border-subtle p-5 lg:border-b-0 lg:border-r">
        {selected ? (
          <AnalysisDetail analysis={selected} />
        ) : (
          <WorkspaceMessage
            icon={unavailable ? AlertTriangle : Scale}
            title={
              unavailable
                ? "Analysis service unavailable"
                : "No cohort evidence yet"
            }
            detail="Snapshots use seeds as the experimental replicates and preserve the exact comparison contract."
          />
        )}
        {error ? <p className="mt-2 text-[10px] text-danger">{error}</p> : null}
      </main>
      <aside className="border-t border-border-subtle bg-bg-subtle/20 p-4 lg:col-span-2 2xl:col-span-1 2xl:border-t-0">
        <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
          Evidence contract
        </div>
        <ul className="mt-3 space-y-2.5 text-[10.5px] leading-relaxed text-fg-subtle">
          <li>Seed-level replicates</li>
          <li>95% percentile-bootstrap interval</li>
          <li>10,000 resamples · seed 42</li>
          <li>Matched suite and generation settings</li>
          <li>Missing metrics remain unavailable</li>
        </ul>
        {selected?.analysis.compatibility &&
        !selected.analysis.compatibility.compatible ? (
          <div className="mt-4 border-l-2 border-warning pl-3 text-[10.5px] text-warning">
            {selected.analysis.compatibility.reasons?.join(" · ") ||
              "Evidence is not compatible."}
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function CoreAnalysisDetail({
  analysis,
}: {
  analysis: CohortAnalysisSnapshot;
}) {
  const result = analysis.analysis;
  const comparison = primaryComparison(analysis);
  const interval =
    result.interval ?? comparison?.confidence_interval ?? undefined;
  const metric =
    result.primary_metric ?? String(analysis.request.metric ?? "—");
  const matchedSeeds =
    result.matched_seed_count ??
    result.compatibility?.matched_seed_count ??
    comparison?.matched_seed_count;
  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
            Primary conclusion
          </div>
          <h3 className="mt-1 text-[18px] font-medium capitalize text-fg">
            {classification(analysis).replaceAll("_", " ")}
          </h3>
        </div>
        <AnalysisBadge value={classification(analysis)} />
      </div>
      <div className="mt-5 grid grid-cols-2 gap-px border-y border-border-subtle bg-border-subtle sm:grid-cols-4">
        <EvidenceReadout label="Metric" value={metric} />
        <EvidenceReadout
          label="Matched seeds"
          value={String(matchedSeeds ?? "—")}
        />
        <EvidenceReadout
          label="Interval low"
          value={interval?.lower == null ? "—" : formatMetric(interval.lower)}
        />
        <EvidenceReadout
          label="Interval high"
          value={interval?.upper == null ? "—" : formatMetric(interval.upper)}
        />
      </div>
      <div className="mt-5">
        <div className="mb-2 flex justify-between text-[9.5px] text-fg-disabled">
          <span>Regression</span>
          <span>Practical equivalence</span>
          <span>Improvement</span>
        </div>
        <div className="relative h-2 bg-surface-pressed">
          <span className="absolute inset-y-0 left-1/2 w-px bg-fg-disabled" />
          <span className="absolute inset-y-0 left-[42%] w-[16%] bg-accent-bg" />
          {interval?.lower != null && interval?.upper != null ? (
            <span
              className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-accent"
              style={{
                left: `${intervalPosition(interval.lower)}%`,
                width: `${Math.max(2, intervalPosition(interval.upper) - intervalPosition(interval.lower))}%`,
              }}
            />
          ) : null}
        </div>
      </div>
      {comparison?.reason ? (
        <div className="mt-4 border-l-2 border-warning pl-3 text-[10.5px] text-warning">
          {comparison.reason.replaceAll("_", " ")}
        </div>
      ) : null}
      <p className="mt-5 text-[10.5px] leading-relaxed text-fg-subtle">
        Per-record deltas remain diagnostic evidence. The decision
        classification uses matched seed outcomes only.
      </p>
    </div>
  );
}

function AnalysisDetail({ analysis }: { analysis: CohortAnalysisSnapshot }) {
  return (
    <>
      <CoreAnalysisDetail analysis={analysis} />
      <ParetoEvidence analysis={analysis} />
    </>
  );
}

const TRADEOFF_DIMENSIONS = [
  {
    id: "quality",
    label: "Primary quality",
    aliases: ["primary_metric"],
    direction: "dynamic",
  },
  {
    id: "latency",
    label: "Latency",
    aliases: [
      "total_latency_ms",
      "latency_ms",
      "ttft_ms",
      "time_to_first_token_ms",
    ],
    direction: "minimize",
  },
  {
    id: "throughput",
    label: "Throughput",
    aliases: ["output_tokens_per_second", "tokens_per_second", "throughput"],
    direction: "maximize",
  },
  {
    id: "memory",
    label: "Memory",
    aliases: [
      "peak_device_memory_bytes",
      "peak_memory_bytes",
      "memory_bytes",
      "memory",
    ],
    direction: "minimize",
  },
  {
    id: "energy",
    label: "Energy",
    aliases: ["energy_joules", "energy", "power_watts"],
    direction: "minimize",
  },
  {
    id: "size",
    label: "Artifact size",
    aliases: ["artifact_size_bytes", "size_bytes", "artifact_size"],
    direction: "minimize",
  },
] as const;

function ParetoEvidence({ analysis }: { analysis: CohortAnalysisSnapshot }) {
  const rows = analysis.analysis.pareto ?? [];
  if (!rows.length)
    return (
      <div className="mt-5 border-t border-border-subtle pt-4">
        <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
          Pareto evidence
        </div>
        <p className="mt-2 text-[10.5px] leading-relaxed text-fg-subtle">
          Operational tradeoffs are unavailable for this snapshot. No latency,
          throughput, memory, energy, or artifact-size value is inferred.
        </p>
      </div>
    );
  const resolved = TRADEOFF_DIMENSIONS.map((dimension) => ({
    ...dimension,
    key:
      dimension.aliases.find((alias) =>
        rows.some((row) => typeof row[alias] === "number"),
      ) ?? dimension.aliases[0],
  }));
  const common = resolved.filter((dimension) =>
    rows.every((row) => typeof row[dimension.key] === "number"),
  );
  const comparable = common.length >= 2;
  const frontier = comparable
    ? paretoFront(rows, common, analysis.analysis.direction ?? "maximize")
    : new Set<string>();
  return (
    <div className="mt-5 border-t border-border-subtle pt-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
            Pareto evidence
          </div>
          <h4 className="mt-1 text-[12px] font-medium text-fg">
            Primary quality and operational tradeoffs
          </h4>
        </div>
        <span className="text-[9px] text-fg-disabled">
          secondary measures are constraints, not a weighted score
        </span>
      </div>
      <div className="mt-3 divide-y divide-border-subtle border-y border-border-subtle">
        {rows.map((row, index) => {
          const subject = String(row.subject_id ?? `subject-${index + 1}`);
          return (
            <div key={subject} className="py-3">
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className="truncate font-mono text-[9.5px] text-fg-muted">
                  {shortId(subject)}
                </span>
                {comparable ? (
                  <Badge
                    tone={frontier.has(subject) ? "success" : "neutral"}
                    size="sm"
                  >
                    {frontier.has(subject) ? "frontier" : "dominated"}
                  </Badge>
                ) : (
                  <span className="text-[9px] text-fg-disabled">incomplete tradeoff</span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-px bg-border-subtle sm:grid-cols-3">
                {resolved.map((dimension) => (
                  <div key={dimension.id} className="min-w-0 bg-bg px-2 py-2">
                    <div className={cn("truncate text-[8px] uppercase tracking-wider text-fg-disabled", dimension.id === "quality" && "text-accent")}>{dimension.label}</div>
                    <div className={cn("mt-1 truncate font-mono text-[9.5px]", dimension.id === "quality" ? "text-accent" : typeof row[dimension.key] === "number" ? "text-fg-subtle" : "text-fg-disabled")}>{formatTradeoff(row[dimension.key], dimension.key)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-[9.5px] leading-relaxed text-fg-disabled">
        {comparable
          ? `Frontier classification uses ${common.map((dimension) => dimension.label.toLowerCase()).join(" and ")}, the dimensions available for every subject.`
          : "Pareto classification is unavailable until primary quality and at least one operational dimension exist for every subject."}{" "}
        Missing cells remain explicitly unavailable.
      </p>
    </div>
  );
}

function DecisionWorkspace({
  group,
  analysis,
  rationale,
  onRationale,
  decision,
  onDecide,
  deciding,
  decisionError,
  onBundle,
  bundling,
  bundleMessage,
}: {
  group: RunGroup;
  analysis: CohortAnalysisSnapshot | null;
  rationale: string;
  onRationale: (value: string) => void;
  decision: ResearchDecisionRecord | null;
  onDecide: () => void;
  deciding: boolean;
  decisionError: string | null;
  onBundle: () => void;
  bundling: boolean;
  bundleMessage: string | null;
}) {
  const insufficient =
    !analysis ||
    classification(analysis) === "insufficient_evidence" ||
    analysis.status !== "completed";
  return (
    <div className="grid min-h-[320px] lg:grid-cols-[minmax(0,1fr)_300px]">
      <main className="border-b border-border-subtle p-5 lg:border-b-0 lg:border-r">
        <div className="max-w-2xl">
          <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
            Reviewed decision
          </div>
          <h3 className="mt-1 text-[16px] font-medium text-fg">
            Select evidence, explain the choice, then fork separately.
          </h3>
          <p className="mt-1 text-[10.5px] leading-relaxed text-fg-subtle">
            A decision record is append-only. It never promotes, retrains, or
            creates data automatically.
          </p>
          <div className="mt-5 divide-y divide-border-subtle border-y border-border-subtle">
            <InspectorValue
              label="Analysis"
              value={
                analysis ? shortId(analysis.id) : "Select or create an analysis"
              }
              mono
            />
            <InspectorValue
              label="Conclusion"
              value={
                analysis
                  ? classification(analysis).replaceAll("_", " ")
                  : "No evidence"
              }
            />
            <InspectorValue
              label="Selected trial"
              value={
                group.best_trial_id
                  ? shortId(group.best_trial_id)
                  : "No ranked trial"
              }
              mono
            />
          </div>
          <label className="mt-5 block">
            <span className="text-[9.5px] font-medium uppercase tracking-wider text-fg-disabled">
              Decision rationale
            </span>
            <textarea
              value={rationale}
              onChange={(event) => onRationale(event.target.value)}
              rows={4}
              placeholder="Why this checkpoint or configuration is the right next branch"
              className="mt-2 w-full resize-y rounded-md border border-border bg-surface px-3 py-2 text-[11.5px] leading-relaxed text-fg outline-none focus:border-accent"
            />
          </label>
          {insufficient ? (
            <div className="mt-3 flex gap-2 text-[10.5px] text-warning">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              The current evidence is incomplete or insufficient. Record an
              explicit override in the backend workflow before choosing a
              winner.
            </div>
          ) : null}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              onClick={onDecide}
              disabled={
                !rationale.trim() || !analysis || deciding || insufficient
              }
            >
              {deciding ? <Loader2 className="animate-spin" /> : <FileCheck2 />}{" "}
              Record decision
            </Button>
            {decision ? (
              <span className="inline-flex items-center gap-1.5 px-2 text-[10.5px] text-fg-subtle">
                <GitFork className="h-3.5 w-3.5" /> Fork context recorded
              </span>
            ) : null}
          </div>
          {decisionError ? (
            <p className="mt-2 text-[10px] text-danger">{decisionError}</p>
          ) : null}
        </div>
      </main>
      <aside className="bg-bg-subtle/20 p-4">
        <div className="text-[9.5px] uppercase tracking-wider text-fg-disabled">
          Reproducible report
        </div>
        <p className="mt-2 text-[10.5px] leading-relaxed text-fg-subtle">
          Bundle the exact data, suite, checkpoint lineage, hardware identity,
          plots, assumptions, and missing-evidence inventory.
        </p>
        <Button
          className="mt-4"
          size="sm"
          variant="secondary"
          onClick={onBundle}
          disabled={!analysis || !decision || bundling}
        >
          {bundling ? <Loader2 className="animate-spin" /> : <Download />}{" "}
          Export evidence
        </Button>
        {bundleMessage ? (
          <div className="mt-3 border-l-2 border-success pl-2 text-[10px] text-success">
            {bundleMessage}
          </div>
        ) : null}
        {decision ? (
          <div className="mt-5 border-t border-border-subtle pt-3">
            <div className="flex items-center gap-2 text-[10.5px] text-success">
              <ShieldCheck className="h-3.5 w-3.5" />
              Decision recorded
            </div>
            <div className="mt-1 font-mono text-[9px] text-fg-disabled">
              {shortId(decision.content_hash)}
            </div>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function WorkspaceMessage({ icon: Icon, title, detail, spin }: { icon: typeof GitFork; title: string; detail?: string; spin?: boolean }) { return <div className="grid min-h-48 place-items-center px-5 py-8 text-center"><div><Icon className={cn("mx-auto h-5 w-5 text-fg-disabled", spin && "animate-spin")} /><h3 className="mt-3 text-[12.5px] font-medium text-fg">{title}</h3>{detail ? <p className="mx-auto mt-1 max-w-sm text-[10.5px] leading-relaxed text-fg-subtle">{detail}</p> : null}</div></div>; }
function InspectorValue({ label, value, mono }: { label: string; value: string; mono?: boolean }) { return <div className="flex items-start justify-between gap-4 py-2"><dt className="text-[10px] text-fg-subtle">{label}</dt><dd className={cn("max-w-[65%] break-all text-right text-[10.5px] capitalize text-fg-muted", mono && "font-mono normal-case")}>{value}</dd></div>; }
function EvidenceReadout({ label, value }: { label: string; value: string }) { return <div className="bg-bg px-3 py-3"><div className="text-[8.5px] uppercase tracking-wider text-fg-disabled">{label}</div><div className="mt-1 truncate font-mono text-[11.5px] text-fg">{value}</div></div>; }
function AnalysisBadge({ value }: { value: string }) { const tone = value === "improved" || value === "practically_equivalent" ? "success" : value === "regressed" ? "danger" : value === "insufficient_evidence" ? "warning" : "neutral"; return <Badge tone={tone} size="sm">{value.replaceAll("_", " ")}</Badge>; }
function primaryComparison(item: CohortAnalysisSnapshot) { return Object.values(item.analysis.comparisons ?? {})[0]; }
function classification(item: CohortAnalysisSnapshot): string { return item.analysis.classification ?? primaryComparison(item)?.classification ?? (item.status === "completed" ? "inconclusive" : item.status); }
function pointTone(point: CheckpointTrajectoryPoint): string { if (point.gate_action === "await_review" || point.status === "awaiting_review") return "bg-warning"; if (point.gate_action === "stop" || point.status === "stopped") return "bg-danger"; if (point.status === "completed" || point.gate_action === "continue") return "bg-success"; if (point.status === "running") return "bg-accent animate-pulse motion-reduce:animate-none"; return "bg-fg-disabled"; }
function shortId(value: string): string { return value.length > 18 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value; }
function formatMetric(value: number): string { if (!Number.isFinite(value)) return "—"; if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2); return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, ""); }
function formatTradeoff(value: unknown, key: string): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "unavailable";
  if (key.includes("bytes") || key.includes("size") || key.includes("memory")) {
    const units = ["B", "KB", "MB", "GB", "TB"];
    let amount = value;
    let index = 0;
    while (Math.abs(amount) >= 1024 && index < units.length - 1) { amount /= 1024; index += 1; }
    return `${amount.toFixed(index ? 1 : 0)} ${units[index]}`;
  }
  if (key.includes("latency") || key.includes("ttft") || key.includes("time_to_first")) return `${formatMetric(value)} ms`;
  if (key.includes("tokens_per_second") || key === "throughput") return `${formatMetric(value)} tok/s`;
  if (key.includes("joule") || key === "energy") return `${formatMetric(value)} J`;
  if (key.includes("power") || key.includes("watt")) return `${formatMetric(value)} W`;
  return formatMetric(value);
}
function paretoFront(rows: Array<Record<string, unknown>>, dimensions: Array<{ key: string; direction: string }>, primaryDirection: string): Set<string> {
  const identity = (row: Record<string, unknown>, index: number) => String(row.subject_id ?? `subject-${index + 1}`);
  const favorable = (dimension: { key: string; direction: string }, value: number) => (dimension.direction === "dynamic" ? primaryDirection : dimension.direction) === "minimize" ? -value : value;
  const result = new Set<string>();
  rows.forEach((candidate, candidateIndex) => {
    const dominated = rows.some((other, otherIndex) => {
      if (candidateIndex === otherIndex) return false;
      const pairs = dimensions.map((dimension) => [favorable(dimension, Number(other[dimension.key])), favorable(dimension, Number(candidate[dimension.key]))]);
      return pairs.every(([otherValue, candidateValue]) => otherValue >= candidateValue) && pairs.some(([otherValue, candidateValue]) => otherValue > candidateValue);
    });
    if (!dominated) result.add(identity(candidate, candidateIndex));
  });
  return result;
}
function intervalPosition(value: number): number { return Math.max(2, Math.min(98, 50 + value * 100)); }
