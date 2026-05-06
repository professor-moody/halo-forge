import { useQuery } from "@tanstack/react-query";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Filter,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, type RunSample, type RunSamples } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardEyebrow,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

/**
 * Sample inspector — the operator's "what is the model actually doing"
 * surface. RAFT-specific: scrubs through generated samples for a cycle
 * with their verifier scores.
 *
 * Anatomy:
 *   ┌─────────┬───────────────────────────────────────────┐
 *   │ Cycle ▾ │ kind: samples / accepted    filter: ↑ pass │
 *   ├─────────┴───────────────────────────────────────────┤
 *   │ Reward histogram (small)                            │
 *   ├─────────┬───────────────────────────────────────────┤
 *   │ List    │ Detail: full prompt + completion          │
 *   │  ▸ #0   │   reward: 0.62  success: ✓                │
 *   │  ▸ #1   │   verifier details: { compile: true, ... }│
 *   │  …      │                                           │
 *   └─────────┴───────────────────────────────────────────┘
 *
 * The list is virtual-feeling but capped at 50 items per request — most
 * runs sample in the hundreds, not thousands, so we don't need full
 * windowing yet.
 */

export interface SampleInspectorProps {
  runId: string;
  /** Cycles available for this run; passed in so we can render the
   *  selector before the first samples request returns. */
  availableCycles: number[];
  /** Hide entirely for SFT runs (no per-cycle samples). */
  enabled?: boolean;
}

type SampleKind = "samples" | "accepted";
type SampleFilter = "all" | "passed" | "failed";

export function SampleInspector({
  runId,
  availableCycles,
  enabled = true,
}: SampleInspectorProps) {
  const defaultCycle = availableCycles.length
    ? availableCycles[availableCycles.length - 1]
    : 0;
  const [cycle, setCycle] = useState<number>(defaultCycle);
  const [kind, setKind] = useState<SampleKind>("samples");
  const [filter, setFilter] = useState<SampleFilter>("all");
  const [selectedIdx, setSelectedIdx] = useState(0);

  // Keep `cycle` in sync if the available list arrives later.
  useEffect(() => {
    if (availableCycles.length && !availableCycles.includes(cycle)) {
      setCycle(availableCycles[availableCycles.length - 1]);
      setSelectedIdx(0);
    }
  }, [availableCycles, cycle]);

  const { data, isLoading, isError, error } = useQuery<RunSamples>({
    queryKey: ["run-samples", runId, cycle, kind],
    queryFn: () => api.runSamples(runId, { cycle, kind, limit: 100 }),
    enabled: enabled && availableCycles.length > 0,
  });

  const samples = useMemo<RunSample[]>(() => {
    const all = data?.samples ?? [];
    if (filter === "all") return all;
    return all.filter((s) => Boolean(s.success) === (filter === "passed"));
  }, [data, filter]);

  // Reset selection when the underlying list changes
  useEffect(() => {
    setSelectedIdx(0);
  }, [cycle, kind, filter, data?.samples.length]);

  const selected = samples[selectedIdx];

  if (!enabled) return null;

  return (
    <Card>
      <CardHeader className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2">
          <CardEyebrow>VERIFIER</CardEyebrow>
          <CardTitle>Samples</CardTitle>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 ml-auto">
          {/* Cycle selector */}
          <CycleSelector
            cycle={cycle}
            available={availableCycles}
            onChange={setCycle}
          />
          {/* kind toggle */}
          <KindToggle value={kind} onChange={setKind} />
          {/* filter toggle */}
          <FilterToggle value={filter} onChange={setFilter} />
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {isLoading ? (
          <div className="px-4 py-10 text-center text-xs text-fg-subtle">
            Loading cycle {cycle} samples…
          </div>
        ) : isError ? (
          <div className="px-4 py-10 text-center text-xs text-danger">
            {(error as Error)?.message ?? "Could not load samples."}
          </div>
        ) : !data?.available ? (
          <div className="px-4 py-10 flex flex-col items-center gap-1.5 text-center">
            <div className="text-[12px] text-fg">No sample artifacts on disk.</div>
            <div className="text-[11px] text-fg-subtle max-w-[44ch]">
              {data?.reason ??
                "Samples appear here when the trainer writes cycle_N_samples.jsonl."}
            </div>
          </div>
        ) : samples.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-fg-subtle">
            {filter === "all"
              ? "No samples in this cycle."
              : `No ${filter} samples in this cycle.`}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] divide-x divide-border-subtle">
            {/* Distribution mini-histogram + sample list */}
            <div className="flex flex-col">
              <RewardSparkline samples={data.samples} />
              <div className="overflow-y-auto" style={{ maxHeight: 360 }}>
                {samples.map((s, i) => {
                  const isSelected = i === selectedIdx;
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setSelectedIdx(i)}
                      className={cn(
                        "w-full text-left flex items-center justify-between gap-2 px-3.5 py-2 border-b border-border-subtle hover:bg-surface-hover/40 transition-colors",
                        isSelected && "bg-accent-bg/40",
                      )}
                    >
                      <span className="flex items-center gap-2 min-w-0">
                        {Boolean(s.success) ? (
                          <CheckCircle2 className="h-3 w-3 text-success shrink-0" />
                        ) : (
                          <XCircle className="h-3 w-3 text-danger shrink-0" />
                        )}
                        <span
                          className={cn(
                            "font-mono text-[11px] truncate",
                            isSelected ? "text-accent" : "text-fg-muted",
                          )}
                        >
                          #{i.toString().padStart(2, "0")}
                        </span>
                        <span className="text-[11.5px] text-fg-muted truncate max-w-[14ch]">
                          {previewLine(s)}
                        </span>
                      </span>
                      <RewardChip reward={s.reward} />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Detail */}
            <div className="overflow-y-auto" style={{ maxHeight: 360 + 56 }}>
              {selected ? (
                <SampleDetail sample={selected} cycle={cycle} index={selectedIdx} />
              ) : null}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ---------------------------------------------------------------------- */

function CycleSelector({
  cycle,
  available,
  onChange,
}: {
  cycle: number;
  available: number[];
  onChange: (n: number) => void;
}) {
  if (available.length === 0) return null;
  const idx = available.indexOf(cycle);
  const prevDisabled = idx <= 0;
  const nextDisabled = idx === -1 || idx >= available.length - 1;

  return (
    <div className="flex items-center gap-1 rounded-md border border-border bg-bg-subtle pl-1.5 pr-1 h-8">
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 -mx-0.5"
        disabled={prevDisabled}
        onClick={() => onChange(available[idx - 1])}
        aria-label="Previous cycle"
      >
        <ChevronLeft className="h-3 w-3" />
      </Button>
      <Select value={String(cycle)} onValueChange={(v) => onChange(Number(v))}>
        <SelectTrigger className="border-0 bg-transparent shadow-none h-6 px-1.5 text-[11px] focus-visible:ring-0">
          <SelectValue placeholder="Cycle" />
        </SelectTrigger>
        <SelectContent>
          {available.map((c) => (
            <SelectItem key={c} value={String(c)}>
              Cycle {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 -mx-0.5"
        disabled={nextDisabled}
        onClick={() => onChange(available[idx + 1])}
        aria-label="Next cycle"
      >
        <ChevronRight className="h-3 w-3" />
      </Button>
    </div>
  );
}

function KindToggle({
  value,
  onChange,
}: {
  value: SampleKind;
  onChange: (v: SampleKind) => void;
}) {
  return (
    <div className="inline-flex items-center rounded-md border border-border bg-bg-subtle p-0.5 h-8">
      {(["samples", "accepted"] as const).map((k) => (
        <button
          key={k}
          type="button"
          onClick={() => onChange(k)}
          className={cn(
            "px-2 py-0.5 rounded-sm text-[11px] font-medium transition-colors",
            value === k
              ? "bg-surface-hover text-fg"
              : "text-fg-muted hover:text-fg",
          )}
        >
          {k}
        </button>
      ))}
    </div>
  );
}

function FilterToggle({
  value,
  onChange,
}: {
  value: SampleFilter;
  onChange: (v: SampleFilter) => void;
}) {
  const next: Record<SampleFilter, SampleFilter> = {
    all: "passed",
    passed: "failed",
    failed: "all",
  };
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => onChange(next[value])}
      className="text-[11px] h-8"
    >
      <Filter className="h-3 w-3" />
      {value}
    </Button>
  );
}

function RewardChip({ reward }: { reward?: number }) {
  if (typeof reward !== "number") return null;
  const tone =
    reward >= 0.7 ? "success" : reward >= 0.4 ? "warning" : "danger";
  return (
    <Badge tone={tone} size="sm">
      {reward.toFixed(2)}
    </Badge>
  );
}

function RewardSparkline({ samples }: { samples: RunSample[] }) {
  // 5-bucket histogram of the reward distribution. Render as horizontal
  // tick marks above the list so the operator can see "what's the spread"
  // without flipping to a chart.
  const buckets = [0, 0, 0, 0, 0];
  let max = 0;
  for (const s of samples) {
    const r = typeof s.reward === "number" ? s.reward : null;
    if (r === null) continue;
    const idx = Math.min(4, Math.max(0, Math.floor(r * 5)));
    buckets[idx]++;
    if (buckets[idx] > max) max = buckets[idx];
  }
  if (max === 0) return null;
  return (
    <div className="border-b border-border-subtle px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wider text-fg-disabled font-medium pb-1.5">
        Reward distribution
      </div>
      <div className="flex items-end gap-1 h-8">
        {buckets.map((n, i) => {
          const tone =
            i >= 4 ? "success" : i >= 2 ? "warning" : i >= 1 ? "danger" : "muted";
          const bg =
            tone === "success"
              ? "bg-success"
              : tone === "warning"
                ? "bg-warning"
                : tone === "danger"
                  ? "bg-danger"
                  : "bg-fg-disabled";
          return (
            <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
              <div
                className={cn("w-full rounded-sm transition-all", bg)}
                style={{ height: `${(n / max) * 100}%` }}
                title={`${(i / 5).toFixed(1)}–${((i + 1) / 5).toFixed(1)}: ${n}`}
              />
              <span className="font-mono text-[9px] text-fg-disabled">
                {(i / 5).toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SampleDetail({
  sample,
  cycle,
  index,
}: {
  sample: RunSample;
  cycle: number;
  index: number;
}) {
  const promptText = sample.prompt ?? extractMessageContent(sample, "user") ?? "";
  const completionText =
    sample.completion ?? extractMessageContent(sample, "assistant") ?? "";

  return (
    <div className="flex flex-col gap-3 p-4 text-[12px]">
      <div className="flex items-center gap-2 text-fg-muted">
        <span className="font-mono text-[11px]">cycle {cycle} · #{index}</span>
        <RewardChip reward={sample.reward} />
        {sample.success !== undefined ? (
          <Badge tone={sample.success ? "success" : "danger"} dot size="sm">
            {sample.success ? "passed" : "failed"}
          </Badge>
        ) : null}
      </div>

      {promptText ? (
        <Section label="Prompt">
          <pre className="font-mono text-[11.5px] whitespace-pre-wrap break-words text-fg-muted">
            {promptText}
          </pre>
        </Section>
      ) : null}

      {completionText ? (
        <Section label="Completion">
          <pre className="font-mono text-[11.5px] whitespace-pre-wrap break-words text-fg">
            {completionText}
          </pre>
        </Section>
      ) : null}

      {sample.details && Object.keys(sample.details).length ? (
        <Section label="Verifier details">
          <pre className="font-mono text-[11px] whitespace-pre-wrap break-words text-fg-muted">
            {JSON.stringify(sample.details, null, 2)}
          </pre>
        </Section>
      ) : null}
    </div>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled font-medium">
        {label}
      </div>
      <div className="rounded-md border border-border-subtle bg-bg-subtle/40 p-3">
        {children}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- */

function previewLine(sample: RunSample): string {
  const text =
    sample.prompt ??
    sample.completion ??
    extractMessageContent(sample, "user") ??
    extractMessageContent(sample, "assistant") ??
    "";
  return text.replace(/\s+/g, " ").trim().slice(0, 36);
}

function extractMessageContent(sample: RunSample, role: string): string | undefined {
  if (!Array.isArray(sample.messages)) return undefined;
  const m = sample.messages.find((m) => m.role === role);
  return m?.content;
}
