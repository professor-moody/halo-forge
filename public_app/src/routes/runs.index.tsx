import { createFileRoute, Link } from "@tanstack/react-router";
import { Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useRunSearch } from "@/lib/hooks";
import { Topbar } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/runs/")({
  component: RunsListRoute,
});

/**
 * Runs list — Track F-G commit 3.
 *
 * Powered by `/runs/search` (DB-backed) so filtering, sorting, and
 * paging all happen server-side. The chip rail across the top is
 * driven by the `facets` payload the search endpoint returns: when
 * the index has 12 distinct modalities, the chip set has 12 chips.
 *
 * Filter state lives in component state for now; a follow-up will
 * push it into the URL so links are shareable. Selecting "passed"
 * narrows on `effectiveness_verdict`; the substring box matches
 * `model_name` (case-sensitive — matches the backend behavior).
 */

type StatusKey = "completed" | "running" | "failed" | "pending";
const STATUS_OPTIONS: StatusKey[] = ["completed", "running", "failed", "pending"];

/**
 * Canonical training kinds halo-forge can produce. Render them all so
 * users see what's available even before they've used a given trainer.
 * Order: training algorithms first (sft/raft/dpo/grpo/rm), then
 * modality-specific paths (vlm/audio/reasoning/agentic).
 *
 * Keep in sync with the trainer modules under halo_forge/. If a new
 * trainer is added, append it here too — chips with zero runs render
 * dim instead of disappearing.
 */
const CANONICAL_MODALITIES: ReadonlyArray<{ key: string; label: string; hint: string }> = [
  { key: "sft", label: "sft", hint: "Supervised fine-tune" },
  { key: "raft", label: "raft", hint: "RAFT (rejection sampling + SFT)" },
  { key: "dpo", label: "dpo", hint: "Direct Preference Optimization" },
  { key: "grpo", label: "grpo", hint: "Group Relative Policy Optimization" },
  { key: "rm", label: "rm", hint: "Reward model (Bradley-Terry)" },
  { key: "vlm", label: "vlm", hint: "Vision-language" },
  { key: "audio", label: "audio", hint: "Speech / audio" },
  { key: "reasoning", label: "reasoning", hint: "Reasoning / chain-of-thought" },
  { key: "agentic", label: "agentic", hint: "Agentic / tool-use" },
];

function RunsListRoute() {
  const [modalities, setModalities] = useState<string[]>([]);
  const [statuses, setStatuses] = useState<StatusKey[]>([]);
  const [model, setModel] = useState("");
  const [hasEval, setHasEval] = useState<boolean | undefined>(undefined);

  const params = useMemo(
    () => ({
      modality: modalities.length ? modalities : undefined,
      status: statuses.length ? statuses : undefined,
      model: model.trim() || undefined,
      hasEval,
      limit: 100,
      sortBy: "timestamp" as const,
      sortDir: "desc" as const,
    }),
    [modalities, statuses, model, hasEval],
  );

  const { data, isLoading, isFetching } = useRunSearch(params);
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const facets = data?.facets ?? { modalities: [], models: [] };

  const filtersActive =
    modalities.length || statuses.length || model.trim() || hasEval !== undefined;

  function clearAll() {
    setModalities([]);
    setStatuses([]);
    setModel("");
    setHasEval(undefined);
  }

  function toggle<T extends string>(set: T[], value: T): T[] {
    return set.includes(value) ? set.filter((s) => s !== value) : [...set, value];
  }

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Runs"
        subtitle={`${total} total · ${items.length} shown`}
        actions={
          filtersActive ? (
            <Button variant="ghost" size="sm" onClick={clearAll} title="Clear filters">
              <X />
              Clear filters
            </Button>
          ) : null
        }
      />
      <div className="px-6 py-5 space-y-3">
        {/* Filter rail */}
        <Card>
          <CardContent className="p-3 space-y-2.5">
            {/* Modality chips — canonical full set, with counts when
                populated. Kinds with zero runs render dim so the user
                sees what's available before they've tried it (otherwise
                only sft/raft would ever appear in a fresh install). */}
            <ChipRow label="Modality">
              {(() => {
                const counts = facets.modality_counts ?? {};
                const known = new Set(CANONICAL_MODALITIES.map((m) => m.key));
                const extras = facets.modalities.filter((m) => !known.has(m));
                const all = [
                  ...CANONICAL_MODALITIES,
                  ...extras.map((k) => ({ key: k, label: k, hint: "" })),
                ];
                return all.map(({ key, label, hint }) => {
                  const count = counts[key] ?? 0;
                  return (
                    <Chip
                      key={key}
                      active={modalities.includes(key)}
                      empty={count === 0}
                      onClick={() => setModalities((prev) => toggle(prev, key))}
                      title={hint || undefined}
                      count={count}
                    >
                      {label}
                    </Chip>
                  );
                });
              })()}
            </ChipRow>

            <ChipRow label="Status">
              {STATUS_OPTIONS.map((s) => (
                <Chip
                  key={s}
                  active={statuses.includes(s)}
                  onClick={() => setStatuses((prev) => toggle(prev, s))}
                >
                  {s}
                </Chip>
              ))}
              <Chip
                active={hasEval === true}
                onClick={() => setHasEval(hasEval === true ? undefined : true)}
              >
                has eval
              </Chip>
              <Chip
                active={hasEval === false}
                onClick={() => setHasEval(hasEval === false ? undefined : false)}
              >
                no eval
              </Chip>
            </ChipRow>

            {/* Model substring search */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled w-[60px]">
                Model
              </span>
              <div className="flex-1 flex items-center gap-1.5 rounded-md border border-border-subtle bg-bg px-2.5 py-1.5 focus-within:border-accent transition-colors">
                <Search className="h-3 w-3 text-fg-disabled" />
                <input
                  type="text"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="substring (e.g. Qwen, Llama-3.2)…"
                  className="flex-1 bg-transparent border-0 text-[12px] focus:outline-none placeholder:text-fg-disabled"
                />
                {model ? (
                  <button
                    type="button"
                    onClick={() => setModel("")}
                    className="text-fg-disabled hover:text-fg"
                    aria-label="Clear model filter"
                  >
                    <X className="h-3 w-3" />
                  </button>
                ) : null}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <Card>
          <CardContent className="p-0">
            {isLoading ? (
              <div className="space-y-px p-6">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-12 animate-pulse bg-surface-hover/40" />
                ))}
              </div>
            ) : items.length === 0 ? (
              <div className="px-6 py-12 text-center text-sm text-fg-muted">
                {filtersActive
                  ? "No runs match the current filters."
                  : "No runs indexed yet. Launch a training run from /train to populate."}
              </div>
            ) : (
              <div className={cn(isFetching && "opacity-70 transition-opacity")}>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border-subtle text-[11px] uppercase tracking-wider text-fg-subtle">
                      <th className="px-4 py-2 text-left font-medium">Run</th>
                      <th className="px-4 py-2 text-left font-medium">Modality</th>
                      <th className="px-4 py-2 text-left font-medium">Model</th>
                      <th className="px-4 py-2 text-left font-medium">Status</th>
                      <th className="px-4 py-2 text-right font-medium">Cycles</th>
                      <th className="px-4 py-2 text-right font-medium">Loss</th>
                      <th className="px-4 py-2 text-right font-medium">When</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((run) => {
                      const verdict = run.effectiveness?.verdict;
                      const tone =
                        verdict === "passed"
                          ? "success"
                          : verdict === "failed"
                            ? "danger"
                            : verdict
                              ? "warning"
                              : "neutral";
                      return (
                        <tr
                          key={run.run_id}
                          className="border-b border-border-subtle last:border-0 hover:bg-surface-hover transition-colors"
                        >
                          <td className="px-4 py-2.5">
                            <Link
                              to="/runs/$runId"
                              params={{ runId: run.run_id }}
                              className="font-mono text-xs text-accent hover:underline"
                            >
                              {run.run_id.slice(0, 16)}
                            </Link>
                          </td>
                          <td className="px-4 py-2.5 capitalize text-fg-muted">
                            {run.modality}
                          </td>
                          <td className="px-4 py-2.5 truncate max-w-[28ch] text-fg">
                            {run.model_name}
                          </td>
                          <td className="px-4 py-2.5">
                            <Badge tone={tone} dot size="sm">
                              {verdict ?? run.status ?? "pending"}
                            </Badge>
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-fg-muted">
                            {run.cycles_executed ?? "—"}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-fg">
                            {typeof run.final_train_loss === "number"
                              ? run.final_train_loss.toFixed(3)
                              : "—"}
                          </td>
                          <td className="px-4 py-2.5 text-right text-fg-muted whitespace-nowrap">
                            {relativeTime((run as { timestamp?: string }).timestamp)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function ChipRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled w-[60px] pt-1.5">
        {label}
      </span>
      <div className="flex-1 flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function Chip({
  active,
  empty = false,
  onClick,
  count,
  title,
  children,
}: {
  active: boolean;
  empty?: boolean;
  onClick: () => void;
  count?: number;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={title}
      className={cn(
        "h-6 px-2 rounded-md border text-[11px] capitalize tracking-tight transition-colors inline-flex items-center gap-1",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
        active
          ? "border-accent bg-accent-bg text-accent font-medium"
          : empty
            // Dim look for kinds with zero runs — still clickable so
            // users can pre-select a filter, but visibly secondary.
            ? "border-border-subtle/60 text-fg-disabled hover:text-fg-muted hover:border-border-subtle"
            : "border-border-subtle text-fg-muted hover:text-fg hover:border-border",
      )}
    >
      {children}
      {typeof count === "number" && count > 0 ? (
        <span
          className={cn(
            "font-mono text-[10px] tabular-nums px-1 rounded-sm normal-case tracking-normal",
            active ? "bg-accent/15 text-accent" : "bg-surface text-fg-disabled",
          )}
        >
          {count}
        </span>
      ) : null}
    </button>
  );
}
