import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bookmark, GitCompareArrows, Plus, Search, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useRunSearch } from "@/lib/hooks";
import { Topbar } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, relativeTime } from "@/lib/utils";
import { api, type RegistryEntry } from "@/lib/api";
import { clearPinned, pinRun, usePinnedRuns } from "@/lib/pinned-runs";
import { Input } from "@/components/ui/input";

export const Route = createFileRoute("/runs/")({
  validateSearch: (search: Record<string, unknown>): { view?: "all" | "completed" | "collections" } => ({
    view: ["all", "completed", "collections"].includes(String(search.view))
      ? search.view as "all" | "completed" | "collections"
      : undefined,
  }),
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
  { key: "orpo", label: "orpo", hint: "Odds-Ratio Preference Optimization (reference-free)" },
  { key: "grpo", label: "grpo", hint: "Group Relative Policy Optimization" },
  { key: "rm", label: "rm", hint: "Reward model (Bradley-Terry)" },
  { key: "vlm", label: "vlm", hint: "Vision-language" },
  { key: "audio", label: "audio", hint: "Speech / audio" },
  { key: "reasoning", label: "reasoning", hint: "Reasoning / chain-of-thought" },
  { key: "agentic", label: "agentic", hint: "Agentic / tool-use" },
];

function RunsListRoute() {
  const view = Route.useSearch().view ?? "all";
  const [modalities, setModalities] = useState<string[]>([]);
  const [statuses, setStatuses] = useState<StatusKey[]>(view === "completed" ? ["completed"] : []);
  const [model, setModel] = useState("");
  const [hasEval, setHasEval] = useState<boolean | undefined>(undefined);

  useEffect(() => {
    if (view === "completed") setStatuses(["completed"]);
    if (view === "all") setStatuses([]);
  }, [view]);

  if (view === "collections") {
    return (
      <>
        <Topbar eyebrow="Workspace" title="Runs" subtitle="Monitor work, review completed outputs, and reopen saved comparison sets." />
        <RunsTabs active={view} />
        <CollectionsWorkspace />
      </>
    );
  }

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
      <RunsTabs active={view} />
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
                  : "No runs indexed yet. Launch a guided run from Train to populate."}
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

function RunsTabs({ active }: { active: "all" | "completed" | "collections" }) {
  const navigate = useNavigate();
  const tabs = [
    { id: "all" as const, label: "All runs" },
    { id: "completed" as const, label: "Completed" },
    { id: "collections" as const, label: "Collections" },
  ];
  return (
    <div className="flex border-b border-border bg-bg-subtle/55 px-4 md:px-6">
      {tabs.map((tab) => (
        <button key={tab.id} type="button" onClick={() => navigate({ to: "/runs", search: { view: tab.id } })} className={cn("relative h-10 px-3 text-[11.5px] transition-colors", active === tab.id ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>
          {tab.label}
          {active === tab.id ? <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" /> : null}
        </button>
      ))}
    </div>
  );
}

function CollectionsWorkspace() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const pinned = usePinnedRuns();
  const [name, setName] = useState("");
  const collections = useQuery<{ items: RegistryEntry[] }>({
    queryKey: ["registry"],
    queryFn: api.listRegistry,
    retry: false,
  });
  const create = useMutation({
    mutationFn: () => api.createRegistryEntry({ name: name.trim(), run_ids: pinned }),
    onSuccess: () => {
      setName("");
      queryClient.invalidateQueries({ queryKey: ["registry"] });
    },
  });
  const remove = useMutation({
    mutationFn: (id: number) => api.deleteRegistryEntry(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["registry"] }),
  });

  function openCollection(entry: RegistryEntry) {
    clearPinned();
    entry.run_ids.forEach(pinRun);
    navigate({ to: "/runs/compare" });
  }

  return (
    <div className="mx-auto max-w-5xl px-5 py-5">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
        <section>
          <div className="mb-3 flex items-end justify-between gap-3">
            <div><h2 className="text-[13px] font-medium text-fg">Saved comparison sets</h2><p className="mt-1 text-[10.5px] text-fg-subtle">Named groups preserve a repeatable set of run identities.</p></div>
            <span className="font-mono text-[10px] text-fg-disabled">{collections.data?.items.length ?? 0} collections</span>
          </div>
          <div className="divide-y divide-border-subtle border-y border-border-subtle">
            {collections.isLoading ? (
              <div className="flex h-32 items-center justify-center gap-2 text-[11px] text-fg-muted"><Search className="h-3.5 w-3.5 animate-pulse" /> Loading collections</div>
            ) : collections.data?.items.length ? collections.data.items.map((entry) => (
              <div key={entry.id} className="group flex items-center gap-3 py-3">
                <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md border border-border-subtle bg-surface text-fg-subtle"><Bookmark className="h-3.5 w-3.5" /></div>
                <div className="min-w-0 flex-1"><div className="truncate text-[12px] font-medium text-fg">{entry.name}</div><div className="mt-1 text-[10px] text-fg-disabled">{entry.run_ids.length} run{entry.run_ids.length === 1 ? "" : "s"} · {relativeTime(entry.updated_at)}</div>{entry.tags.length ? <div className="mt-1.5 flex flex-wrap gap-1">{entry.tags.map((tag) => <span key={tag} className="rounded-sm border border-border-subtle px-1.5 py-0.5 text-[9px] text-fg-subtle">{tag}</span>)}</div> : null}</div>
                <Button size="sm" variant="ghost" onClick={() => openCollection(entry)} disabled={!entry.run_ids.length}><GitCompareArrows /> Compare</Button>
                <Button size="icon" variant="ghost" onClick={() => remove.mutate(entry.id)} className="opacity-0 group-hover:opacity-100" aria-label={`Delete ${entry.name}`}><Trash2 /></Button>
              </div>
            )) : <div className="grid h-36 place-items-center text-center"><div><Bookmark className="mx-auto h-4 w-4 text-fg-disabled" /><p className="mt-2 text-[11.5px] text-fg-muted">No collections yet</p><p className="mt-1 text-[10px] text-fg-disabled">Pin runs, then save the working set.</p></div></div>}
          </div>
        </section>
        <aside className="h-fit rounded-md border border-border-subtle bg-bg-subtle/40 p-4">
          <div className="text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">Save current working set</div>
          <div className="mt-2 font-mono text-[18px] text-fg">{pinned.length} pinned</div>
          <p className="mt-1 text-[10.5px] leading-relaxed text-fg-subtle">Collections store run identities, not copies of model files.</p>
          <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Collection name" className="mt-3 h-8 text-[11px]" />
          <Button className="mt-2 w-full" size="sm" onClick={() => create.mutate()} disabled={!name.trim() || !pinned.length || create.isPending}>{create.isPending ? <Plus className="animate-pulse" /> : <Plus />} Save collection</Button>
          {create.error ? <p className="mt-2 text-[10px] text-danger">{create.error.message}</p> : null}
        </aside>
      </div>
    </div>
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
