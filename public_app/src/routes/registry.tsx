import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  ArrowRight,
  BookmarkPlus,
  Loader2,
  Trash2,
  X,
} from "lucide-react";
import { useState } from "react";
import { api, type RegistryEntry } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardEyebrow,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { clearPinned, pinRun, usePinnedRuns } from "@/lib/pinned-runs";
import { cn, relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/registry")({
  beforeLoad: () => {
    throw redirect({ to: "/runs", search: { view: "collections" }, replace: true });
  },
  component: RegistryRoute,
});

/**
 * Run bundle registry (Track F-J).
 *
 * Lists every saved registry entry — named bundles of runs the user
 * wants to compare/promote as a unit. Two affordances:
 *
 *   1. "New entry" — form below the list. Save the currently pinned
 *      runs (or create empty and add later).
 *   2. "Load to cohort" — pins all the entry's runs, navigates to
 *      /eval. Bridges saved sets back to the comparison surface.
 *
 * Edit-in-place isn't shipped here; PATCH lives on the API surface
 * for the day a fancier editor lands. Today: create + delete + load.
 */

function RegistryRoute() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const pinned = usePinnedRuns();

  const { data, isLoading } = useQuery<{ items: RegistryEntry[] }>({
    queryKey: ["registry"],
    queryFn: () => api.listRegistry(),
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
  });

  const items = data?.items ?? [];

  const createMutation = useMutation({
    mutationFn: api.createRegistryEntry,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => api.deleteRegistryEntry(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry"] });
    },
  });

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [tagsRaw, setTagsRaw] = useState("");
  const [error, setError] = useState<string | null>(null);

  function reset() {
    setName("");
    setDescription("");
    setBaseModel("");
    setTagsRaw("");
    setError(null);
  }

  async function handleCreate(opts: { fromPinned: boolean }) {
    setError(null);
    if (!name.trim()) {
      setError("name is required");
      return;
    }
    try {
      await createMutation.mutateAsync({
        name: name.trim(),
        description: description.trim() || null,
        base_model: baseModel.trim() || null,
        run_ids: opts.fromPinned ? pinned : [],
        tags: tagsRaw
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      });
      reset();
    } catch (err) {
      setError((err as Error).message ?? "create failed");
    }
  }

  function loadToCohort(entry: RegistryEntry) {
    if (!entry.run_ids.length) return;
    clearPinned();
    for (const runId of entry.run_ids) pinRun(runId);
    navigate({ to: "/eval" });
  }

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Run Bundles"
        subtitle={
          items.length === 0
            ? "Save named bundles of trained runs for cohort eval and side-by-side comparison."
            : `${items.length} saved bundle${items.length === 1 ? "" : "s"}`
        }
        actions={
          <Button variant="ghost" size="icon" asChild aria-label="Back to runs">
            <Link to="/runs">
              <ArrowLeft />
            </Link>
          </Button>
        }
      />
      <div className="px-5 py-5 space-y-4">
        <NewEntryForm
          name={name}
          setName={setName}
          description={description}
          setDescription={setDescription}
          baseModel={baseModel}
          setBaseModel={setBaseModel}
          tagsRaw={tagsRaw}
          setTagsRaw={setTagsRaw}
          pinnedCount={pinned.length}
          isPending={createMutation.isPending}
          error={error}
          onCreate={handleCreate}
        />

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>SAVED</CardEyebrow>
              <CardTitle>Bundles</CardTitle>
            </div>
            <span className="text-[11px] text-fg-subtle">{items.length} total</span>
          </CardHeader>
          <CardContent className="p-0">
            {isLoading ? (
              <div className="px-6 py-12 text-center text-sm text-fg-muted">
                Loading…
              </div>
            ) : items.length === 0 ? (
              <div className="px-6 py-10 text-center text-xs text-fg-muted max-w-[44ch] mx-auto">
                No bundles saved yet. Pin a few runs from{" "}
                <Link to="/runs" className="text-accent hover:underline">
                  /runs
                </Link>
                , then create an entry above.
              </div>
            ) : (
              <table className="w-full text-[12.5px]">
                <thead>
                  <tr className="border-b border-border-subtle text-[10px] uppercase tracking-[0.12em] text-fg-disabled">
                    <th className="px-4 py-2 text-left font-medium">Name</th>
                    <th className="px-4 py-2 text-left font-medium">Base model</th>
                    <th className="px-4 py-2 text-right font-medium">Runs</th>
                    <th className="px-4 py-2 text-left font-medium">Tags</th>
                    <th className="px-4 py-2 text-right font-medium">Updated</th>
                    <th className="px-4 py-2 text-right font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((entry) => (
                    <tr
                      key={entry.id}
                      className="border-b border-border-subtle last:border-0 hover:bg-surface-hover/40 transition-colors"
                    >
                      <td className="px-4 py-2.5">
                        <div className="font-medium text-fg" title={entry.description ?? undefined}>
                          {entry.name}
                        </div>
                        {entry.description ? (
                          <div className="text-[11px] text-fg-muted truncate max-w-[40ch]">
                            {entry.description}
                          </div>
                        ) : null}
                      </td>
                      <td className="px-4 py-2.5 font-mono text-[11px] text-fg-muted truncate max-w-[24ch]">
                        {entry.base_model ?? "—"}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                        {entry.run_ids.length}
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex flex-wrap gap-1">
                          {entry.tags.map((t) => (
                            <Badge key={t} tone="neutral" size="sm">
                              {t}
                            </Badge>
                          ))}
                        </div>
                      </td>
                      <td className="px-4 py-2.5 text-right text-fg-muted whitespace-nowrap">
                        {relativeTime(entry.updated_at)}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <div className="inline-flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            disabled={!entry.run_ids.length}
                            onClick={() => loadToCohort(entry)}
                            title={
                              entry.run_ids.length
                                ? "Pin all runs and open cohort eval"
                                : "Bundle has no runs to load"
                            }
                          >
                            Load <ArrowRight className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            disabled={deleteMutation.isPending}
                            onClick={() => {
                              if (confirm(`Delete bundle "${entry.name}"? This can't be undone.`)) {
                                deleteMutation.mutate(entry.id);
                              }
                            }}
                            aria-label={`Delete ${entry.name}`}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function NewEntryForm({
  name, setName,
  description, setDescription,
  baseModel, setBaseModel,
  tagsRaw, setTagsRaw,
  pinnedCount,
  isPending,
  error,
  onCreate,
}: {
  name: string; setName: (v: string) => void;
  description: string; setDescription: (v: string) => void;
  baseModel: string; setBaseModel: (v: string) => void;
  tagsRaw: string; setTagsRaw: (v: string) => void;
  pinnedCount: number;
  isPending: boolean;
  error: string | null;
  onCreate: (opts: { fromPinned: boolean }) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>NEW</CardEyebrow>
          <CardTitle>Create bundle</CardTitle>
          <BookmarkPlus className="h-3.5 w-3.5 text-fg-disabled" />
        </div>
        <span className="text-[11px] text-fg-subtle">
          {pinnedCount} pinned run{pinnedCount === 1 ? "" : "s"} ready to capture
        </span>
      </CardHeader>
      <CardContent className="space-y-2.5">
        <FormRow label="Name">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="prod-2026-q2"
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent"
          />
        </FormRow>
        <FormRow label="Description">
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Top picks from the Q2 sweep"
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent"
          />
        </FormRow>
        <FormRow label="Base model">
          <input
            type="text"
            value={baseModel}
            onChange={(e) => setBaseModel(e.target.value)}
            placeholder="Qwen/Qwen2.5-3B-Instruct"
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent font-mono"
          />
        </FormRow>
        <FormRow label="Tags">
          <input
            type="text"
            value={tagsRaw}
            onChange={(e) => setTagsRaw(e.target.value)}
            placeholder="dpo, production, candidate"
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent"
          />
        </FormRow>

        {error ? (
          <div className="flex items-center gap-2 text-[11px] text-danger">
            <X className="h-3 w-3" />
            {error}
          </div>
        ) : null}

        <div className="flex flex-wrap gap-2 pt-1">
          <Button
            variant="primary"
            size="sm"
            onClick={() => onCreate({ fromPinned: true })}
            disabled={isPending || pinnedCount === 0}
            title={
              pinnedCount === 0
                ? "Pin runs from /runs first"
                : "Capture the currently pinned runs"
            }
          >
            {isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <BookmarkPlus className="h-3 w-3" />}
            Save with {pinnedCount} pinned run{pinnedCount === 1 ? "" : "s"}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCreate({ fromPinned: false })}
            disabled={isPending}
          >
            Save empty
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function FormRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className={cn(
        "text-[10px] uppercase tracking-[0.12em] text-fg-disabled",
        "w-[88px] shrink-0",
      )}>
        {label}
      </span>
      <div className="flex-1">{children}</div>
    </div>
  );
}
