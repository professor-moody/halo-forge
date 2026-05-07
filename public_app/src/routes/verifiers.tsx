import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Loader2, ShieldCheck, Plug, Package } from "lucide-react";
import { useMemo, useState } from "react";
import { api, type VerifierCatalogEntry } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardEyebrow,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/verifiers")({
  component: VerifiersRoute,
});

/**
 * Verifier catalog (Track F-O lite).
 *
 * Inventory of every verifier the runtime can resolve at the moment —
 * built-ins, files dropped into `~/.halo-forge/verifiers/`, and entry-
 * point plugins from pip-installed packages. Useful for two questions:
 *
 *   1. "Which name do I pass to `--verifier=` for X?" — search the list.
 *   2. "Did my plugin actually load?" — filter to user_plugin /
 *      entry_point. If it's missing, the trainer wouldn't have seen it
 *      either.
 */

type Filter = "all" | "builtin" | "user_plugin" | "entry_point";

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "builtin", label: "Built-in" },
  { id: "user_plugin", label: "User plugins" },
  { id: "entry_point", label: "Entry points" },
];

function VerifiersRoute() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["verifier-catalog"],
    queryFn: () => api.verifierCatalog(),
    staleTime: 30_000,
  });

  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  const items = data?.items ?? [];
  const counts = data?.counts ?? {};

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((e) => {
      if (filter !== "all" && e.origin !== filter) return false;
      if (!q) return true;
      return (
        e.name.toLowerCase().includes(q) ||
        e.cls.toLowerCase().includes(q) ||
        (e.doc ?? "").toLowerCase().includes(q)
      );
    });
  }, [items, filter, query]);

  return (
    <>
      <Topbar
        eyebrow="Plugins"
        title="Verifier catalog"
        subtitle={
          data
            ? `${data.total} registered · drop a .py into ${data.plugin_dir} to add one`
            : undefined
        }
        statusBar={
          <>
            <span>built-in {counts.builtin ?? 0}</span>
            <span aria-hidden>·</span>
            <span>user {counts.user_plugin ?? 0}</span>
            <span aria-hidden>·</span>
            <span>entry-point {counts.entry_point ?? 0}</span>
          </>
        }
      />

      <div className="px-5 py-5 space-y-4 max-w-5xl">
        {/* Filter chips + search */}
        <div className="flex flex-wrap items-center gap-2">
          <div role="radiogroup" aria-label="Filter by origin" className="flex gap-1">
            {FILTERS.map((f) => {
              const active = filter === f.id;
              return (
                <button
                  key={f.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() => setFilter(f.id)}
                  className={cn(
                    "h-7 rounded-md px-2.5 text-[12px] transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                    active
                      ? "bg-accent-bg text-accent font-medium"
                      : "bg-surface text-fg-muted hover:bg-surface-hover hover:text-fg border border-border",
                  )}
                >
                  {f.label}
                </button>
              );
            })}
          </div>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by name, class, or doc…"
            className={cn(
              "ml-auto h-7 min-w-[220px] flex-1 max-w-[360px] rounded-md border border-border bg-surface px-2.5 text-[12px]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
            )}
            aria-label="Search verifiers"
          />
        </div>

        {isLoading ? (
          <div className="flex h-32 items-center justify-center text-fg-muted text-sm gap-2">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading catalog…
          </div>
        ) : isError ? (
          <div className="text-danger text-sm">Failed to load verifier catalog.</div>
        ) : filtered.length === 0 ? (
          <div className="text-fg-muted text-sm py-8 text-center">
            No verifiers match the current filter.
          </div>
        ) : (
          <ul className="space-y-2" aria-label="Registered verifiers">
            {filtered.map((entry) => (
              <li key={entry.name}>
                <CatalogRow entry={entry} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

function CatalogRow({ entry }: { entry: VerifierCatalogEntry }) {
  return (
    <Card className="hover:border-border-strong transition-colors">
      <CardHeader className="pb-2">
        <CardEyebrow>{entry.module || "—"}</CardEyebrow>
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="font-mono text-[14px]">{entry.name}</CardTitle>
          <OriginBadge origin={entry.origin} />
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-1.5">
        {entry.doc ? (
          <p className="text-[12px] text-fg-muted">{entry.doc}</p>
        ) : null}
        <code className="block font-mono text-[11px] text-fg-disabled break-all">
          {entry.cls}
        </code>
      </CardContent>
    </Card>
  );
}

function OriginBadge({ origin }: { origin: VerifierCatalogEntry["origin"] }) {
  switch (origin) {
    case "builtin":
      return (
        <Badge tone="info" size="sm">
          <ShieldCheck className="h-3 w-3" aria-hidden /> built-in
        </Badge>
      );
    case "user_plugin":
      return (
        <Badge tone="success" size="sm">
          <Plug className="h-3 w-3" aria-hidden /> user plugin
        </Badge>
      );
    case "entry_point":
      return (
        <Badge tone="neutral" size="sm">
          <Package className="h-3 w-3" aria-hidden /> entry-point
        </Badge>
      );
  }
}
