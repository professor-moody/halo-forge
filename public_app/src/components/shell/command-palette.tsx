import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import {
  Activity,
  ArrowRight,
  Box,
  Command,
  HardDrive,
  MessageSquare,
  Search,
} from "lucide-react";
import { api, type GlobalSearchResult } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ACTIVITY_NAV_ITEM, PRIMARY_NAV, SYSTEM_NAV } from "./navigation";

type PaletteCommand = {
  id: string;
  label: string;
  description: string;
  group: "Navigate" | "Search" | "Models" | "Actions";
  icon: typeof Activity;
  shortcut?: string;
  keywords: string;
  run: () => void;
};

export function CommandPalette({
  open,
  onOpenChange,
  onOpenActivity,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onOpenActivity: () => void;
}) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [remoteQuery, setRemoteQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const chordRef = useRef<string | null>(null);
  const chordTimerRef = useRef<number | null>(null);

  const commands = useMemo<PaletteCommand[]>(() => {
    const navigateCommands = [...PRIMARY_NAV, ...SYSTEM_NAV].map((item) => ({
      id: item.id,
      label: item.label,
      description: item.description,
      group: "Navigate" as const,
      icon: item.icon,
      shortcut: item.shortcut,
      keywords: `${item.label} ${item.description} ${(item.aliases ?? []).join(" ")}`,
      run: () => navigate({ to: item.to as "/" }),
    }));
    return [
      ...navigateCommands,
      {
        id: "new-experiment",
        label: "New experiment",
        description: "Configure seeds, checkpoint gates, and evidence",
        group: "Actions",
        icon: Activity,
        keywords: "new experiment repeat sweep checkpoint gate adaptive",
        run: () => navigate({ to: "/sweeps" }),
      },
      {
        id: "models-artifacts",
        label: "Trained artifacts",
        description: "Open the content-addressed artifact library",
        group: "Models",
        icon: Box,
        keywords: "models artifacts checkpoints adapters merged converted quantized",
        run: () => navigate({ to: "/models", search: { tab: "artifacts" } }),
      },
      {
        id: "models-cache",
        label: "Cached models",
        description: "Inspect downloaded model storage",
        group: "Models",
        icon: HardDrive,
        keywords: "models cache disk downloads",
        run: () => navigate({ to: "/models", search: { tab: "cached" } }),
      },
      {
        id: "models-serve",
        label: "Serve & Test",
        description: "Start a local endpoint and open a session",
        group: "Models",
        icon: MessageSquare,
        keywords: "models serve playground test chat inference",
        run: () => navigate({ to: "/models", search: { tab: "serve" } }),
      },
      {
        id: ACTIVITY_NAV_ITEM.id,
        label: ACTIVITY_NAV_ITEM.label,
        description: ACTIVITY_NAV_ITEM.description,
        group: "Actions",
        icon: ACTIVITY_NAV_ITEM.icon,
        shortcut: ACTIVITY_NAV_ITEM.shortcut,
        keywords: "activity queue workers jobs telemetry blockers retry",
        run: onOpenActivity,
      },
    ];
  }, [navigate, onOpenActivity]);

  useEffect(() => {
    const timer = window.setTimeout(() => setRemoteQuery(query.trim()), 180);
    return () => window.clearTimeout(timer);
  }, [query]);

  const remote = useQuery({
    queryKey: ["global-search", remoteQuery],
    queryFn: () => api.globalSearch(remoteQuery, { limit: 30 }),
    enabled: open && remoteQuery.length >= 2,
    retry: false,
    staleTime: 15_000,
  });

  const searchCommands = useMemo<PaletteCommand[]>(() => (remote.data?.items ?? []).map((result) => ({
    id: `search-${result.type}-${result.id}`,
    label: result.label,
    description: [friendlyType(result.type), result.status, result.description].filter(Boolean).join(" · "),
    group: "Search" as const,
    icon: searchIcon(result.type),
    keywords: `${result.label} ${result.description ?? ""} ${result.short_id ?? result.short_hash ?? ""} ${result.type}`,
    run: () => navigate({ to: resolvedSearchTarget(result) as "/" }),
  })), [navigate, remote.data?.items]);

  const visible = useMemo(() => {
    const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) return commands;
    const local = commands.filter((command) => {
      const haystack = `${command.label} ${command.keywords}`.toLowerCase();
      return tokens.every((token) => haystack.includes(token));
    });
    return [...local, ...searchCommands];
  }, [commands, query, searchCommands]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const editing = Boolean(target?.closest("input, textarea, select, [contenteditable='true']"));
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        onOpenChange(!open);
        return;
      }
      if (editing || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "?") {
        event.preventDefault();
        onOpenChange(true);
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "g") {
        chordRef.current = "g";
        if (chordTimerRef.current != null) window.clearTimeout(chordTimerRef.current);
        chordTimerRef.current = window.setTimeout(() => {
          chordRef.current = null;
        }, 1_000);
        return;
      }
      if (chordRef.current === "g") {
        chordRef.current = null;
        if (chordTimerRef.current != null) window.clearTimeout(chordTimerRef.current);
        const command = commands.find((item) => item.shortcut?.toLowerCase() === `g ${key}`);
        if (command) {
          event.preventDefault();
          command.run();
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      if (chordTimerRef.current != null) window.clearTimeout(chordTimerRef.current);
    };
  }, [commands, onOpenChange, open]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setSelected(0);
  }, [open]);

  useEffect(() => setSelected(0), [query]);

  if (!open) return null;

  function run(command: PaletteCommand | undefined) {
    if (!command) return;
    command.run();
    onOpenChange(false);
  }

  return (
    <div className="workspace-overlay" role="presentation" onMouseDown={() => onOpenChange(false)}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="command-palette-enter mt-[12vh] w-[min(640px,calc(100vw-24px))] overflow-hidden rounded-lg border border-border-strong bg-elevated shadow-2xl shadow-black/35"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Escape") onOpenChange(false);
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setSelected((value) => Math.min(visible.length - 1, value + 1));
          }
          if (event.key === "ArrowUp") {
            event.preventDefault();
            setSelected((value) => Math.max(0, value - 1));
          }
          if (event.key === "Enter") {
            event.preventDefault();
            run(visible[selected]);
          }
        }}
      >
        <div className="flex h-12 items-center gap-3 border-b border-border px-4">
          <Search className="h-4 w-4 text-fg-disabled" />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search data, runs, suites, artifacts, policies, or actions…"
            aria-label="Search commands"
            className="h-full min-w-0 flex-1 bg-transparent text-[14px] text-fg outline-none placeholder:text-fg-disabled"
          />
          <kbd className="rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] text-fg-disabled">ESC</kbd>
        </div>
        <div className="max-h-[min(480px,65vh)] overflow-y-auto p-2">
          {visible.length ? (
            visible.map((command, index) => {
              const showGroup = index === 0 || visible[index - 1]?.group !== command.group;
              return (
                <div key={command.id}>
                  {showGroup ? (
                    <div className="px-2 pb-1 pt-2 text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">
                      {command.group}
                    </div>
                  ) : null}
                  <button
                    type="button"
                    onMouseEnter={() => setSelected(index)}
                    onClick={() => run(command)}
                    className={cn(
                      "group flex w-full items-center gap-3 rounded-md px-2.5 py-2 text-left transition-colors",
                      selected === index ? "bg-accent-bg text-fg" : "text-fg-muted hover:bg-surface",
                    )}
                  >
                    <command.icon className={cn("h-4 w-4", selected === index ? "text-accent" : "text-fg-subtle")} />
                    <span className="min-w-0 flex-1">
                      <span className="block text-[12.5px] font-medium text-fg">{command.label}</span>
                      <span className="block truncate text-[10.5px] text-fg-subtle">{command.description}</span>
                    </span>
                    {command.shortcut ? (
                      <kbd className="font-mono text-[10px] text-fg-disabled">{command.shortcut}</kbd>
                    ) : (
                      <ArrowRight className="h-3.5 w-3.5 opacity-0 transition-opacity group-hover:opacity-100" />
                    )}
                  </button>
                </div>
              );
            })
          ) : remote.isFetching ? (
            <div className="grid h-28 place-items-center text-center"><div><Search className="mx-auto mb-2 h-4 w-4 animate-pulse text-accent" /><p className="text-[12px] text-fg-muted">Searching the workspace</p></div></div>
          ) : (
            <div className="grid h-28 place-items-center text-center">
              <div>
                <Command className="mx-auto mb-2 h-4 w-4 text-fg-disabled" />
                <p className="text-[12px] text-fg-muted">No matching command</p>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between border-t border-border-subtle px-4 py-2 text-[10px] text-fg-disabled">
          <span>↑↓ select · ↵ open</span>
          <span>Press G then a shortcut key anywhere</span>
        </div>
      </div>
    </div>
  );
}

function friendlyType(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function searchIcon(type: string): typeof Activity {
  if (type.includes("artifact") || type.includes("checkpoint")) return Box;
  if (type.includes("dataset")) return HardDrive;
  if (type.includes("suite") || type.includes("group")) return Activity;
  return Search;
}

function searchTarget(result: GlobalSearchResult): string {
  if (result.type === "run") return `/runs/${encodeURIComponent(result.id)}`;
  if (result.type === "run_group") return `/sweeps?group=${encodeURIComponent(result.id)}`;
  if (result.type === "artifact") return `/models?tab=artifacts&artifact=${encodeURIComponent(result.id)}`;
  if (result.type === "dataset") return `/datasets/${encodeURIComponent(result.id)}`;
  if (result.type === "suite") return `/eval?suite=${encodeURIComponent(result.id)}`;
  if (result.type === "checkpoint_policy") return `/sweeps?new=1&policy=${encodeURIComponent(result.id)}`;
  return "/";
}

function resolvedSearchTarget(result: GlobalSearchResult): string {
  if (result.type === "run_group" || result.type === "checkpoint_policy") return searchTarget(result);
  return result.target || result.url || searchTarget(result);
}
