import { useState } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { Activity, ChevronDown, GitCompareArrows, Search, Settings2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { ApiError, connectionMode, getApiToken, isAuthRequiredError } from "@/lib/api";
import { useActivity, useBackendInfo, useVersionInfo } from "@/lib/hooks";
import { usePinnedRuns } from "@/lib/pinned-runs";
import { ThemeToggle } from "./theme-toggle";
import { PRIMARY_NAV, SYSTEM_NAV, isNavigationActive } from "./navigation";

export function Sidebar({
  onOpenActivity,
  onOpenCommand,
}: {
  onOpenActivity: () => void;
  onOpenCommand: () => void;
}) {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const activity = useActivity(100);
  const [systemOpen, setSystemOpen] = useState(() =>
    SYSTEM_NAV.some((item) => isNavigationActive(pathname, item.to)),
  );
  const activeCount = (activity.data?.items ?? []).filter((item) =>
    ["queued", "running", "blocked", "preparing"].includes(item.status),
  ).length;
  const attentionCount = (activity.data?.items ?? []).filter((item) =>
    ["failed", "interrupted", "needs_reconciliation"].includes(item.status),
  ).length;

  return (
    <aside className="hidden h-screen w-56 shrink-0 flex-col border-r border-border bg-bg-subtle md:flex">
      <div className="flex h-12 items-center gap-1 border-b border-border-subtle px-2">
        <Link to="/" className="group flex min-w-0 flex-1 items-center gap-2.5 rounded-sm px-1.5 py-1.5 hover:bg-surface/40">
          <img src="/mark.svg" alt="" width={20} height={20} className="opacity-95 transition-opacity group-hover:opacity-100" />
          <span className="truncate text-[13.5px] font-medium tracking-tight text-fg">halo<span className="text-fg-subtle">-</span>forge</span>
        </Link>
        <button type="button" onClick={onOpenCommand} className="grid h-7 w-7 place-items-center rounded-sm text-fg-disabled transition-colors hover:bg-surface hover:text-fg" title="Open command palette (⌘K)" aria-label="Open command palette">
          <Search className="h-3.5 w-3.5" />
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-1.5 py-2.5" aria-label="Workspace navigation">
        <SectionLabel>Workspace</SectionLabel>
        <ul className="space-y-px">
          {PRIMARY_NAV.map((item) => <NavigationLink key={item.id} item={item} active={isNavigationActive(pathname, item.to)} />)}
        </ul>
      </nav>

      <ComparisonTray />

      <div className="border-t border-border-subtle px-1.5 py-2">
        <SectionLabel>Operations</SectionLabel>
        <button
          type="button"
          onClick={onOpenActivity}
          className="group flex h-8 w-full items-center gap-2.5 rounded-sm px-2 text-[12.5px] text-fg-muted transition-colors hover:bg-surface hover:text-fg focus-visible:ring-2 focus-visible:ring-accent"
        >
          <Activity className={cn("h-3.5 w-3.5", activeCount > 0 && "text-accent")} />
          <span className="flex-1 text-left">Activity</span>
          {attentionCount > 0 ? (
            <span className="min-w-5 rounded-sm bg-danger-bg px-1 font-mono text-[9.5px] text-danger">{attentionCount}</span>
          ) : activeCount > 0 ? (
            <span className="min-w-5 rounded-sm bg-accent-bg px-1 font-mono text-[9.5px] text-accent">{activeCount}</span>
          ) : (
            <kbd className="font-mono text-[9.5px] text-fg-disabled">G A</kbd>
          )}
        </button>

        <button
          type="button"
          onClick={() => setSystemOpen((value) => !value)}
          aria-expanded={systemOpen}
          className="group mt-px flex h-8 w-full items-center gap-2.5 rounded-sm px-2 text-[12.5px] text-fg-muted transition-colors hover:bg-surface hover:text-fg"
        >
          <Settings2 className="h-3.5 w-3.5" />
          <span className="flex-1 text-left">System</span>
          <ChevronDown className={cn("h-3 w-3 text-fg-disabled transition-transform duration-150", systemOpen && "rotate-180")} />
        </button>
        {systemOpen ? (
          <ul className="mt-1 space-y-px border-l border-border-subtle pl-2">
            {SYSTEM_NAV.map((item) => <NavigationLink key={item.id} item={item} active={isNavigationActive(pathname, item.to)} compact />)}
          </ul>
        ) : null}
      </div>

      <div className="border-t border-border-subtle px-2.5 py-2">
        <div className="flex items-center justify-between">
          <SectionLabel>Appearance</SectionLabel>
          <kbd className="pb-1.5 font-mono text-[9px] text-fg-disabled">⌘K</kbd>
        </div>
        <ThemeToggle />
      </div>
      <ComputePanel />
    </aside>
  );
}

function NavigationLink({ item, active, compact }: { item: (typeof PRIMARY_NAV)[number]; active: boolean; compact?: boolean }) {
  return (
    <li>
      <Link
        to={item.to}
        aria-current={active ? "page" : undefined}
        className={cn(
          "relative flex items-center gap-2.5 rounded-sm px-2 text-[12.5px] transition-colors focus-visible:ring-2 focus-visible:ring-accent",
          compact ? "h-7" : "h-8",
          active ? "bg-accent-bg text-accent font-medium" : "text-fg-muted hover:bg-surface hover:text-fg",
        )}
      >
        {active ? <span aria-hidden className="absolute -left-1.5 inset-y-1.5 w-0.5 rounded-full bg-accent" /> : null}
        <item.icon className="h-3.5 w-3.5" />
        <span className="min-w-0 flex-1 truncate">{item.label}</span>
        {!compact && item.shortcut ? <kbd className={cn("font-mono text-[9.5px] tracking-tight", active ? "text-accent/70" : "text-fg-disabled")}>{item.shortcut}</kbd> : null}
      </Link>
    </li>
  );
}

function ComparisonTray() {
  const pinned = usePinnedRuns();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  if (!pinned.length) return null;
  const active = pathname === "/runs/compare";
  return (
    <div className="border-t border-border-subtle px-1.5 py-2">
      <SectionLabel>Working set</SectionLabel>
      <Link to="/runs/compare" aria-current={active ? "page" : undefined} className={cn("relative flex h-8 items-center gap-2.5 rounded-sm px-2 text-[12.5px] transition-colors", active ? "bg-accent-bg text-accent font-medium" : "text-fg-muted hover:bg-surface hover:text-fg")}>
        {active ? <span aria-hidden className="absolute -left-1.5 inset-y-1.5 w-0.5 rounded-full bg-accent" /> : null}
        <GitCompareArrows className="h-3.5 w-3.5" />
        <span className="flex-1">Compare runs</span>
        <span className={cn("rounded-sm px-1.5 font-mono text-[9.5px]", active ? "bg-accent/15 text-accent" : "bg-surface text-fg-subtle")}>{pinned.length}</span>
      </Link>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="px-2 pb-1.5 pt-1 text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">{children}</div>;
}

function ComputePanel() {
  const { data, isLoading, isError, error } = useBackendInfo();
  const version = useVersionInfo();
  const mode = connectionMode();
  const authNeeded = isAuthRequiredError(error);
  const tokenStored = Boolean(getApiToken());
  return (
    <div className="border-t border-border-subtle p-2.5">
      <SectionLabel>Compute</SectionLabel>
      <div className="rounded-md border border-border-subtle bg-surface/60">
        {isLoading ? <div className="h-12 animate-pulse" /> : authNeeded ? (
          <div className="space-y-1.5 px-2.5 py-2"><Badge tone="danger" dot size="sm">Auth needed</Badge><Link to="/connect" className="block text-[11px] text-accent hover:underline">Enter token</Link></div>
        ) : isError || !data ? (
          <div className="space-y-1.5 px-2.5 py-2"><Badge tone="danger" dot size="sm">Offline</Badge>{error instanceof ApiError ? <div className="font-mono text-[10px] text-fg-disabled">{error.status}</div> : null}</div>
        ) : (
          <div className="space-y-1.5 px-2.5 py-2">
            <div className="flex items-center justify-between gap-2"><span className="font-mono text-[11px] text-fg">{prettyBackendName(data.name)}</span>{data.chip ? <span className="truncate font-mono text-[10px] text-fg-subtle">{data.chip.brand}</span> : null}<span aria-label="Online" className="status-dot" style={{ background: data.name === "mlx" ? "var(--color-accent)" : data.name.startsWith("rocm") ? "var(--color-success)" : "var(--color-info)" }} /></div>
            <div className="flex items-center justify-between gap-2 font-mono text-[9.5px] uppercase tracking-wider text-fg-subtle"><span>{data.capabilities.preferred_dtype_str}</span><span>{data.mlx_readiness?.executable ? "MLX ready" : data.capabilities.preferred_attn_impl}</span></div>
            <div className="flex items-center justify-between gap-2 border-t border-border-subtle pt-1.5 font-mono text-[9.5px] uppercase tracking-wider"><span className={mode === "remote" ? "text-warning" : "text-fg-subtle"}>{mode}</span><Link to="/connect" className={cn("hover:underline", tokenStored ? "text-success" : "text-fg-disabled")}>{tokenStored ? "token" : "no token"}</Link></div>
            {version.data ? <div className="flex items-center justify-between gap-2 border-t border-border-subtle pt-1.5 font-mono text-[9.5px] uppercase tracking-wider text-fg-disabled"><span>{version.data.release_channel}</span><span>{version.data.display_version}</span></div> : null}
          </div>
        )}
      </div>
    </div>
  );
}

function prettyBackendName(name: string): string {
  if (name === "rocm_gfx1151") return "ROCm · gfx1151";
  if (name === "mps") return "Apple · MPS";
  if (name === "mlx") return "Apple · MLX";
  return name.toUpperCase();
}
