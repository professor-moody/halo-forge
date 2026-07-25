import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { Activity, Menu, Search, X } from "lucide-react";
import { AUTH_REQUIRED_EVENT } from "@/lib/api";
import { useActivity } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import { ActivityCenter } from "./activity-center";
import { CommandPalette } from "./command-palette";
import { PRIMARY_NAV, SYSTEM_NAV, isNavigationActive } from "./navigation";
import { Sidebar } from "./sidebar";
import { TelemetryStrip } from "./telemetry";
import { ThemeToggle } from "./theme-toggle";

export function AppShell({ children }: { children: ReactNode }) {
  const [activityOpen, setActivityOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const openActivity = useCallback(() => {
    setCommandOpen(false);
    setActivityOpen(true);
  }, []);
  const openCommand = useCallback(() => {
    setActivityOpen(false);
    setCommandOpen(true);
  }, []);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-fg">
      <AuthRedirector />
      <Sidebar onOpenActivity={openActivity} onOpenCommand={openCommand} />
      <main id="main" tabIndex={-1} className="flex min-w-0 flex-1 flex-col overflow-hidden focus:outline-none">
        <MobileNav
          open={mobileOpen}
          onOpenChange={setMobileOpen}
          onOpenActivity={openActivity}
          onOpenCommand={openCommand}
        />
        <TelemetryStrip />
        <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
      </main>
      <ActivityCenter open={activityOpen} onClose={() => setActivityOpen(false)} />
      <CommandPalette open={commandOpen} onOpenChange={setCommandOpen} onOpenActivity={openActivity} />
    </div>
  );
}

export { Topbar } from "./topbar";

function MobileNav({
  open,
  onOpenChange,
  onOpenActivity,
  onOpenCommand,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onOpenActivity: () => void;
  onOpenCommand: () => void;
}) {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const activity = useActivity(100);
  const activeCount = (activity.data?.items ?? []).filter((item) => ["queued", "running", "blocked", "preparing"].includes(item.status)).length;

  useEffect(() => onOpenChange(false), [onOpenChange, pathname]);
  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) { if (event.key === "Escape") onOpenChange(false); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onOpenChange, open]);

  return (
    <>
      <div className="flex h-11 shrink-0 items-center gap-1 border-b border-border bg-bg-subtle px-2 md:hidden">
        <button type="button" onClick={() => onOpenChange(true)} className="grid h-8 w-8 place-items-center rounded-sm text-fg-muted hover:bg-surface hover:text-fg" aria-label="Open navigation"><Menu className="h-4 w-4" /></button>
        <Link to="/" className="mr-auto flex min-w-0 items-center gap-2 rounded-sm px-1.5 py-1 text-[13px] font-medium text-fg"><img src="/mark.svg" alt="" width={18} height={18} /><span className="truncate">halo-forge</span></Link>
        <button type="button" onClick={onOpenCommand} className="grid h-8 w-8 place-items-center rounded-sm text-fg-muted hover:bg-surface hover:text-fg" aria-label="Open command palette"><Search className="h-3.5 w-3.5" /></button>
        <button type="button" onClick={onOpenActivity} className="relative grid h-8 w-8 place-items-center rounded-sm text-fg-muted hover:bg-surface hover:text-fg" aria-label={`Open Activity Center${activeCount ? `, ${activeCount} active` : ""}`}><Activity className={cn("h-3.5 w-3.5", activeCount > 0 && "text-accent")} />{activeCount > 0 ? <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-accent" /> : null}</button>
      </div>
      {open ? (
        <div className="workspace-overlay justify-start md:hidden" role="presentation" onMouseDown={() => onOpenChange(false)}>
          <aside className="mobile-drawer-enter flex h-full w-[min(320px,88vw)] flex-col border-r border-border bg-bg-subtle shadow-2xl shadow-black/40" onMouseDown={(event) => event.stopPropagation()}>
            <div className="flex h-12 items-center gap-2 border-b border-border px-3"><img src="/mark.svg" alt="" width={20} height={20} /><span className="flex-1 text-[13.5px] font-medium text-fg">halo-forge</span><button type="button" onClick={() => onOpenChange(false)} className="grid h-8 w-8 place-items-center rounded-sm text-fg-muted hover:bg-surface hover:text-fg" aria-label="Close navigation"><X className="h-4 w-4" /></button></div>
            <nav className="min-h-0 flex-1 overflow-y-auto px-2 py-3" aria-label="Mobile navigation">
              <MobileSection label="Workspace">
                {PRIMARY_NAV.map((item) => <MobileLink key={item.id} item={item} active={isNavigationActive(pathname, item.to)} />)}
              </MobileSection>
              <MobileSection label="Operations">
                <button type="button" onClick={() => { onOpenChange(false); onOpenActivity(); }} className="flex h-11 w-full items-center gap-3 rounded-md px-3 text-left text-[13px] text-fg-muted hover:bg-surface hover:text-fg"><Activity className={cn("h-4 w-4", activeCount > 0 && "text-accent")} /><span className="flex-1">Activity</span>{activeCount > 0 ? <span className="rounded-sm bg-accent-bg px-1.5 font-mono text-[10px] text-accent">{activeCount}</span> : null}</button>
              </MobileSection>
              <MobileSection label="System">
                {SYSTEM_NAV.map((item) => <MobileLink key={item.id} item={item} active={isNavigationActive(pathname, item.to)} />)}
              </MobileSection>
            </nav>
            <div className="border-t border-border p-3"><div className="mb-2 text-[9.5px] uppercase tracking-[0.13em] text-fg-disabled">Appearance</div><ThemeToggle /></div>
          </aside>
        </div>
      ) : null}
    </>
  );
}

function MobileSection({ label, children }: { label: string; children: ReactNode }) {
  return <section className="mb-4"><h2 className="px-3 pb-1.5 text-[9.5px] font-medium uppercase tracking-[0.13em] text-fg-disabled">{label}</h2><div className="space-y-0.5">{children}</div></section>;
}

function MobileLink({ item, active }: { item: (typeof PRIMARY_NAV)[number]; active: boolean }) {
  return <Link to={item.to} aria-current={active ? "page" : undefined} className={cn("relative flex h-11 items-center gap-3 rounded-md px-3 text-[13px] transition-colors", active ? "bg-accent-bg font-medium text-accent" : "text-fg-muted hover:bg-surface hover:text-fg")}>{active ? <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}<item.icon className="h-4 w-4 shrink-0" /><span className="shrink-0">{item.label}</span><span className="ml-auto max-w-[145px] truncate text-right text-[10px] text-fg-disabled">{item.description}</span></Link>;
}

function AuthRedirector() {
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  useEffect(() => {
    function onAuthRequired() { if (pathname !== "/connect") navigate({ to: "/connect", search: { from: pathname } }); }
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, [navigate, pathname]);
  return null;
}
