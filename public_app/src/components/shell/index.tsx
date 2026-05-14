import type { ReactNode } from "react";
import { useEffect } from "react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { AUTH_REQUIRED_EVENT } from "@/lib/api";
import { Sidebar } from "./sidebar";
import { TelemetryStrip } from "./telemetry";

/**
 * Root layout. Three persistent zones:
 *
 *   - Sidebar pinned on the left (workspace nav + compute panel).
 *   - Telemetry strip pinned at the top of the content area (the
 *     visual heartbeat of halo-forge — present on every route, never
 *     not there).
 *   - Scrollable content slot; each route is responsible for its own
 *     Topbar so per-page headers live next to the data they describe.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-fg">
      <AuthRedirector />
      <Sidebar />
      {/* `id="main"` is the skip-to-content target. `tabIndex={-1}` lets
          focus land here programmatically when the skip link fires
          without making it tab-stop in the normal flow. */}
      <main
        id="main"
        tabIndex={-1}
        className="flex flex-1 flex-col overflow-hidden focus:outline-none"
      >
        <MobileNav />
        <TelemetryStrip />
        <div className="flex-1 overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}

export { Topbar } from "./topbar";

function MobileNav() {
  return (
    <div className="flex items-center gap-2 border-b border-border bg-bg-subtle px-3 py-2 md:hidden">
      <Link to="/" className="mr-auto flex items-center gap-2 text-[13px] font-medium text-fg">
        <img src="/mark.svg" alt="" width={18} height={18} />
        halo-forge
      </Link>
      <MobileNavLink to="/start" search={{ goal: undefined }}>Start</MobileNavLink>
      <MobileNavLink to="/runs">Runs</MobileNavLink>
      <MobileNavLink to="/connect">Connect</MobileNavLink>
    </div>
  );
}

function MobileNavLink({
  to,
  search,
  children,
}: {
  to: "/" | "/start" | "/runs" | "/connect";
  search?: Record<string, unknown>;
  children: ReactNode;
}) {
  return (
    <Link
      to={to}
      search={search}
      className="rounded-sm px-2 py-1 text-[12px] text-fg-muted hover:bg-surface hover:text-fg"
    >
      {children}
    </Link>
  );
}

function AuthRedirector() {
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    function onAuthRequired() {
      if (pathname !== "/connect") {
        navigate({ to: "/connect", search: { from: pathname } });
      }
    }
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, [navigate, pathname]);

  return null;
}
