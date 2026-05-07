import type { ReactNode } from "react";
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
      <Sidebar />
      {/* `id="main"` is the skip-to-content target. `tabIndex={-1}` lets
          focus land here programmatically when the skip link fires
          without making it tab-stop in the normal flow. */}
      <main
        id="main"
        tabIndex={-1}
        className="flex flex-1 flex-col overflow-hidden focus:outline-none"
      >
        <TelemetryStrip />
        <div className="flex-1 overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}

export { Topbar } from "./topbar";
