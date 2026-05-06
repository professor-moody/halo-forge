import type { ReactNode } from "react";
import { Sidebar } from "./sidebar";

/**
 * The single root layout for every authenticated route. Sidebar pinned on
 * the left, scrollable content on the right. Each route is responsible for
 * its own Topbar so per-page headers, status rows, and CTAs live next to
 * the page-specific data they describe.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-fg">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}

export { Topbar } from "./topbar";
