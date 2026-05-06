import { createRootRoute, Outlet } from "@tanstack/react-router";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppShell } from "@/components/shell";

/**
 * Root route. Mounts the AppShell (sidebar + content slot) once and
 * delegates Topbar rendering to each child route, so per-page headers
 * live next to the data they describe.
 */
export const Route = createRootRoute({
  component: () => (
    <TooltipProvider delayDuration={150}>
      <AppShell>
        <Outlet />
      </AppShell>
    </TooltipProvider>
  ),
});
