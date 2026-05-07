import { createRootRoute, Outlet } from "@tanstack/react-router";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppShell } from "@/components/shell";

/**
 * Root route. Mounts the AppShell (sidebar + content slot) once and
 * delegates Topbar rendering to each child route, so per-page headers
 * live next to the data they describe.
 *
 * The `<a class="skip-to-content">` is the first focusable element on
 * the page; keyboard users hitting Tab once get a "Skip to main content"
 * link that jumps past the sidebar to `#main`. Visually hidden until
 * focus-visible, then slides in from the top edge in copper.
 */
export const Route = createRootRoute({
  component: () => (
    <TooltipProvider delayDuration={150}>
      <a href="#main" className="skip-to-content">
        Skip to main content
      </a>
      <AppShell>
        <Outlet />
      </AppShell>
    </TooltipProvider>
  ),
});
