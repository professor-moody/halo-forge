import { createFileRoute, Outlet } from "@tanstack/react-router";

/**
 * Layout route for `/runs/...`. Tanstack Router's dot-naming convention
 * treats `runs.$runId.tsx` as a child of `runs.tsx`, so we need this
 * file to render the Outlet — without it the child detail route never
 * mounts and `/runs/$id` falls back to the parent path.
 *
 * The actual list view lives in `runs.index.tsx` so siblings render
 * cleanly: `/runs` → list, `/runs/$id` → detail.
 */
export const Route = createFileRoute("/runs")({
  component: () => <Outlet />,
});
