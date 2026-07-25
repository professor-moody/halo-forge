import { createFileRoute, redirect } from "@tanstack/react-router";

/** Legacy deep link retained for compatibility. Completed runs live in Runs. */
export const Route = createFileRoute("/results")({
  beforeLoad: () => {
    throw redirect({ to: "/runs", search: { view: "completed" }, replace: true });
  },
  component: LegacyResultsRedirect,
});

function LegacyResultsRedirect() {
  return null;
}
