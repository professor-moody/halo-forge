import { createFileRoute, redirect } from "@tanstack/react-router";

/** Legacy deep link retained for compatibility. The supported surface is Train. */
export const Route = createFileRoute("/start")({
  validateSearch: (search: Record<string, unknown>) => ({
    goal: typeof search.goal === "string" ? search.goal : undefined,
  }),
  beforeLoad: ({ search }) => {
    throw redirect({
      to: "/train",
      search: { goal: search.goal },
      replace: true,
    });
  },
  component: LegacyStartRedirect,
});

function LegacyStartRedirect() {
  return null;
}
