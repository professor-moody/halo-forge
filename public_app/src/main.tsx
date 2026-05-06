import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { createRouter, RouterProvider } from "@tanstack/react-router";

import { routeTree } from "./routeTree.gen";
import "./styles/globals.css";

// Geist + Geist Mono are loaded via Google Fonts in index.html.
// globals.css references them by family name (--font-sans / --font-mono).

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // halo-forge runs are long-lived; staleness in the UI is fine for
      // a few seconds. Aggressive refetch on focus + reconnect would
      // make a long-running training tab thrash on every alt-tab.
      staleTime: 5_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("#root missing in index.html");

createRoot(rootEl).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      {import.meta.env.DEV ? <ReactQueryDevtools buttonPosition="bottom-left" /> : null}
    </QueryClientProvider>
  </StrictMode>,
);
