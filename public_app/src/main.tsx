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

installChunkLoadRecovery();

createRoot(rootEl).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      {import.meta.env.DEV ? <ReactQueryDevtools buttonPosition="bottom-left" /> : null}
    </QueryClientProvider>
  </StrictMode>,
);

function installChunkLoadRecovery() {
  if (typeof window === "undefined") return;
  let shown = false;

  const showRecovery = () => {
    if (shown) return;
    shown = true;
    const overlay = document.createElement("div");
    overlay.setAttribute("role", "alert");
    overlay.style.cssText = [
      "position:fixed",
      "inset:0",
      "z-index:99999",
      "display:grid",
      "place-items:center",
      "background:rgba(5,3,2,.82)",
      "color:#f6ede7",
      "font:14px system-ui,-apple-system,BlinkMacSystemFont,sans-serif",
    ].join(";");
    overlay.innerHTML = `
      <div style="max-width:420px;border:1px solid rgba(255,255,255,.16);background:#120d0a;padding:20px;border-radius:8px">
        <div style="font-weight:650;font-size:16px">Dashboard update ready</div>
        <div style="margin-top:8px;color:#b7aba3;line-height:1.45">
          A cached dashboard chunk is stale after a rebuild. Reload to pick up the current assets.
        </div>
        <button type="button" style="margin-top:16px;border:1px solid #f97316;background:#7c2d12;color:#fff;padding:8px 12px;border-radius:4px;font-weight:650">
          Reload dashboard
        </button>
      </div>
    `;
    overlay.querySelector("button")?.addEventListener("click", () => window.location.reload());
    document.body.appendChild(overlay);
  };

  const isChunkIssue = (value: unknown) => {
    const text = String(value instanceof Error ? value.message : value ?? "");
    return (
      text.includes("Failed to fetch dynamically imported module") ||
      text.includes("Importing a module script failed") ||
      text.includes("error loading dynamically imported module") ||
      text.includes("/assets/")
    );
  };

  window.addEventListener("error", (event) => {
    const target = event.target as HTMLScriptElement | HTMLLinkElement | null;
    const src = "src" in (target ?? {}) ? (target as HTMLScriptElement).src : "";
    const href = "href" in (target ?? {}) ? (target as HTMLLinkElement).href : "";
    if (isChunkIssue(event.error) || src.includes("/assets/") || href.includes("/assets/")) {
      showRecovery();
    }
  }, true);
  window.addEventListener("unhandledrejection", (event) => {
    if (isChunkIssue(event.reason)) showRecovery();
  });
}
