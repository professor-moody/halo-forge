import { useEffect, useRef, useState } from "react";
import { getApiToken, reportAuthRequired } from "@/lib/api";

/**
 * Tiny generic SSE hook. Holds the latest parsed event and a
 * ReadyState-style status so consumers can render connecting / open /
 * closed UX consistently.
 *
 * Usage:
 *   const { data, status, error } = useEventSource<TelemetrySample>(
 *     enabled ? "/api/public/telemetry/stream" : null,
 *   );
 *
 * Pass `null` (or `undefined`) as the URL to disable. Uses fetch instead
 * of native EventSource so remote workstation tokens can ride on the
 * Authorization header.
 *
 * Why not native EventSource? It cannot attach custom headers, which
 * breaks non-loopback bearer-token auth. This small fetch reader keeps
 * the same consumer API while supporting remote access.
 */

export type StreamStatus = "idle" | "connecting" | "open" | "closed" | "error";

export interface UseEventSourceResult<T> {
  data: T | null;
  status: StreamStatus;
  error: string | null;
}

export function useEventSource<T = unknown>(
  url: string | null | undefined,
): UseEventSourceResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!url) {
      setStatus("idle");
      return;
    }

    const streamUrl = url;
    let cancelled = false;
    let retryTimer: number | undefined;
    const decoder = new TextDecoder();

    async function connect() {
      const controller = new AbortController();
      abortRef.current = controller;
      setStatus("connecting");
      setError(null);

      try {
        const token = getApiToken();
        const headers: Record<string, string> = { Accept: "text/event-stream" };
        if (token) headers.Authorization = `Bearer ${token}`;

        const res = await fetch(streamUrl, {
          headers,
          signal: controller.signal,
          cache: "no-store",
        });

        if (res.status === 401) {
          reportAuthRequired({ source: "stream", url: streamUrl });
          setStatus("error");
          setError("Remote token required.");
          return;
        }

        if (!res.ok || !res.body) {
          throw new Error(`${res.status} ${res.statusText || "stream failed"}`);
        }

        setStatus("open");
        const reader = res.body.getReader();
        let buffer = "";

        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() ?? "";
          for (const event of events) {
            const dataLines = event
              .split(/\r?\n/)
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trimStart());
            if (!dataLines.length) continue;
            try {
              setData(JSON.parse(dataLines.join("\n")) as T);
            } catch {
              // Ignore non-JSON payloads. Shipped streams emit JSON.
            }
          }
        }
      } catch (e) {
        if (cancelled || controller.signal.aborted) return;
        setStatus("error");
        setError(e instanceof Error ? e.message : "Connection lost.");
      }

      if (!cancelled) {
        retryTimer = window.setTimeout(connect, 3000);
      }
    }

    connect();

    return () => {
      cancelled = true;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      abortRef.current?.abort();
      abortRef.current = null;
      setStatus("closed");
    };
  }, [url]);

  return { data, status, error };
}
