import { useEffect, useRef, useState } from "react";

/**
 * Tiny generic EventSource hook. Holds the latest parsed event and a
 * ReadyState-style status so consumers can render connecting / open /
 * closed UX consistently.
 *
 * Usage:
 *   const { data, status, error } = useEventSource<TelemetrySample>(
 *     enabled ? "/api/public/telemetry/stream" : null,
 *   );
 *
 * Pass `null` (or `undefined`) as the URL to disable. The hook closes
 * the underlying EventSource on unmount and on URL change.
 *
 * Why not a full reconnect/backoff library? The browser's EventSource
 * already retries on connection loss using the server's `retry:` hint.
 * The streaming endpoints emit `retry: 3000` so we get 3-second reopens
 * for free; no need to reinvent that.
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
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!url) {
      setStatus("idle");
      return;
    }

    setStatus("connecting");
    setError(null);

    const es = new EventSource(url);
    sourceRef.current = es;

    es.onopen = () => {
      setStatus("open");
      setError(null);
    };

    es.onmessage = (e) => {
      try {
        const parsed = JSON.parse(e.data) as T;
        setData(parsed);
      } catch {
        // Non-JSON event payload — ignore silently. The streaming
        // endpoints we ship always emit JSON, but a stray retry: line
        // could trip parse failure on some clients.
      }
    };

    es.onerror = () => {
      // EventSource will auto-retry; reflect the transient state but
      // don't tear down — the browser handles reconnection.
      setStatus("error");
      setError("Connection lost (auto-reconnecting…)");
    };

    return () => {
      es.close();
      sourceRef.current = null;
      setStatus("closed");
    };
  }, [url]);

  return { data, status, error };
}
