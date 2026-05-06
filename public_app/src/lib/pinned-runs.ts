import { useEffect, useSyncExternalStore } from "react";

/**
 * Pinned-runs store — the tray that powers the comparison surface.
 *
 * Pure client-side state, persisted to localStorage. No backend changes
 * required: the comparison route fetches each pinned run by id from
 * `/api/public/runs/{id}` and overlays the cycle_metrics arrays.
 *
 * State shape: ordered list of run ids (insertion order = chart series
 * order, so pinning A then B means A is series-0). Cap at 6 runs to keep
 * the comparison legible — past that, charts get illegible and the
 * config diff becomes a wall of columns.
 *
 * useSyncExternalStore is the React 18+ way to subscribe components
 * to a store that lives outside React state without tearing — perfect
 * for "this localStorage entry just changed, re-render anyone reading".
 */

const KEY = "halo-forge:pinned-runs";
const MAX_PINNED = 6;

type Listener = () => void;
const listeners = new Set<Listener>();

function readStorage(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((v) => typeof v === "string" && v.length > 0)
      .slice(0, MAX_PINNED);
  } catch {
    return [];
  }
}

function writeStorage(ids: string[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(ids));
  } catch {
    // localStorage may throw in private mode / quota exceeded. Silent
    // failure is acceptable here — pin state is convenience, not data.
  }
}

function notify() {
  for (const l of listeners) l();
}

let cachedSnapshot: string[] = readStorage();

function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): string[] {
  return cachedSnapshot;
}

function getServerSnapshot(): string[] {
  // SSR isn't a target for halo-forge but useSyncExternalStore needs
  // this for completeness; an empty array is the safe default.
  return [];
}

/** Subscribe to the pinned-runs list. Re-renders when it changes. */
export function usePinnedRuns(): string[] {
  const ids = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // Keep us in sync with localStorage edits from other tabs (rare for
  // a desktop training tool, but cheap to support).
  useEffect(() => {
    function onStorage(e: StorageEvent) {
      if (e.key !== KEY) return;
      cachedSnapshot = readStorage();
      notify();
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return ids;
}

export function pinRun(runId: string): void {
  if (!runId) return;
  if (cachedSnapshot.includes(runId)) return;
  cachedSnapshot = [...cachedSnapshot, runId].slice(-MAX_PINNED);
  writeStorage(cachedSnapshot);
  notify();
}

export function unpinRun(runId: string): void {
  if (!cachedSnapshot.includes(runId)) return;
  cachedSnapshot = cachedSnapshot.filter((id) => id !== runId);
  writeStorage(cachedSnapshot);
  notify();
}

export function clearPinned(): void {
  if (cachedSnapshot.length === 0) return;
  cachedSnapshot = [];
  writeStorage(cachedSnapshot);
  notify();
}

export function isPinned(runId: string): boolean {
  return cachedSnapshot.includes(runId);
}

export const PINNED_RUNS_LIMIT = MAX_PINNED;
