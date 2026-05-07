/**
 * Theme handling (Track F-N).
 *
 * Two modes plus auto:
 *   - "auto"  follow the OS `prefers-color-scheme` media query
 *   - "dark"  force dark (the default workstation look)
 *   - "light" force light (off-white + darker copper)
 *
 * The chosen mode persists to `localStorage`; a tiny applyTheme()
 * sets `data-theme` on <html> so the CSS in `globals.css` picks
 * it up via `:root[data-theme="light"]` overrides.
 *
 * On first paint we apply the persisted choice synchronously from
 * a small bootstrap script in `index.html` to avoid a dark→light
 * flicker. The React component owns *changes* after mount.
 */

import { useEffect, useSyncExternalStore } from "react";

export type ThemeMode = "auto" | "dark" | "light";
const STORAGE_KEY = "halo-forge:theme";

const listeners = new Set<() => void>();

function notify() {
  for (const l of listeners) l();
}

function readMode(): ThemeMode {
  if (typeof window === "undefined") return "dark";
  const v = window.localStorage.getItem(STORAGE_KEY);
  if (v === "light" || v === "dark" || v === "auto") return v;
  return "dark";
}

function osPrefersLight(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ?? false;
}

/** Resolve the effective scheme given a mode. */
export function effectiveScheme(mode: ThemeMode): "dark" | "light" {
  if (mode === "auto") return osPrefersLight() ? "light" : "dark";
  return mode;
}

/** Apply a mode immediately to the DOM (sets `data-theme` on <html>). */
export function applyTheme(mode: ThemeMode): void {
  if (typeof document === "undefined") return;
  const scheme = effectiveScheme(mode);
  document.documentElement.dataset.theme = scheme;
  // The legacy `class="dark"` on <html> stays for any third-party
  // component checking it; we keep it in sync with the resolved scheme.
  document.documentElement.classList.toggle("dark", scheme === "dark");
  document.documentElement.classList.toggle("light", scheme === "light");
}

export function setThemeMode(mode: ThemeMode): void {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, mode);
  }
  applyTheme(mode);
  notify();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): ThemeMode {
  return readMode();
}

function getServerSnapshot(): ThemeMode {
  return "dark";
}

/**
 * React hook returning the current theme mode and a setter. Subscribes
 * to OS scheme changes when in "auto" mode so flipping the system
 * preference takes effect without a reload.
 */
export function useTheme(): {
  mode: ThemeMode;
  scheme: "dark" | "light";
  setMode: (m: ThemeMode) => void;
} {
  const mode = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // Re-apply when the OS pref changes under "auto".
  useEffect(() => {
    if (mode !== "auto") return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    function handler() {
      applyTheme("auto");
    }
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [mode]);

  return {
    mode,
    scheme: effectiveScheme(mode),
    setMode: setThemeMode,
  };
}
