import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Tailwind class composer. Use this everywhere instead of bare template
 * strings so conflicting utilities (`p-4 p-6`) merge predictably.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format an ISO timestamp as a compact relative time ("3m ago", "2h ago").
 * Falls back to the formatted date for anything older than 7 days.
 */
export function relativeTime(input: string | number | Date | null | undefined): string {
  if (!input) return "—";
  const ts = typeof input === "string" || typeof input === "number" ? new Date(input) : input;
  const diff = Date.now() - ts.getTime();
  if (diff < 0) return "just now";
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  return ts.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/** Compact number formatter ("1.2K", "3.4M"). Used for metric pills. */
export function compactNumber(value: number | null | undefined, fractionDigits = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: fractionDigits,
  }).format(value);
}
