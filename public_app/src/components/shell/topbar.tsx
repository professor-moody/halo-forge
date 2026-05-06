import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * Page header. Eyebrow + title on the left, actions on the right, with
 * an optional status row pinned underneath. Sticky with backdrop blur
 * so dense scrolling content doesn't lose its anchor.
 *
 * Visual rules:
 *   - the title is *the* page identifier — set in IBM Plex Sans
 *     SemiBold at 16px with -0.01em tracking. No taglines, no
 *     decorative subtitles unless they carry actual information
 *     (last-updated timestamp, run id, etc.).
 *   - the status bar below the title is mono — operators read these
 *     values like instrument readouts.
 */
export interface TopbarProps {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  /** Status row pinned below the title (chips, last-updated, etc.) — rendered in mono. */
  statusBar?: ReactNode;
  /** Render a 1px copper pulse along the bottom — signals "live updates flowing". */
  live?: boolean;
  className?: string;
}

export function Topbar({
  eyebrow,
  title,
  subtitle,
  actions,
  statusBar,
  live,
  className,
}: TopbarProps) {
  return (
    <header
      className={cn(
        "sticky top-0 z-20 border-b border-border bg-bg-subtle/80 backdrop-blur-sm",
        className,
      )}
    >
      <div className="flex min-h-12 items-start justify-between gap-4 px-5 py-2.5">
        <div className="min-w-0 flex-1">
          {eyebrow ? (
            <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled mb-0.5">
              {eyebrow}
            </div>
          ) : null}
          <h1 className="truncate text-base font-semibold tracking-tight text-fg leading-tight">
            {title}
          </h1>
          {subtitle ? (
            <p className="text-xs text-fg-muted mt-0.5">{subtitle}</p>
          ) : null}
        </div>
        {actions ? <div className="flex items-center gap-1.5">{actions}</div> : null}
      </div>
      {statusBar ? (
        <div className="flex items-center gap-3 border-t border-border-subtle px-5 py-1.5 font-mono text-[11px] text-fg-muted">
          {statusBar}
        </div>
      ) : null}
      {live ? (
        <div
          aria-hidden
          className="pulse-accent absolute bottom-0 left-0 right-0 h-px bg-accent"
        />
      ) : null}
    </header>
  );
}
