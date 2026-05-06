import { Link, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  BookOpen,
  CheckCircle2,
  LayoutDashboard,
  Play,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { useBackendInfo } from "@/lib/hooks";

/**
 * Left rail. Three vertical zones:
 *
 *   1. Wordmark — the brand glyph + "halo-forge" set in IBM Plex Sans
 *      Medium with tight tracking. The wordmark has a subtle copper
 *      pulse on hover so the brand has *some* life without being
 *      decorative.
 *   2. Workspace nav — primary routes. Active state is a soft copper
 *      fill plus a left edge marker; left-edge markers are the
 *      workstation-app vocabulary (Logic Pro, DAWs, IDE tabs) for
 *      "this is the active document".
 *   3. Compute panel — pinned to the bottom. The single most
 *      distinctive piece of halo-forge. Shows backend, dtype, attention
 *      impl. Phase B will expand this into a full telemetry strip
 *      (memory bar, watts, throughput).
 */

type NavItem = {
  to: string;
  label: string;
  icon: LucideIcon;
  /** Keyboard hint shown on the right rail. Plain text, will be wrapped in <kbd>. */
  kbd?: string;
};

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: LayoutDashboard, kbd: "G O" },
  { to: "/train", label: "Training", icon: Play, kbd: "G T" },
  { to: "/runs", label: "Runs", icon: Activity, kbd: "G R" },
  { to: "/results", label: "Results", icon: CheckCircle2, kbd: "G S" },
  { to: "/docs", label: "Docs", icon: BookOpen, kbd: "G D" },
];

export function Sidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <aside className="flex h-screen w-56 flex-col border-r border-border bg-bg-subtle">
      {/* Wordmark */}
      <Link
        to="/"
        className="flex h-12 items-center gap-2.5 border-b border-border-subtle px-3.5 hover:bg-surface/40 transition-colors group"
      >
        <img
          src="/mark.svg"
          alt=""
          width={20}
          height={20}
          className="opacity-95 group-hover:opacity-100 transition-opacity"
        />
        <span className="font-medium tracking-tight text-fg text-[13.5px]">
          halo<span className="text-fg-subtle">-</span>forge
        </span>
      </Link>

      {/* Primary nav */}
      <nav className="flex-1 overflow-y-auto px-1.5 py-2.5">
        <SectionLabel>Workspace</SectionLabel>
        <ul className="space-y-px">
          {NAV_ITEMS.map((item) => {
            const active =
              item.to === "/"
                ? pathname === "/"
                : pathname === item.to || pathname.startsWith(`${item.to}/`);
            return (
              <li key={item.to}>
                <Link
                  to={item.to}
                  className={cn(
                    "relative flex h-7 items-center gap-2.5 rounded-sm px-2 text-[13px] transition-colors group",
                    active
                      ? "bg-accent-bg text-accent font-medium"
                      : "text-fg-muted hover:bg-surface hover:text-fg",
                  )}
                >
                  {/* Active edge marker — Logic Pro / DAW vocabulary
                      for "this is the active panel". */}
                  {active ? (
                    <span
                      aria-hidden
                      className="absolute -left-1.5 top-1.5 bottom-1.5 w-0.5 rounded-full bg-accent"
                    />
                  ) : null}
                  <item.icon className="h-3.5 w-3.5" />
                  <span className="flex-1">{item.label}</span>
                  {item.kbd ? (
                    <kbd
                      className={cn(
                        "font-mono text-[10px] tracking-tight",
                        active ? "text-accent/70" : "text-fg-disabled group-hover:text-fg-subtle",
                      )}
                    >
                      {item.kbd}
                    </kbd>
                  ) : null}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Compute panel */}
      <ComputePanel />
    </aside>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 pb-1.5 pt-1 text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">
      {children}
    </div>
  );
}

function ComputePanel() {
  const { data, isLoading, isError } = useBackendInfo();

  return (
    <div className="border-t border-border-subtle p-2.5">
      <SectionLabel>Compute</SectionLabel>
      <div className="rounded-md border border-border-subtle bg-surface/60">
        {isLoading ? (
          <div className="h-12 animate-pulse" />
        ) : isError || !data ? (
          <div className="px-2.5 py-2 flex items-center gap-2">
            <Badge tone="danger" dot size="sm">
              Offline
            </Badge>
          </div>
        ) : (
          <div className="px-2.5 py-2 space-y-1.5">
            {/* Top row: backend identity */}
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-[11px] text-fg">
                {prettyBackendName(data.name)}
              </span>
              <span
                aria-label="Online"
                className="status-dot"
                style={{
                  background:
                    data.name === "mlx"
                      ? "var(--color-accent)"
                      : data.name.startsWith("rocm")
                        ? "var(--color-success)"
                        : "var(--color-info)",
                }}
              />
            </div>
            {/* Bottom row: dtype + attention */}
            <div className="flex items-center justify-between gap-2 font-mono text-[10px] text-fg-subtle uppercase tracking-wider">
              <span>{data.capabilities.preferred_dtype_str}</span>
              <span>{data.capabilities.preferred_attn_impl}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function prettyBackendName(name: string): string {
  switch (name) {
    case "rocm_gfx1151":
      return "ROCm · gfx1151";
    case "rocm":
      return "ROCm";
    case "cuda":
      return "CUDA";
    case "mps":
      return "Apple · MPS";
    case "mlx":
      return "Apple · MLX";
    case "cpu":
      return "CPU";
    default:
      return name;
  }
}
