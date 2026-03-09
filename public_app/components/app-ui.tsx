"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { PropsWithChildren, type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type ShellStatus = {
  label: string;
  value: string;
  tone?: "success" | "warning" | "danger" | "neutral" | string;
};

type AppShellProps = PropsWithChildren<{
  title?: string;
  subtitle?: string;
  statusItems?: ShellStatus[];
  headerActions?: ReactNode;
}>;

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/train", label: "Training" },
  { href: "/results", label: "Results" },
  { href: "/readiness", label: "Readiness" },
  { href: "/docs", label: "Docs" },
];

function toneToBadgeVariant(tone?: string) {
  switch (tone) {
    case "success":
      return "success" as const;
    case "warning":
      return "warning" as const;
    case "danger":
      return "danger" as const;
    default:
      return "secondary" as const;
  }
}

export function AppShell({
  children,
  title = "Overview",
  subtitle,
  statusItems = [],
  headerActions,
}: AppShellProps) {
  const pathname = usePathname();

  return (
    <div className="grid grid-cols-[220px_minmax(0,1fr)] min-h-screen">
      <aside className="sticky top-0 h-screen border-r border-border bg-zinc-950 flex flex-col px-3 py-4">
        <div className="px-3 mb-6">
          <div className="text-sm font-semibold text-foreground">Halo-Forge</div>
          <div className="text-xs text-muted-foreground mt-0.5">Training platform</div>
        </div>

        <nav className="flex flex-col gap-0.5">
          {NAV_ITEMS.map((item) => {
            const active =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(`${item.href}/`));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center h-8 px-3 rounded-md text-sm transition-colors",
                  active
                    ? "bg-accent text-foreground font-medium border-l-2 border-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>

      <div className="flex flex-col min-h-screen">
        <header className="flex items-center justify-end gap-2 px-6 py-3 border-b border-border">
          {statusItems.map((item) => (
            <div key={`${item.label}-${item.value}`} className="flex items-center gap-1.5">
              <span className="text-xs text-muted-foreground">{item.label}</span>
              <Badge variant={toneToBadgeVariant(item.tone)}>{item.value}</Badge>
            </div>
          ))}
        </header>

        <div className="flex-1 px-6 py-5">
          <div className="flex items-start justify-between mb-5">
            <div>
              <h1 className="text-xl font-semibold text-foreground">{title}</h1>
              {subtitle ? (
                <p className="text-sm text-muted-foreground mt-1 max-w-2xl">{subtitle}</p>
              ) : null}
            </div>
            {headerActions ? <div className="flex gap-2 shrink-0">{headerActions}</div> : null}
          </div>

          <main className="space-y-4">{children}</main>
        </div>
      </div>
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  actions,
  eyebrow,
  className,
  children,
}: PropsWithChildren<{
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  eyebrow?: string;
  className?: string;
}>) {
  return (
    <section className={cn("rounded-lg border border-border bg-card p-4", className)}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          {eyebrow ? (
            <div className="text-xs font-medium text-muted-foreground mb-1">{eyebrow}</div>
          ) : null}
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          {subtitle ? (
            <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
          ) : null}
        </div>
        {actions ? <div className="flex gap-2 shrink-0">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function StatusBadge({
  tone,
  label,
}: {
  tone: "success" | "warning" | "danger" | "neutral" | string;
  label: string;
}) {
  return <Badge variant={toneToBadgeVariant(tone)}>{label}</Badge>;
}

export function MetricRow({
  label,
  value,
  meta,
}: {
  label: string;
  value: string;
  meta?: string;
}) {
  return (
    <div className="flex items-baseline justify-between px-3 py-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="text-right">
        <span className="text-sm font-medium text-foreground">{value}</span>
        {meta ? (
          <span className="block text-xs text-muted-foreground">{meta}</span>
        ) : null}
      </div>
    </div>
  );
}

export function ActionLink({
  href,
  label,
  tone = "secondary",
}: {
  href: string;
  label: string;
  tone?: "primary" | "secondary";
}) {
  return (
    <Link
      href={href}
      className={cn(
        "inline-flex items-center justify-center h-8 px-3 rounded-md text-sm font-medium transition-colors",
        tone === "primary"
          ? "bg-primary text-primary-foreground hover:bg-primary/90"
          : "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
      )}
    >
      {label}
    </Link>
  );
}

export function ActionButton({
  label,
  tone = "secondary",
  onClick,
  disabled = false,
}: {
  label: string;
  tone?: "primary" | "secondary";
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center h-8 px-3 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none",
        tone === "primary"
          ? "bg-primary text-primary-foreground hover:bg-primary/90"
          : "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
      )}
      onClick={onClick}
      disabled={disabled}
      type="button"
    >
      {label}
    </button>
  );
}

export function Callout({
  title,
  body,
  tone = "neutral",
  actions,
}: {
  title: string;
  body: string;
  tone?: "success" | "warning" | "danger" | "neutral";
  actions?: ReactNode;
}) {
  const borderColor = {
    success: "border-l-emerald-500",
    warning: "border-l-amber-500",
    danger: "border-l-red-500",
    neutral: "border-l-border",
  }[tone];

  return (
    <div className={cn("rounded-md border border-border bg-muted/30 p-3 border-l-2", borderColor)}>
      <h3 className="text-sm font-medium text-foreground">{title}</h3>
      <p className="text-xs text-muted-foreground mt-1">{body}</p>
      {actions ? <div className="flex gap-2 mt-2">{actions}</div> : null}
    </div>
  );
}

export function ResearchSection({
  title,
  summary,
  children,
  defaultOpen = false,
}: PropsWithChildren<{ title: string; summary: string; defaultOpen?: boolean }>) {
  return (
    <details
      className="rounded-md border border-border bg-card group"
      open={defaultOpen}
    >
      <summary className="flex items-center justify-between gap-3 px-3 py-2.5 cursor-pointer list-none [&::-webkit-details-marker]:hidden">
        <div>
          <span className="text-sm font-medium text-foreground">{title}</span>
          <span className="block text-xs text-muted-foreground mt-0.5">{summary}</span>
        </div>
        <span className="text-xs font-medium text-primary shrink-0">Details</span>
      </summary>
      <div className="px-3 pb-3 border-t border-border pt-2">{children}</div>
    </details>
  );
}

export function EmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="rounded-md border border-dashed border-border bg-muted/20 px-4 py-6 text-center">
      <h3 className="text-sm font-medium text-foreground">{title}</h3>
      <p className="text-xs text-muted-foreground mt-1">{body}</p>
    </div>
  );
}
