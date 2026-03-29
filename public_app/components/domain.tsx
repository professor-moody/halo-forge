import { type PropsWithChildren, type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

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
