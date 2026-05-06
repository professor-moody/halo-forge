import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/**
 * Status pill. `tone` covers the standard dashboard cases; for arbitrary
 * coloring drop down to plain Tailwind classes instead of bolting on
 * another variant.
 */
const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      tone: {
        neutral: "border-border bg-bg-subtle text-fg-muted",
        accent: "border-accent/30 bg-accent/10 text-accent",
        success: "border-success/30 bg-success-bg text-success",
        warning: "border-warning/30 bg-warning-bg text-warning",
        danger: "border-danger/30 bg-danger-bg text-danger",
        info: "border-info/30 bg-info-bg text-info",
      },
      size: {
        sm: "h-5 px-1.5 text-[11px]",
        md: "h-6 px-2 text-xs",
      },
    },
    defaultVariants: {
      tone: "neutral",
      size: "md",
    },
  },
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  /** Optional leading dot — used in run status pills. */
  dot?: boolean;
}

export function Badge({ className, tone, size, dot, children, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ tone, size }), className)} {...props}>
      {dot ? (
        <span
          className="status-dot"
          style={{
            background:
              tone === "success"
                ? "var(--color-success)"
                : tone === "warning"
                  ? "var(--color-warning)"
                  : tone === "danger"
                    ? "var(--color-danger)"
                    : tone === "info"
                      ? "var(--color-info)"
                      : tone === "accent"
                        ? "var(--color-accent)"
                        : "var(--color-fg-subtle)",
          }}
        />
      ) : null}
      {children}
    </span>
  );
}

export { badgeVariants };
