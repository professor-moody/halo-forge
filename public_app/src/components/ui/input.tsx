import { type InputHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

/**
 * Workstation-style input. Plex Mono numerics, 1px borders, copper focus
 * ring. Sized to match the Button md height (9 = 36px) so they line up
 * cleanly when adjacent in a form row.
 */
export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Render the value in mono — use for any numeric / identifier input. */
  mono?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", mono, ...props }, ref) => (
    <input
      ref={ref}
      type={type}
      className={cn(
        "flex h-9 w-full rounded-md border border-border bg-bg-subtle px-3 py-1 text-sm text-fg",
        "placeholder:text-fg-disabled",
        "transition-colors hover:border-border-strong",
        "focus-visible:outline-none focus-visible:border-accent focus-visible:ring-2 focus-visible:ring-accent/30",
        "disabled:cursor-not-allowed disabled:opacity-50",
        "file:border-0 file:bg-transparent file:text-sm file:font-medium",
        mono && "font-mono tabular-nums",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";
