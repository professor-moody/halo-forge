import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { type ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

/**
 * Button variants are deliberately limited:
 *   primary  – the single CTA per page (run, save, launch). Accent fill.
 *   secondary – default action (cancel, view, navigate). Surface fill.
 *   ghost    – inline / icon actions inside dense rows.
 *   danger   – destructive (delete, abort).
 *
 * If you find yourself reaching for a fifth variant, the page probably
 * has too many competing CTAs — fix the IA first.
 */
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-bg transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary:
          "bg-accent text-accent-fg hover:bg-accent-hover active:translate-y-px shadow-sm shadow-accent/20",
        secondary:
          "bg-surface text-fg border border-border hover:bg-surface-hover hover:border-border-strong active:bg-surface-pressed",
        ghost:
          "text-fg-muted hover:bg-surface hover:text-fg active:bg-surface-pressed",
        danger:
          "bg-danger/90 text-white hover:bg-danger active:translate-y-px",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-9 px-3.5",
        lg: "h-10 px-4 text-sm",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { buttonVariants };
