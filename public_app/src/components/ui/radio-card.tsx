import * as RadioGroupPrimitive from "@radix-ui/react-radio-group";
import {
  type ComponentPropsWithoutRef,
  type ElementRef,
  type ReactNode,
  forwardRef,
} from "react";
import { cn } from "@/lib/utils";

/**
 * Radio cards for binary/few-choice config selections (modality, verifier).
 *
 * Workstation aesthetic: not pill-shaped buttons, not generic radio dots.
 * Each option is a small panel with title + optional description; the
 * selected card gets a copper border + accent fill, all others stay
 * neutral. The actual radio input is hidden but accessible.
 */

export const RadioCardGroup = forwardRef<
  ElementRef<typeof RadioGroupPrimitive.Root>,
  ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Root>
>(({ className, ...props }, ref) => (
  <RadioGroupPrimitive.Root
    ref={ref}
    className={cn("grid gap-2", className)}
    {...props}
  />
));
RadioCardGroup.displayName = "RadioCardGroup";

export interface RadioCardProps
  extends ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Item> {
  title: string;
  description?: ReactNode;
  badge?: ReactNode;
}

export const RadioCard = forwardRef<
  ElementRef<typeof RadioGroupPrimitive.Item>,
  RadioCardProps
>(({ className, title, description, badge, ...props }, ref) => (
  <RadioGroupPrimitive.Item
    ref={ref}
    className={cn(
      "group relative flex flex-col items-start gap-1 rounded-md border bg-surface px-3.5 py-3 text-left",
      "border-border hover:border-border-strong",
      "data-[state=checked]:border-accent data-[state=checked]:bg-accent-bg/40",
      "transition-colors text-sm",
      "focus-visible:outline-none focus-visible:border-accent focus-visible:ring-2 focus-visible:ring-accent/30",
      "disabled:opacity-50 disabled:cursor-not-allowed",
      className,
    )}
    {...props}
  >
    <div className="flex w-full items-center justify-between gap-2">
      <span className="font-medium text-fg group-data-[state=checked]:text-accent">
        {title}
      </span>
      {badge}
    </div>
    {description ? (
      <span className="text-[11.5px] text-fg-muted leading-relaxed">{description}</span>
    ) : null}
    {/* Indicator — small copper square in the top-right corner of the
        selected card. Hidden when unchecked. */}
    <RadioGroupPrimitive.Indicator className="absolute right-3 top-3 h-1.5 w-1.5 rounded-sm bg-accent" />
  </RadioGroupPrimitive.Item>
));
RadioCard.displayName = "RadioCard";
