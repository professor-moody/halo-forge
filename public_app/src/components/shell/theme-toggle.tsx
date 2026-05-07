import { Monitor, Moon, Sun } from "lucide-react";
import { type ThemeMode, useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/**
 * 3-way segmented control: Auto / Light / Dark.
 *
 * Lives at the bottom of the sidebar above the Compute panel. The control
 * itself is a `<fieldset role="radiogroup">` so screen readers announce
 * "Theme — radio group, 3 options". Each segment is a real radio input
 * (visually hidden) wrapped in a `<label>` for native keyboard support
 * (arrow keys move between options on macOS/Windows).
 */

type Option = {
  mode: ThemeMode;
  label: string;
  Icon: typeof Sun;
};

const OPTIONS: Option[] = [
  { mode: "auto", label: "Auto", Icon: Monitor },
  { mode: "light", label: "Light", Icon: Sun },
  { mode: "dark", label: "Dark", Icon: Moon },
];

export function ThemeToggle() {
  const { mode, setMode } = useTheme();

  return (
    <fieldset
      className="border-0 p-0 m-0"
      aria-label="Theme"
    >
      <legend className="sr-only">Theme</legend>
      <div
        role="radiogroup"
        aria-label="Theme"
        className="grid grid-cols-3 gap-px rounded-md border border-border-subtle bg-border-subtle p-px"
      >
        {OPTIONS.map(({ mode: m, label, Icon }) => {
          const active = mode === m;
          return (
            <Tooltip key={m}>
              <TooltipTrigger asChild>
                <label
                  className={cn(
                    "flex h-7 cursor-pointer items-center justify-center rounded-[4px] transition-colors",
                    "focus-within:ring-2 focus-within:ring-accent focus-within:ring-offset-1 focus-within:ring-offset-bg-subtle",
                    active
                      ? "bg-accent-bg text-accent"
                      : "bg-surface/60 text-fg-muted hover:bg-surface hover:text-fg",
                  )}
                >
                  <input
                    type="radio"
                    name="halo-forge-theme"
                    value={m}
                    checked={active}
                    onChange={() => setMode(m)}
                    className="sr-only"
                    aria-label={label}
                  />
                  <Icon className="h-3.5 w-3.5" aria-hidden />
                </label>
              </TooltipTrigger>
              <TooltipContent side="top">{label}</TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </fieldset>
  );
}
