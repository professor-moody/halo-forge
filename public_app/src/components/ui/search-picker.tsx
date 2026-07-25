import { Check, ChevronsUpDown, Search } from "lucide-react";
import { useId, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export type SearchPickerOption = {
  value: string;
  label: string;
  description?: string;
  status?: string;
  keywords?: string;
};

export function SearchPicker({
  value,
  options,
  onChange,
  placeholder = "Search and choose",
  emptyLabel = "No compatible options",
  disabled,
  allowEmpty,
  ariaLabel,
}: {
  value: string;
  options: SearchPickerOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  emptyLabel?: string;
  disabled?: boolean;
  allowEmpty?: boolean;
  ariaLabel?: string;
}) {
  const selected = options.find((option) => option.value === value);
  const listboxId = useId();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const blurTimer = useRef<number | null>(null);
  const visible = useMemo(() => {
    const tokens = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!tokens.length) return options;
    return options.filter((option) => {
      const haystack = `${option.label} ${option.description ?? ""} ${option.keywords ?? ""}`.toLowerCase();
      return tokens.every((token) => haystack.includes(token));
    });
  }, [options, query]);

  function choose(next: string) {
    onChange(next);
    setOpen(false);
    setQuery("");
  }

  return (
    <div className="relative">
      <div className={cn("flex h-8 items-center rounded-md border border-border bg-surface transition-colors focus-within:border-accent focus-within:ring-1 focus-within:ring-accent", disabled && "cursor-not-allowed opacity-55")}>
        <Search className="ml-2.5 h-3.5 w-3.5 shrink-0 text-fg-disabled" />
        <input
          role="combobox"
          aria-expanded={open}
          aria-controls={listboxId}
          aria-label={ariaLabel ?? placeholder}
          disabled={disabled}
          value={open ? query : selected?.label ?? (value || "")}
          placeholder={placeholder}
          onFocus={() => { if (blurTimer.current != null) window.clearTimeout(blurTimer.current); setOpen(true); setQuery(""); setActive(0); }}
          onBlur={() => { blurTimer.current = window.setTimeout(() => setOpen(false), 120); }}
          onChange={(event) => { setQuery(event.target.value); setOpen(true); setActive(0); }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") { event.preventDefault(); setOpen(true); setActive((index) => Math.min(Math.max(0, visible.length - 1), index + 1)); }
            if (event.key === "ArrowUp") { event.preventDefault(); setActive((index) => Math.max(0, index - 1)); }
            if (event.key === "Enter" && open) { event.preventDefault(); if (visible[active]) choose(visible[active].value); }
            if (event.key === "Escape") { event.preventDefault(); setOpen(false); setQuery(""); }
          }}
          className="h-full min-w-0 flex-1 bg-transparent px-2 text-[12px] text-fg outline-none placeholder:text-fg-disabled"
        />
        <ChevronsUpDown className="mr-2 h-3.5 w-3.5 shrink-0 text-fg-disabled" />
      </div>
      {open && !disabled ? (
        <div id={listboxId} role="listbox" className="absolute z-40 mt-1 max-h-60 w-full overflow-y-auto rounded-md border border-border-strong bg-elevated p-1 shadow-xl shadow-black/25">
          {allowEmpty ? <PickerOption option={{ value: "", label: "None", description: "Use the default behavior" }} active={active === 0 && !query} selected={!value} onChoose={() => choose("")} /> : null}
          {visible.map((option, index) => <PickerOption key={option.value} option={option} active={index === active} selected={option.value === value} onChoose={() => choose(option.value)} />)}
          {!visible.length ? <div className="px-3 py-5 text-center text-[11px] text-fg-disabled">{emptyLabel}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

function PickerOption({ option, active, selected, onChoose }: { option: SearchPickerOption; active: boolean; selected: boolean; onChoose: () => void }) {
  return (
    <button type="button" role="option" aria-selected={selected} onMouseDown={(event) => event.preventDefault()} onClick={onChoose} className={cn("flex w-full items-start gap-2 rounded-sm px-2.5 py-2 text-left", active ? "bg-accent-bg" : "hover:bg-surface")}>
      <span className="min-w-0 flex-1"><span className="block truncate text-[11.5px] font-medium text-fg">{option.label}</span>{option.description ? <span className="mt-0.5 block truncate text-[9.5px] text-fg-subtle">{option.description}</span> : null}</span>
      {option.status ? <span className="mt-0.5 shrink-0 font-mono text-[8.5px] uppercase text-fg-disabled">{option.status}</span> : null}
      {selected ? <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" /> : null}
    </button>
  );
}
