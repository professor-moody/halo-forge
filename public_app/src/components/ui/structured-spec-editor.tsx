import { useMutation } from "@tanstack/react-query";
import { Braces, Check, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type SpecDescriptor, type SpecFieldDescriptor } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

type StructuredSpecEditorProps = {
  descriptor: SpecDescriptor;
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
  className?: string;
  validateRemotely?: boolean;
};

export function StructuredSpecEditor({ descriptor, value, onChange, className, validateRemotely = true }: StructuredSpecEditorProps) {
  const [advanced, setAdvanced] = useState(false);
  const [raw, setRaw] = useState(() => JSON.stringify(value, null, 2));
  const [rawError, setRawError] = useState<string | null>(null);
  const validation = useMutation({
    mutationFn: () => api.validateSpecDescriptor(descriptor.kind, descriptor.id, value),
  });

  useEffect(() => {
    if (!advanced) setRaw(JSON.stringify(value, null, 2));
  }, [advanced, value]);

  function update(name: string, next: unknown) {
    onChange({ ...value, [name]: next });
    validation.reset();
  }

  function applyRaw(next: string) {
    setRaw(next);
    try {
      const parsed = JSON.parse(next) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Use a JSON object.");
      onChange(parsed as Record<string, unknown>);
      setRawError(null);
    } catch (error) {
      setRawError(error instanceof Error ? error.message : "Invalid JSON");
    }
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11.5px] font-medium text-fg">{descriptor.label}</div>
          {descriptor.description ? <p className="mt-0.5 max-w-2xl text-[10.5px] leading-4 text-fg-subtle">{descriptor.description}</p> : null}
        </div>
        <Button type="button" size="sm" variant="ghost" onClick={() => setAdvanced((open) => !open)} aria-expanded={advanced}>
          <Braces /> {advanced ? "Form" : "Advanced"}
        </Button>
      </div>

      {advanced ? (
        <div>
          <textarea
            aria-label={`${descriptor.label} JSON`}
            value={raw}
            onChange={(event) => applyRaw(event.target.value)}
            rows={Math.max(6, descriptor.fields.length * 2)}
            spellCheck={false}
            className="w-full resize-y rounded-md border border-border bg-bg px-3 py-2 font-mono text-[10.5px] leading-5 text-fg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
          />
          {rawError ? <p role="alert" className="mt-1 text-[10px] text-danger">{rawError}</p> : null}
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {descriptor.fields.filter((field) => isVisible(field, value)).map((field) => (
            <SpecField key={field.name} field={field} value={value[field.name] ?? field.default} onChange={(next) => update(field.name, next)} />
          ))}
        </div>
      )}

      {validateRemotely ? (
        <div className="flex flex-wrap items-center gap-2 border-t border-border-subtle pt-2">
          <Button type="button" size="sm" variant="ghost" disabled={validation.isPending || Boolean(rawError)} onClick={() => validation.mutate()}>
            {validation.isPending ? <Loader2 className="animate-spin" /> : validation.data?.valid ? <Check /> : null}
            {validation.data?.valid ? "Validated" : "Check configuration"}
          </Button>
          {validation.data && !validation.data.valid ? <span role="alert" className="text-[10px] text-danger">{validation.data.errors.map((error) => error.message).join(" · ")}</span> : null}
          {validation.isError ? <span role="alert" className="text-[10px] text-danger">{validation.error.message}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function SpecField({ field, value, onChange }: { field: SpecFieldDescriptor; value: unknown; onChange: (value: unknown) => void }) {
  const options = (field.options ?? []).map((option) => typeof option === "string" ? { value: option, label: humanize(option) } : { value: option.value, label: option.label ?? humanize(option.value) });
  const complex = ["field_mapping", "json", "object", "array", "list"].includes(field.value_type);
  const [complexText, setComplexText] = useState(() => JSON.stringify(value ?? (field.value_type === "array" || field.value_type === "list" ? [] : {}), null, 2));
  const [complexError, setComplexError] = useState<string | null>(null);

  useEffect(() => {
    if (complex) setComplexText(JSON.stringify(value ?? {}, null, 2));
  }, [complex, value]);

  return (
    <div className={cn("space-y-1.5", complex && "sm:col-span-2")}>
      <Label htmlFor={`spec-${field.name}`}>{field.label}{field.required ? <span className="ml-1 text-accent" aria-label="required">*</span> : null}</Label>
      {field.value_type === "boolean" ? (
        <label className="flex min-h-8 items-center gap-2 rounded-md border border-border bg-surface px-2.5 text-[11px] text-fg-muted">
          <input id={`spec-${field.name}`} type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
          Enabled
        </label>
      ) : field.value_type === "select" || options.length ? (
        <select id={`spec-${field.name}`} value={String(value ?? "")} onChange={(event) => onChange(event.target.value)} className={fieldClass}>
          <option value="">Choose {field.label.toLowerCase()}</option>
          {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      ) : complex ? (
        <>
          <textarea id={`spec-${field.name}`} value={complexText} rows={4} onChange={(event) => {
            const next = event.target.value;
            setComplexText(next);
            try { onChange(JSON.parse(next)); setComplexError(null); } catch { setComplexError("Enter valid JSON."); }
          }} className={`${fieldClass} h-auto py-2 font-mono leading-4`} />
          {complexError ? <p role="alert" className="text-[9.5px] text-danger">{complexError}</p> : null}
        </>
      ) : (
        <Input id={`spec-${field.name}`} type={["integer", "number", "float"].includes(field.value_type) ? "number" : "text"} value={value == null ? "" : String(value)} placeholder={field.placeholder} onChange={(event) => onChange(["integer", "number", "float"].includes(field.value_type) ? Number(event.target.value) : event.target.value)} />
      )}
      {field.description ? <p className="text-[9.5px] leading-4 text-fg-disabled">{field.description}</p> : null}
    </div>
  );
}

function isVisible(field: SpecFieldDescriptor, value: Record<string, unknown>) {
  if (!field.visible_when || !Object.keys(field.visible_when).length) return true;
  return Object.entries(field.visible_when).every(([name, expected]) => value[name] === expected);
}

function humanize(value: string) {
  return value.replace(/[_-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

const fieldClass = "h-8 w-full rounded-md border border-border bg-surface px-2.5 text-[11.5px] text-fg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20";
