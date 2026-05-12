import { createFileRoute, Link } from "@tanstack/react-router";
import {
  AudioLines,
  Boxes,
  ChevronRight,
  Code2,
  Cpu,
  FlaskConical,
  Loader2,
  Search,
  Sparkles,
  ScanEye,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardEyebrow,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useModelCatalog } from "@/lib/hooks";
import type { ModelCatalogEntry } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/models")({
  component: ModelsRoute,
});

const STATUS_TONE: Record<string, "success" | "warning" | "neutral" | "info"> = {
  recommended: "success",
  compatible: "info",
  experimental: "warning",
  deprecated: "neutral",
};

const RISK_TONE: Record<string, "success" | "warning" | "neutral" | "info"> = {
  safe: "success",
  caveated: "warning",
  experimental: "warning",
};

type IntentPreset = {
  label: string;
  description: string;
  icon: typeof Sparkles;
  query?: string;
  provider?: string;
  status?: string;
  modality?: string;
};

const INTENT_PRESETS: IntentPreset[] = [
  {
    label: "First run",
    description: "Small, proven catalog picks for smoke tests.",
    icon: Sparkles,
    query: "quickstart",
    status: "recommended",
  },
  {
    label: "Code RAFT",
    description: "Coder models that fit verifier-ranked training.",
    icon: Code2,
    query: "code",
    status: "recommended",
  },
  {
    label: "Apple Silicon",
    description: "MLX-native or Apple-friendly local models.",
    icon: Cpu,
    query: "mlx",
  },
  {
    label: "VLM",
    description: "Vision-language candidates and caveats.",
    icon: ScanEye,
    modality: "vision",
  },
  {
    label: "Audio",
    description: "ASR/audio training starting points.",
    icon: AudioLines,
    modality: "audio",
  },
  {
    label: "Liquid AI",
    description: "Interesting LFM entries, marked experimental.",
    icon: FlaskConical,
    provider: "Liquid AI",
  },
];

function ModelsRoute() {
  const { data, isLoading, isError } = useModelCatalog();
  const [query, setQuery] = useState("");
  const [provider, setProvider] = useState("all");
  const [status, setStatus] = useState("all");
  const [modality, setModality] = useState("all");

  const items = data?.items ?? [];
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((item) => {
      if (provider !== "all" && item.provider !== provider) return false;
      if (status !== "all" && item.status !== status) return false;
      if (modality !== "all" && !item.modalities.includes(modality)) return false;
      if (!q) return true;
      return [
        item.id,
        item.label,
        item.provider,
        item.family,
        item.recommended_use,
        ...item.tasks,
        ...item.trainer_support,
      ].some((value) => value.toLowerCase().includes(q));
    });
  }, [items, modality, provider, query, status]);

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Models"
        subtitle="Curated upstream and MLX-format base models for training, eval, and serving."
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/train">
              Train <ChevronRight className="h-3.5 w-3.5" />
            </Link>
          </Button>
        }
      />
      <div className="px-5 py-5 space-y-4">
        <IntentBar
          presets={INTENT_PRESETS}
          onSelect={(preset) => {
            setQuery(preset.query ?? "");
            setProvider(preset.provider ?? "all");
            setStatus(preset.status ?? "all");
            setModality(preset.modality ?? "all");
          }}
        />

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>CATALOG</CardEyebrow>
              <CardTitle>Model Browser</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-fg-subtle">
                {filtered.length}/{items.length} models
                {data?.catalog_version ? ` · v${data.catalog_version}` : ""}
              </span>
              <button
                type="button"
                onClick={() => {
                  setQuery("");
                  setProvider("all");
                  setStatus("all");
                  setModality("all");
                }}
                className="text-[11px] text-fg-subtle hover:text-fg"
              >
                Clear
              </button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2 md:grid-cols-[1fr_auto_auto_auto]">
              <label className="relative">
                <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-disabled" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search by model, family, task, trainer..."
                  className="pl-7"
                />
              </label>
              <FilterSelect
                label="Provider"
                value={provider}
                values={data?.facets.providers ?? []}
                onChange={setProvider}
              />
              <FilterSelect
                label="Status"
                value={status}
                values={data?.facets.statuses ?? []}
                onChange={setStatus}
              />
              <FilterSelect
                label="Modality"
                value={modality}
                values={data?.facets.modalities ?? []}
                onChange={setModality}
              />
            </div>
          </CardContent>
        </Card>

        {isLoading ? (
          <div className="flex h-32 items-center justify-center gap-2 text-sm text-fg-muted">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading model catalog...
          </div>
        ) : isError || !data ? (
          <div className="text-sm text-danger">Failed to load model catalog.</div>
        ) : (
          <div className="grid gap-3">
            {filtered.map((model) => (
              <ModelRow key={model.id} model={model} />
            ))}
            {filtered.length === 0 ? (
              <div className="rounded-md border border-border-subtle px-5 py-10 text-center text-sm text-fg-muted">
                No models match the current filters.
              </div>
            ) : null}
          </div>
        )}
      </div>
    </>
  );
}

function IntentBar({
  presets,
  onSelect,
}: {
  presets: IntentPreset[];
  onSelect: (preset: IntentPreset) => void;
}) {
  return (
    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
      {presets.map((preset) => (
        <button
          key={preset.label}
          type="button"
          onClick={() => onSelect(preset)}
          className={cn(
            "group rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors",
            "hover:border-border-strong hover:bg-surface-hover/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
          )}
        >
          <div className="flex items-center gap-2">
            <preset.icon className="h-3.5 w-3.5 text-accent" />
            <span className="text-[12.5px] font-medium text-fg">{preset.label}</span>
          </div>
          <p className="mt-1 text-[11.5px] leading-snug text-fg-subtle">
            {preset.description}
          </p>
        </button>
      ))}
    </div>
  );
}

function FilterSelect({
  label,
  value,
  values,
  onChange,
}: {
  label: string;
  value: string;
  values: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px] text-fg-muted">
      <span className="uppercase tracking-wider text-fg-disabled">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-8 rounded-sm border border-border-subtle bg-surface px-2 text-[12px] text-fg"
      >
        <option value="all">All</option>
        {values.map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
      </select>
    </label>
  );
}

function ModelRow({ model }: { model: ModelCatalogEntry }) {
  const caveated = model.known_caveats.length > 0 || model.trust_remote_code_required;

  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Boxes className="h-4 w-4 text-accent" />
            <h2 className="font-mono text-[13px] font-semibold text-fg">{model.id}</h2>
            <Badge tone={STATUS_TONE[model.status] ?? "neutral"} size="sm">
              {model.status}
            </Badge>
            {model.recommended_first_run ? (
              <Badge tone="success" size="sm">
                best first pick
              </Badge>
            ) : null}
            <Badge tone={RISK_TONE[model.risk_level] ?? "neutral"} size="sm">
              {model.risk_level}
            </Badge>
            <Badge tone="neutral" size="sm">
              {model.memory_tier}
            </Badge>
            {model.estimated_memory_gb ? (
              <Badge tone="neutral" size="sm">
                ~{model.estimated_memory_gb}GB
              </Badge>
            ) : null}
            {caveated ? (
              <Badge tone="warning" size="sm">
                caveats
              </Badge>
            ) : null}
          </div>
          <p className="text-[13px] text-fg-muted">{model.recommended_use}</p>
          <div className="flex flex-wrap gap-1.5">
            {[model.provider, model.family, model.parameter_count, ...model.modalities].map((chip) => (
              <span
                key={chip}
                className="rounded-sm border border-border-subtle px-1.5 py-0.5 text-[10px] text-fg-subtle"
              >
                {chip}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {model.trainer_support.map((mode) => (
              <span key={mode} className="font-mono text-[10px] text-fg-disabled">
                {mode}
              </span>
            ))}
          </div>
          {model.fit_notes.length ? (
            <div className="space-y-1 text-[11px] text-fg-subtle">
              {model.fit_notes.map((note) => (
                <div key={note}>{note}</div>
              ))}
            </div>
          ) : null}
          {model.license_note || model.download_note || model.trust_remote_code_required ? (
            <div className="space-y-1 text-[11px] text-warning">
              {model.license_note ? <div>{model.license_note}</div> : null}
              {model.download_note ? <div>{model.download_note}</div> : null}
              {model.trust_remote_code_required ? (
                <div>Remote model code is required; enable it only for trusted repositories.</div>
              ) : null}
            </div>
          ) : null}
          {model.known_caveats.length ? (
            <ul className="space-y-1 text-[11px] text-warning">
              {model.known_caveats.map((caveat) => (
                <li key={caveat}>{caveat}</li>
              ))}
            </ul>
          ) : null}
        </div>
        <div className="flex shrink-0 flex-wrap gap-2 md:justify-end">
          <Button asChild size="sm" variant="primary">
            <Link to="/train" search={{ model: model.id, mode: preferredTrainMode(model) }}>
              Use in Training
            </Link>
          </Button>
          {model.mlx_variant ? (
            <button
              type="button"
              className={cn(
                "rounded-sm border border-border-subtle px-2 py-1 text-[11px] font-mono text-fg-subtle",
                "cursor-default",
              )}
              title={model.mlx_variant}
            >
              MLX variant
            </button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function preferredTrainMode(model: ModelCatalogEntry): "sft" | "raft" {
  return model.trainer_support.includes("raft") ? "raft" : "sft";
}
