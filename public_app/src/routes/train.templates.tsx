import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  AudioLines,
  BookOpen,
  Brain,
  CheckCheck,
  ChevronRight,
  Clock,
  Code,
  Eye,
  Loader2,
  ScrollText,
  Sparkles,
  Terminal,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { useState } from "react";
import { api, type TrainingTemplate } from "@/lib/api";
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
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/train/templates")({
  component: TemplatesGalleryRoute,
});

/**
 * Templates gallery — the intent-first entrypoint to training.
 *
 * The /train page is the knob-by-knob configurator: useful once you
 * know what you want, painful as a starting point. This page flips
 * the question. Each card is a goal a user walks in with — "train
 * Python coding", "fine-tune Whisper for podcasts" — and binds the
 * underlying modality + model + dataset + hyperparams in one click.
 *
 * Two affordances per card:
 *   - "Use template" -> deep-link into /train with the template applied.
 *     The configurator picks up the values via the ?template= search param.
 *   - "Show CLI" -> reveal the matching halo-forge invocation for CLI parity.
 */

const CATEGORY_ICONS: Record<string, LucideIcon> = {
  code: Code,
  reasoning: Brain,
  vision: Eye,
  audio: AudioLines,
  preference: CheckCheck,
  agentic: Wrench,
};

const FORM_SUPPORTED: ReadonlySet<string> = new Set([
  "sft",
  "raft",
  "dpo",
  "orpo",
  "rm",
  "grpo",
  "vlm",
  "audio",
  "reasoning",
  "agentic",
]);

function TemplatesGalleryRoute() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["training-templates"],
    queryFn: () => api.trainingTemplates(),
    staleTime: 60_000,
  });

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Training templates"
        subtitle="Pick a goal — model, dataset, and hyperparams come pre-filled."
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/train">
              Configure manually <ChevronRight className="h-3.5 w-3.5" />
            </Link>
          </Button>
        }
      />

      <div className="px-5 py-5 max-w-6xl space-y-8">
        {isLoading ? (
          <div className="flex h-32 items-center justify-center text-fg-muted text-sm gap-2">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading templates…
          </div>
        ) : isError || !data ? (
          <div className="text-danger text-sm">Failed to load templates.</div>
        ) : (
          data.categories.map((cat) => {
            const items = data.items.filter((t) => t.category === cat.id);
            if (items.length === 0) return null;
            const Icon = CATEGORY_ICONS[cat.id] ?? Sparkles;
            return (
              <section key={cat.id} aria-labelledby={`cat-${cat.id}-h`}>
                <header className="mb-3">
                  <h2
                    id={`cat-${cat.id}-h`}
                    className="flex items-center gap-2 text-[14px] font-semibold tracking-tight text-fg"
                  >
                    <Icon className="h-4 w-4 text-accent" />
                    {cat.label}
                  </h2>
                  <p className="text-[12px] text-fg-muted mt-0.5 ml-6">
                    {cat.description}
                  </p>
                </header>
                <div className="grid gap-3 md:grid-cols-2">
                  {items.map((tpl) => (
                    <TemplateCard key={tpl.id} tpl={tpl} />
                  ))}
                </div>
              </section>
            );
          })
        )}
      </div>
    </>
  );
}

function TemplateCard({ tpl }: { tpl: TrainingTemplate }) {
  const [showCli, setShowCli] = useState(false);
  const formSupported = FORM_SUPPORTED.has(tpl.modality);

  return (
    <Card className="hover:border-border-strong transition-colors">
      <CardHeader className="pb-2">
        <CardEyebrow className="flex items-center gap-2">
          <Badge tone="info" size="sm">{tpl.modality}</Badge>
          {tpl.expected_runtime ? (
            <span className="flex items-center gap-1 text-fg-disabled">
              <Clock className="h-3 w-3" />
              {tpl.expected_runtime}
            </span>
          ) : null}
        </CardEyebrow>
        <CardTitle>{tpl.name}</CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        <p className="text-[13px] text-fg-muted leading-snug">{tpl.intent}</p>

        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[11px] font-mono">
          <dt className="text-fg-disabled">model</dt>
          <dd className="text-fg truncate">{tpl.model_hint}</dd>
          <dt className="text-fg-disabled">dataset</dt>
          <dd className="text-fg truncate">
            {tpl.dataset_hint === "@custom" ? (
              <span className="italic text-fg-muted">your own data</span>
            ) : (
              tpl.dataset_hint
            )}
          </dd>
          {tpl.verifier ? (
            <>
              <dt className="text-fg-disabled">verifier</dt>
              <dd className="text-fg">{tpl.verifier}</dd>
            </>
          ) : null}
        </dl>

        <div className="flex items-center gap-2 pt-1">
          {formSupported ? (
            <Button asChild size="sm" variant="primary">
              <Link to="/train" search={{ template: tpl.id }}>
                Use template <ChevronRight className="h-3 w-3" />
              </Link>
            </Button>
          ) : (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setShowCli((v) => !v)}
            >
              <Terminal className="h-3 w-3" />
              {showCli ? "Hide CLI" : "Show CLI"}
            </Button>
          )}
          {tpl.learn_more ? (
            <Button asChild size="sm" variant="ghost">
              <a
                href={tpl.learn_more}
                target="_blank"
                rel="noopener noreferrer"
              >
                <BookOpen className="h-3 w-3" />
                Docs
              </a>
            </Button>
          ) : null}
          {!formSupported ? null : (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowCli((v) => !v)}
            >
              <Terminal className="h-3 w-3" />
              {showCli ? "Hide CLI" : "Show CLI"}
            </Button>
          )}
        </div>

        {showCli ? <CliBlock templateId={tpl.id} /> : null}
      </CardContent>
    </Card>
  );
}

function CliBlock({ templateId }: { templateId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["training-template", templateId],
    queryFn: () => api.trainingTemplate(templateId),
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <div className="flex h-10 items-center text-fg-muted text-[11px] gap-1.5">
        <Loader2 className="h-3 w-3 animate-spin" /> Loading…
      </div>
    );
  }
  if (!data?.cli) {
    return (
      <div className="text-danger text-[11px]">CLI invocation unavailable.</div>
    );
  }
  return (
    <div className="rounded-md border border-border-subtle bg-bg-subtle px-3 py-2 space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-fg-disabled">
          <ScrollText className="h-3 w-3" /> CLI
        </span>
        <button
          type="button"
          onClick={() => {
            navigator.clipboard?.writeText(data.cli);
          }}
          className={cn(
            "text-[10px] text-accent hover:text-accent/80 transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-sm",
          )}
        >
          Copy
        </button>
      </div>
      <code className="block font-mono text-[11px] text-fg whitespace-pre-wrap break-all leading-relaxed">
        {data.cli}
      </code>
    </div>
  );
}
