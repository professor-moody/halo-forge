import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowUpRight,
  BookOpen,
  Cpu,
  FileText,
  KeyRound,
  LifeBuoy,
  Play,
  Search,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { Topbar } from "@/components/shell";
import { Card, CardContent, CardHeader, CardTitle, CardEyebrow } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/docs")({
  component: DocsRoute,
});

const INTENT_LINKS = [
  {
    icon: Play,
    title: "First guided run",
    body: "Use safe catalog defaults, backend detection, and preflight before launch.",
    to: "/start",
    action: "Open Start",
    internal: true,
  },
  {
    icon: KeyRound,
    title: "Remote workstation",
    body: "Bind one Halo Forge machine to the network, create a token, and monitor it here.",
    to: "/connect",
    action: "Connect",
    internal: true,
  },
  {
    icon: Cpu,
    title: "Backend and hardware",
    body: "ROCm, CUDA, Apple MPS, MLX, CPU caveats, and feature coverage.",
    to: "https://halo-forge.io/docs/getting-started/hardware/",
    action: "Read hardware notes",
  },
  {
    icon: Search,
    title: "Choose a model",
    body: "Catalog guidance by task, backend, memory tier, and first-run risk.",
    to: "https://halo-forge.io/docs/getting-started/choose-a-model/",
    action: "Choose model",
  },
  {
    icon: ShieldCheck,
    title: "Verifiers",
    body: "Execution, compile, schema, metrics, LLM judge, and custom reward plugins.",
    to: "https://halo-forge.io/docs/verifiers/",
    action: "Open verifiers",
  },
  {
    icon: LifeBuoy,
    title: "Troubleshooting",
    body: "Backend setup, launch failures, token auth, and common training issues.",
    to: "https://halo-forge.io/docs/reference/troubleshooting/",
    action: "Troubleshoot",
  },
  {
    icon: Terminal,
    title: "CLI reference",
    body: "Every command, flag, token operation, and test profile.",
    to: "https://halo-forge.io/docs/reference/command-index/",
    action: "Command index",
  },
  {
    icon: FileText,
    title: "Local repo docs",
    body: "Markdown references checked into this repository for offline use.",
    to: "https://github.com/professor-moody/halo-forge/tree/main/docs",
    action: "Browse repo docs",
  },
];

function DocsRoute() {
  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Docs"
        subtitle="Intent-based references for local and remote Halo Forge workstations."
        actions={
          <Button asChild variant="ghost" size="sm">
            <a href="https://halo-forge.io/docs" target="_blank" rel="noreferrer">
              <BookOpen />
              Full docs
            </a>
          </Button>
        }
      />
      <div className="px-5 py-5 space-y-4 max-w-6xl">
        <section className="grid gap-3 lg:grid-cols-2">
          <IntroPanel />
          <RemotePanel />
        </section>

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {INTENT_LINKS.map((link) => (
            <DocIntentCard key={link.title} {...link} />
          ))}
        </section>
      </div>
    </>
  );
}

function IntroPanel() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>START HERE</CardEyebrow>
          <CardTitle>One machine, one workflow</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 text-[13px] text-fg-muted">
        <p>
          Halo Forge turns a base model into a trained, evaluated, served artifact
          on the workstation you control.
        </p>
        <div className="grid gap-2 sm:grid-cols-3">
          <Pill label="Train" value="SFT / RAFT" />
          <Pill label="Check" value="Eval / verifiers" />
          <Pill label="Ship" value="Serve / export" />
        </div>
      </CardContent>
    </Card>
  );
}

function RemotePanel() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>REMOTE V1</CardEyebrow>
          <CardTitle>Network access uses tokens</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 text-[13px] text-fg-muted">
        <p>
          Loopback stays zero-config. When the API is bound to a non-loopback host,
          the same app can control that workstation with a bearer token.
        </p>
        <code className="block rounded-md border border-border-subtle bg-bg-subtle px-3 py-2 font-mono text-[11px] text-fg">
          halo-forge token create dashboard
        </code>
      </CardContent>
    </Card>
  );
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg-subtle/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">{label}</div>
      <div className="mt-1 font-mono text-[12px] text-fg">{value}</div>
    </div>
  );
}

function DocIntentCard({
  icon: Icon,
  title,
  body,
  to,
  action,
  internal,
}: {
  icon: typeof BookOpen;
  title: string;
  body: string;
  to: string;
  action: string;
  internal?: boolean;
}) {
  const content = (
    <Card className="h-full transition-colors hover:border-border-strong hover:bg-surface-hover/25">
      <CardContent className="flex h-full flex-col gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-md border border-border-subtle bg-bg-subtle">
          <Icon className="h-4 w-4 text-accent" />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-fg">{title}</h2>
          <p className="mt-1 text-[12px] leading-5 text-fg-muted">{body}</p>
        </div>
        <div className="mt-auto flex items-center gap-1.5 text-[12px] font-medium text-accent">
          {action}
          <ArrowUpRight className="h-3.5 w-3.5" />
        </div>
      </CardContent>
    </Card>
  );

  if (internal) {
    return (
      <Link to={to} className="block">
        {content}
      </Link>
    );
  }

  return (
    <a href={to} target="_blank" rel="noreferrer" className="block">
      {content}
    </a>
  );
}
