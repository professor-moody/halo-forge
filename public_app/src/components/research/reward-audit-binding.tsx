import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  CheckCircle2,
  CircleDashed,
  Gauge,
  Link2,
  Loader2,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo } from "react";
import {
  api,
  type BenchmarkSuite,
  type RewardAuditProtocolRevision,
  type RewardIntegrityProfileRevision,
  type RewardSystem,
  type TrainingMode,
  type TrainingSignalCapabilityDescriptor,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SearchPicker } from "@/components/ui/search-picker";
import { cn } from "@/lib/utils";

export type RewardAuditBindingValue = {
  enabled: boolean;
  rewardSystemRevisionId: string;
  auditProtocolRevisionId: string;
  integrityProfileRevisionId: string;
  auditBoundaries: string;
  developmentSuiteRevisionId: string;
};

export const EMPTY_REWARD_AUDIT_BINDING: RewardAuditBindingValue = {
  enabled: false,
  rewardSystemRevisionId: "",
  auditProtocolRevisionId: "",
  integrityProfileRevisionId: "",
  auditBoundaries: "",
  developmentSuiteRevisionId: "",
};

const AUDITED_MODES = new Set<TrainingMode>([
  "raft",
  "grpo",
  "reasoning",
  "agentic",
  "vlm",
  "audio",
]);

export function RewardAuditBindingEditor({
  trainerMode,
  backendFamily,
  value,
  onChange,
  totalBudget,
  budgetUnit,
  compact = false,
}: {
  trainerMode: TrainingMode;
  backendFamily: string;
  value: RewardAuditBindingValue;
  onChange: (value: RewardAuditBindingValue) => void;
  totalBudget?: number;
  budgetUnit?: "step" | "cycle" | "epoch";
  compact?: boolean;
}) {
  const applicable = AUDITED_MODES.has(trainerMode);
  const capabilities = useQuery({
    queryKey: ["reward-integrity-capabilities"],
    queryFn: api.rewardIntegrityCapabilities,
    enabled: applicable,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const systems = useQuery({
    queryKey: ["reward-systems", "guided", trainerMode, backendFamily],
    queryFn: () => api.listRewardSystems({ trainerMode, backendFamily, qualifiedOnly: true, limit: 200 }),
    enabled: applicable && value.enabled,
    retry: false,
  });
  const protocols = useQuery({
    queryKey: ["reward-audit-protocols", "guided"],
    queryFn: () => api.listRewardAuditProtocols({ limit: 100 }),
    enabled: applicable && value.enabled,
    retry: false,
  });
  const profiles = useQuery({
    queryKey: ["reward-integrity-profiles", "guided"],
    queryFn: () => api.listRewardIntegrityProfiles({ limit: 100 }),
    enabled: applicable && value.enabled,
    retry: false,
  });
  const developmentSuites = useQuery({
    queryKey: ["benchmark-suites", "reward-audit", "development"],
    queryFn: api.listBenchmarkSuites,
    enabled: applicable && value.enabled,
    retry: false,
  });

  const capability = useMemo(
    () => selectCapability(capabilities.data?.items ?? [], trainerMode, backendFamily),
    [backendFamily, capabilities.data?.items, trainerMode],
  );
  const selectedSystem = useMemo(
    () => (systems.data?.items ?? []).find((item) => rewardSystemRevisionId(item) === value.rewardSystemRevisionId),
    [systems.data?.items, value.rewardSystemRevisionId],
  );
  const selectedProtocol = useMemo(
    () => (protocols.data?.items ?? []).find((item) => item.id === value.auditProtocolRevisionId),
    [protocols.data?.items, value.auditProtocolRevisionId],
  );
  const selectedProfile = useMemo(
    () => (profiles.data?.items ?? []).find((item) => item.id === value.integrityProfileRevisionId),
    [profiles.data?.items, value.integrityProfileRevisionId],
  );
  const eligibleDevelopmentSuites = useMemo(
    () => (developmentSuites.data?.items ?? []).filter(isEligibleDevelopmentSuite),
    [developmentSuites.data?.items],
  );
  const selectedDevelopmentSuite = useMemo(
    () => eligibleDevelopmentSuites.find((item) => benchmarkSuiteRevisionId(item) === value.developmentSuiteRevisionId),
    [eligibleDevelopmentSuites, value.developmentSuiteRevisionId],
  );

  useEffect(() => {
    if (!value.enabled) return;
    const next = { ...value };
    if (!next.auditProtocolRevisionId) {
      next.auditProtocolRevisionId = preferredProtocol(protocols.data?.items ?? [])?.id ?? "";
    }
    if (!next.integrityProfileRevisionId) {
      next.integrityProfileRevisionId = preferredProfile(profiles.data?.items ?? [])?.id ?? "";
    }
    if (capability?.resumable === false) {
      next.auditBoundaries = "final";
    } else if (!next.auditBoundaries && totalBudget && totalBudget > 0) {
      next.auditBoundaries = defaultBoundaries(totalBudget).join(", ");
    }
    if (
      next.auditProtocolRevisionId !== value.auditProtocolRevisionId ||
      next.integrityProfileRevisionId !== value.integrityProfileRevisionId ||
      next.auditBoundaries !== value.auditBoundaries
    ) {
      onChange(next);
    }
  }, [capability?.resumable, onChange, profiles.data?.items, protocols.data?.items, totalBudget, value]);

  if (!applicable) return null;

  const serviceUnavailable = capabilities.isError;
  const gatingCapable = capability && ["exact", "sampled"].includes(capability.capture_fidelity);
  const ready = Boolean(
    value.enabled &&
      gatingCapable &&
      value.rewardSystemRevisionId &&
      value.auditProtocolRevisionId &&
      value.integrityProfileRevisionId,
  );

  return (
    <section className={cn("overflow-hidden border border-border bg-bg", compact ? "rounded-md" : "rounded-lg")} aria-labelledby="reward-audit-binding-title">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border-subtle bg-bg-subtle/45 px-4 py-3">
        <div className="flex min-w-0 items-start gap-3">
          <span className={cn("mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-md", value.enabled ? "bg-accent-bg text-accent" : "bg-surface text-fg-disabled")}>
            <ShieldCheck className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 id="reward-audit-binding-title" className="text-[12px] font-medium text-fg">Training signal audit</h3>
              <Badge tone={ready ? "success" : value.enabled ? "warning" : "neutral"} size="sm" dot>
                {ready ? "ready" : value.enabled ? "needs setup" : "off"}
              </Badge>
            </div>
            <p className="mt-0.5 text-[10px] leading-4 text-fg-subtle">
              Rescore the exact training outputs with an independent qualified sentinel at selected boundaries.
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="sm"
          variant={value.enabled ? "secondary" : "ghost"}
          onClick={() => onChange(value.enabled ? EMPTY_REWARD_AUDIT_BINDING : { ...value, enabled: true })}
          disabled={serviceUnavailable}
          aria-pressed={value.enabled}
        >
          {value.enabled ? "Disable audit" : "Add audit"}
        </Button>
      </div>

      {serviceUnavailable ? (
        <InlineNotice tone="warning" title="Reward integrity service is unavailable">
          The existing verifier launch remains unchanged. Start the v8 service to configure same-output audits.
        </InlineNotice>
      ) : capabilities.isLoading ? (
        <div className="flex items-center gap-2 px-4 py-4 text-[10px] text-fg-muted"><Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />Checking trainer audit capability</div>
      ) : !value.enabled ? (
        <div className="grid gap-px bg-border-subtle sm:grid-cols-3">
          <CompactStep label="Capture" value={capability ? fidelityLabel(capability.capture_fidelity) : "Not declared"} />
          <CompactStep label="Boundary" value={capability?.boundary_unit?.replaceAll("_", " ") || "Not declared"} />
          <CompactStep label="Behavior" value="Fail pauses for review" />
        </div>
      ) : (
        <div>
          {!capability ? (
            <InlineNotice tone="warning" title="No matching trainer capability">
              {trainerMode.toUpperCase()} on {backendFamily} has not declared a trustworthy capture contract. Audits remain report-only until it does.
            </InlineNotice>
          ) : (
            <div className="grid gap-px border-b border-border-subtle bg-border-subtle sm:grid-cols-4">
              <CompactStep label="Capture" value={fidelityLabel(capability.capture_fidelity)} tone={gatingCapable ? "success" : "warning"} />
              <CompactStep label="Boundary unit" value={capability.boundary_unit.replaceAll("_", " ")} />
              <CompactStep label="Resume" value={capability.resumable ? "Verified segments" : "Final audit only"} />
              <CompactStep label="Candidates" value={(capability.candidate_multiplicity || "declared by trainer").replaceAll("_", " ")} />
            </div>
          )}

          <div className="grid gap-5 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_260px]">
            <div className="space-y-4">
              <AuditField label="1–3 · Training verifier, sentinel, and reward mapping">
                <SearchPicker
                  value={value.rewardSystemRevisionId}
                  onChange={(rewardSystemRevisionId) => onChange({ ...value, rewardSystemRevisionId })}
                  options={(systems.data?.items ?? []).map((item) => ({
                    value: rewardSystemRevisionId(item),
                    label: item.name,
                    description: rewardSystemDescription(item),
                    status: rewardSystemStatus(item),
                    keywords: `${item.id} ${rewardSystemRevisionId(item)}`,
                  }))}
                  placeholder="Choose a qualified reward system"
                  emptyLabel={systems.isLoading ? "Loading reward systems" : "No compatible disjoint sentinel is qualified"}
                />
                <p className="mt-1 text-[8.5px] leading-4 text-fg-disabled">One immutable revision pins optimizer, sentinel, normalization, and shaping. Sentinel scores never enter gradients.</p>
              </AuditField>

              <div className="grid gap-3 md:grid-cols-2">
                <AuditField label="4 · Same-sample capture protocol">
                  <SearchPicker
                    value={value.auditProtocolRevisionId}
                    onChange={(auditProtocolRevisionId) => onChange({ ...value, auditProtocolRevisionId })}
                    options={(protocols.data?.items ?? []).map(protocolOption)}
                    placeholder="Choose capture protocol"
                    emptyLabel={protocols.isLoading ? "Loading protocols" : "No protocol revisions available"}
                  />
                </AuditField>
                <AuditField label="5 · Integrity rules">
                  <SearchPicker
                    value={value.integrityProfileRevisionId}
                    onChange={(integrityProfileRevisionId) => onChange({ ...value, integrityProfileRevisionId })}
                    options={(profiles.data?.items ?? []).map(profileOption)}
                    placeholder="Choose integrity policy"
                    emptyLabel={profiles.isLoading ? "Loading policies" : "No integrity profile revisions available"}
                  />
                </AuditField>
              </div>

              <AuditField label={`Selected boundaries${budgetUnit ? ` · ${budgetUnit}` : ""}`}>
                <Input
                  value={value.auditBoundaries}
                  onChange={(event) => onChange({ ...value, auditBoundaries: event.target.value })}
                  placeholder={totalBudget ? defaultBoundaries(totalBudget).join(", ") : "25%, 50%, 75%, 100%"}
                  className="font-mono"
                  aria-describedby="reward-audit-boundary-help"
                />
                <p id="reward-audit-boundary-help" className="mt-1 text-[8.5px] leading-4 text-fg-disabled">
                  Up to four resolved boundaries; the first and final are always included. Non-resumable backends use final only.
                </p>
              </AuditField>

              <AuditField label="Optional checkpoint quality suite">
                <SearchPicker
                  value={value.developmentSuiteRevisionId}
                  onChange={(developmentSuiteRevisionId) => onChange({ ...value, developmentSuiteRevisionId })}
                  options={eligibleDevelopmentSuites.map((item) => ({
                    value: benchmarkSuiteRevisionId(item),
                    label: item.name,
                    description: `${humanizePurpose(item.purpose)} · immutable revision ${benchmarkSuiteRevisionId(item)}`,
                    keywords: `${item.id} ${item.description ?? ""}`,
                  }))}
                  placeholder="Choose a development suite (optional)"
                  emptyLabel={developmentSuites.isLoading ? "Loading eligible suites" : "No development suites are published"}
                />
                <p className="mt-1 text-[8.5px] leading-4 text-fg-disabled">
                  Development or unspecified suites only. The checkpoint evaluation is recorded independently before the reward audit; protected evidence is never offered here.
                </p>
              </AuditField>
            </div>

            <aside className="border-t border-border-subtle pt-4 lg:border-l lg:border-t-0 lg:pl-4 lg:pt-0" aria-label="Reward audit request preview">
              <div className="text-[9px] font-medium uppercase tracking-[0.12em] text-fg-muted">6 · Review</div>
              <dl className="mt-2 divide-y divide-border-subtle border-y border-border-subtle">
                <PreviewRow label="Reward system" value={selectedSystem?.name || "Not selected"} />
                <PreviewRow label="Protocol" value={selectedProtocol?.name || "Not selected"} />
                <PreviewRow label="Integrity" value={selectedProfile?.name || "Not selected"} />
                <PreviewRow label="Quality suite" value={selectedDevelopmentSuite?.name || "Optional · not selected"} />
                <PreviewRow label="Schedule" value={capability?.resumable === false ? "Final only" : value.auditBoundaries || "Resolve at launch"} mono />
                <PreviewRow label="Failure" value="Pause for review" />
                <PreviewRow label="Lease" value="Released before sentinel" />
              </dl>
              <div className={cn("mt-3 flex items-start gap-2 text-[9px] leading-4", ready ? "text-success" : "text-fg-disabled")}>
                {ready ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" /> : <CircleDashed className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
                {ready ? "Same-output audit binding is complete." : "Complete the three immutable selections to enable audited launch."}
              </div>
              <a href="/eval?section=verifiers&verifierView=training-audits&auditView=profiles" className="mt-3 inline-flex items-center gap-1 text-[9.5px] text-accent hover:underline">
                <Link2 className="h-3 w-3" />Manage reward systems and profiles
              </a>
            </aside>
          </div>
        </div>
      )}
    </section>
  );
}

function selectCapability(items: TrainingSignalCapabilityDescriptor[], trainerMode: string, backendFamily: string) {
  return items.find((item) => item.trainer_mode === trainerMode && item.backend_family === backendFamily)
    ?? items.find((item) => item.trainer_mode === trainerMode && item.backend_family === "*")
    ?? items.find((item) => item.trainer_mode === trainerMode);
}

function rewardSystemRevisionId(system: RewardSystem): string {
  return system.latest_revision?.id || system.latest_revision_id || "";
}

function rewardSystemDescription(system: RewardSystem): string {
  const revision = system.latest_revision;
  if (!revision) return system.description || (system.latest_revision_id ? "Published immutable revision" : "No published revision");
  const primary = revision.auditors?.find((item) => item.role === "primary_sentinel");
  return `${revision.modality} · ${revision.task_type} · ${primary?.correlated ? "correlated sentinel · inspect only" : "disjoint primary sentinel"}`;
}

function rewardSystemStatus(system: RewardSystem): string {
  return system.latest_revision?.qualification_state || (system.latest_revision_id ? "published" : "draft");
}

function preferredProtocol(items: RewardAuditProtocolRevision[]) {
  return items.find((item) => item.template === "balanced_256") ?? items[0];
}

function preferredProfile(items: RewardIntegrityProfileRevision[]) {
  return items.find((item) => item.template === "human_aligned_integrity") ?? items.find((item) => item.promotable !== false) ?? items[0];
}

function protocolOption(item: RewardAuditProtocolRevision) {
  return {
    value: item.id,
    label: item.name,
    description: `${item.template.replaceAll("_", " ")} · ${item.uniform_core_limit ?? "all"} core + ${item.diagnostic_limit ?? 0} diagnostic`,
    status: item.capture_required_for_gating === false ? "report only" : "same-output",
  };
}

function profileOption(item: RewardIntegrityProfileRevision) {
  return {
    value: item.id,
    label: item.name,
    description: `${item.template.replaceAll("_", " ")} · ${item.minimum_pass_records ?? 100} records for pass`,
    status: item.promotable === false ? "report only" : "gating",
  };
}

function benchmarkSuiteRevisionId(suite: BenchmarkSuite): string {
  return suite.latest_revision?.id || suite.latest_revision_id || suite.revisions?.at(-1)?.id || "";
}

function isEligibleDevelopmentSuite(suite: BenchmarkSuite): boolean {
  return Boolean(benchmarkSuiteRevisionId(suite)) && ["development", "unspecified"].includes(suite.purpose || "unspecified");
}

function humanizePurpose(value?: string): string {
  return (value || "unspecified").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function defaultBoundaries(total: number): number[] {
  if (total <= 1) return [Math.max(1, total)];
  return [...new Set([1, Math.round(1 + (total - 1) / 3), Math.round(1 + (2 * (total - 1)) / 3), total])].sort((a, b) => a - b);
}

function fidelityLabel(value: string): string {
  if (value === "aggregate_only") return "Aggregate only";
  if (value === "unavailable") return "Unavailable";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function CompactStep({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "success" | "warning" }) {
  const Icon = label === "Capture" ? Activity : label === "Boundary" || label === "Boundary unit" ? Gauge : ShieldCheck;
  return <div className="flex items-center gap-2 bg-bg px-3 py-2.5"><Icon className={cn("h-3.5 w-3.5 shrink-0", tone === "success" ? "text-success" : tone === "warning" ? "text-warning" : "text-fg-disabled")} /><div className="min-w-0"><div className="text-[8.5px] uppercase tracking-[0.11em] text-fg-disabled">{label}</div><div className="truncate text-[10px] text-fg-subtle">{value}</div></div></div>;
}

function AuditField({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="block"><span className="mb-1.5 block text-[9px] font-medium uppercase tracking-[0.11em] text-fg-muted">{label}</span>{children}</label>;
}

function PreviewRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div className="grid grid-cols-[84px_minmax(0,1fr)] gap-2 py-2 text-[9px]"><dt className="text-fg-disabled">{label}</dt><dd className={cn("break-words text-right text-fg-subtle", mono && "font-mono")}>{value}</dd></div>;
}

function InlineNotice({ tone, title, children }: { tone: "warning" | "neutral"; title: string; children: React.ReactNode }) {
  return <div className={cn("flex items-start gap-2 border-b border-border-subtle px-4 py-3", tone === "warning" ? "bg-warning/5" : "bg-accent/5")}><TriangleAlert className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", tone === "warning" ? "text-warning" : "text-accent")} /><div><div className="text-[10px] font-medium text-fg">{title}</div><div className="mt-0.5 text-[9.5px] leading-4 text-fg-subtle">{children}</div></div></div>;
}
