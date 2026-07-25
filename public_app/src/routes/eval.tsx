import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  BarChart3,
  ClipboardCheck,
  Database,
  FlaskConical,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Square,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  api,
  type BenchmarkSuite,
  type BenchmarkSuiteCreatePayload,
  type BenchmarkSuiteItem,
  type BenchmarkSuiteRevision,
  type EvalCohortResponse,
  type Evaluation,
  type EvaluationComparison,
  type EvaluationDrift,
  type EvaluationHistoryItem,
  type EvaluationSampleDelta,
  type EvaluationSubject,
  type FailureMiningPreview,
  type FailureMiningSelector,
  type ModelArtifactOccurrence,
  type ModelCatalogEntry,
  type RunListItem,
} from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardEyebrow, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { SearchPicker } from "@/components/ui/search-picker";
import { useModelCatalog } from "@/lib/hooks";
import { usePinnedRuns } from "@/lib/pinned-runs";
import { cn } from "@/lib/utils";
import { VerifierReliabilityWorkspace, type VerifierStudioView } from "@/routes/verifiers";
import type { RewardAuditStudioView } from "@/components/research/reward-integrity-workspace";

export const Route = createFileRoute("/eval")({
  component: EvaluationLabRoute,
  validateSearch: (search): {
    runId?: string;
    suite?: string;
    evaluation?: string;
    section?: EvalSection;
    verifierView?: VerifierStudioView;
    profile?: string;
    calibration?: string;
    base?: string;
    candidates?: string;
    record?: string;
    classification?: string;
    page?: number;
    auditView?: RewardAuditStudioView;
    audit?: string;
    auditBase?: string;
    auditCandidate?: string;
    auditSample?: string;
    auditPage?: number;
    auditClassification?: string;
    environment?: string;
    episode?: string;
  } => ({
    runId: typeof search.runId === "string" ? search.runId : undefined,
    suite: typeof search.suite === "string" ? search.suite : undefined,
    evaluation: typeof search.evaluation === "string" ? search.evaluation : undefined,
    section: isEvalSection(search.section) ? search.section : undefined,
    verifierView: isVerifierView(search.verifierView) ? search.verifierView : undefined,
    profile: typeof search.profile === "string" ? search.profile : undefined,
    calibration: typeof search.calibration === "string" ? search.calibration : undefined,
    base: typeof search.base === "string" ? search.base : undefined,
    candidates: typeof search.candidates === "string" ? search.candidates : undefined,
    record: typeof search.record === "string" ? search.record : undefined,
    classification: typeof search.classification === "string" ? search.classification : undefined,
    page: typeof search.page === "number" && Number.isFinite(search.page) ? Math.max(1, Math.floor(search.page)) : undefined,
    auditView: isRewardAuditView(search.auditView) ? search.auditView : undefined,
    audit: typeof search.audit === "string" ? search.audit : undefined,
    auditBase: typeof search.auditBase === "string" ? search.auditBase : undefined,
    auditCandidate: typeof search.auditCandidate === "string" ? search.auditCandidate : undefined,
    auditSample: typeof search.auditSample === "string" ? search.auditSample : undefined,
    auditPage: typeof search.auditPage === "number" && Number.isFinite(search.auditPage) ? Math.max(1, Math.floor(search.auditPage)) : undefined,
    auditClassification: typeof search.auditClassification === "string" ? search.auditClassification : undefined,
    environment: typeof search.environment === "string" ? search.environment : undefined,
    episode: typeof search.episode === "string" ? search.episode : undefined,
  }),
});

type EvalView = "lab" | "legacy";
type EvalSection = "suites" | "launch" | "results" | "compare" | "failure-review" | "verifiers" | "environments";
type SubjectKind = EvaluationSubject["kind"];

function EvaluationLabRoute() {
  const search = Route.useSearch();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const pinned = usePinnedRuns();
  const modelCatalog = useModelCatalog();
  const [view, setView] = useState<EvalView>("lab");
  const [selectedSuiteId, setSelectedSuiteId] = useState(search.suite || "");
  const [selectedRevisionId, setSelectedRevisionId] = useState("");
  const [suiteEditorOpen, setSuiteEditorOpen] = useState(false);
  const [revisionEditorOpen, setRevisionEditorOpen] = useState(false);
  const [baseKind, setBaseKind] = useState<SubjectKind>("model");
  const [baseValue, setBaseValue] = useState("");
  const [candidateKind, setCandidateKind] = useState<SubjectKind>(search.runId ? "run" : "model");
  const [candidateValue, setCandidateValue] = useState(search.runId || "");
  const [additionalCandidateValues, setAdditionalCandidateValues] = useState<string[]>([]);
  const [verifierRevisionId, setVerifierRevisionId] = useState("");
  const [baseEvaluationId, setBaseEvaluationId] = useState(search.base || "");
  const initialComparisonCandidates = useMemo(
    () => (search.candidates ? search.candidates.split(",").filter(Boolean).slice(0, 4) : search.evaluation ? [search.evaluation] : []),
    [],
  );
  const [comparisonCandidateIds, setComparisonCandidateIds] = useState<string[]>(initialComparisonCandidates);
  const candidateEvaluationId = comparisonCandidateIds[0] || "";
  const activeSection: EvalSection = search.section || "suites";
  const setCandidateEvaluationId = (value: string) => setComparisonCandidateIds((current) => value ? [value, ...current.slice(1)].slice(0, 4) : current.slice(1));

  useEffect(() => {
    if (!(["compare", "failure-review"] as EvalSection[]).includes(activeSection)) return;
    const candidates = comparisonCandidateIds.join(",");
    if ((search.base || "") === baseEvaluationId && (search.candidates || "") === candidates) return;
    navigate({
      to: "/eval",
      search: { ...search, base: baseEvaluationId || undefined, candidates: candidates || undefined },
      replace: true,
    });
  }, [activeSection, baseEvaluationId, comparisonCandidateIds, navigate, search]);

  const suites = useQuery({
    queryKey: ["benchmark-suites"],
    queryFn: api.listBenchmarkSuites,
  });
  const evaluations = useQuery({
    queryKey: ["evaluations"],
    queryFn: () => api.listEvaluations(),
    refetchInterval: 8_000,
    refetchIntervalInBackground: false,
  });
  const runs = useQuery({ queryKey: ["runs", "evaluation-picker"], queryFn: () => api.listRuns({ limit: 200 }), retry: false });
  const artifacts = useQuery({ queryKey: ["model-artifacts", "evaluation-picker"], queryFn: () => api.listModelArtifacts({ limit: 200 }), retry: false });
  const qualifiedVerifiers = useQuery({ queryKey: ["verifier-profiles", "evaluation-picker", "qualified"], queryFn: () => api.listVerifierProfiles({ qualification: "pass", limit: 200 }), retry: false });
  const suiteDetail = useQuery({
    queryKey: ["benchmark-suites", selectedSuiteId],
    queryFn: () => api.benchmarkSuite(selectedSuiteId),
    enabled: Boolean(selectedSuiteId),
    retry: false,
  });
  const jobs = useQuery({
    queryKey: ["evaluation-jobs"],
    queryFn: api.listEvaluationJobs,
    refetchInterval: 3_000,
    refetchIntervalInBackground: false,
    retry: false,
  });

  useEffect(() => {
    if (selectedSuiteId || !suites.data?.items.length) return;
    setSelectedSuiteId(suites.data.items[0].id);
  }, [selectedSuiteId, suites.data?.items]);

  useEffect(() => {
    if (!search.runId) return;
    setCandidateKind("run");
    setCandidateValue(search.runId);
  }, [search.runId]);

  const selectedSuite = suiteDetail.data ?? suites.data?.items.find((suite) => suite.id === selectedSuiteId) ?? null;
  const selectedRevision = selectedSuite?.revisions?.find(
    (revision) => revision.id === selectedRevisionId,
  ) ?? selectedSuite?.latest_revision;
  const completed = useMemo(
    () => (evaluations.data?.items ?? []).filter((evaluation) => evaluation.status === "completed"),
    [evaluations.data?.items],
  );
  const comparisonReady = Boolean(
    completed.some((evaluation) => evaluation.id === baseEvaluationId) &&
    completed.some((evaluation) => evaluation.id === candidateEvaluationId),
  );
  const baseSubjectOptions = useMemo(() => evaluationSubjectOptions(baseKind, modelCatalog.data?.items ?? [], runs.data?.items ?? [], artifacts.data?.items ?? [], completed), [artifacts.data?.items, baseKind, completed, modelCatalog.data?.items, runs.data?.items]);
  const candidateSubjectOptions = useMemo(() => evaluationSubjectOptions(candidateKind, modelCatalog.data?.items ?? [], runs.data?.items ?? [], artifacts.data?.items ?? [], completed), [artifacts.data?.items, candidateKind, completed, modelCatalog.data?.items, runs.data?.items]);
  const verifierOptions = useMemo(() => (qualifiedVerifiers.data?.items ?? []).flatMap((profile) => { const revision = profile.latest_revision; return revision ? [{ value: revision.id, label: profile.name, description: `${revision.family.replace("_", " ")} · ${revision.modality} · ${revision.task_type}`, status: revision.alias || revision.qualification_state || "qualified", keywords: `${profile.id} ${revision.id}` }] : []; }), [qualifiedVerifiers.data?.items]);
  const suiteVerifierRevisionId = useMemo(() => { const ids = (selectedRevision?.items ?? []).map((item) => String(item.config?.verifier_profile_revision_id ?? "").trim()).filter(Boolean); return ids.length && new Set(ids).size === 1 ? ids[0] : ""; }, [selectedRevision?.items]);
  const suiteNeedsVerifier = Boolean(selectedRevision?.items.some((item) => item.adapter === "verifier"));
  const effectiveVerifierRevisionId = verifierRevisionId || suiteVerifierRevisionId;

  useEffect(() => {
    const requestedCandidate = candidateEvaluationId
      ? completed.find((evaluation) => evaluation.id === candidateEvaluationId)
      : completed.find((evaluation) =>
          (search.evaluation && evaluation.id === search.evaluation) ||
          (search.runId && (evaluation.run_id === search.runId || evaluation.subject.run_id === search.runId)),
        );
    if (requestedCandidate && !candidateEvaluationId) setCandidateEvaluationId(requestedCandidate.id);
    const candidate = requestedCandidate ?? completed.find((evaluation) => evaluation.id === candidateEvaluationId);
    if (!baseEvaluationId && candidate) {
      const baseline = completed.find(
        (evaluation) =>
          evaluation.id !== candidate.id &&
          evaluation.suite_revision_id === candidate.suite_revision_id,
      );
      if (baseline) setBaseEvaluationId(baseline.id);
    }
  }, [baseEvaluationId, candidateEvaluationId, completed, search.evaluation, search.runId]);

  const createSuite = useMutation({
    mutationFn: (payload: BenchmarkSuiteCreatePayload) => api.createBenchmarkSuite(payload),
    onSuccess: (suite) => {
      queryClient.invalidateQueries({ queryKey: ["benchmark-suites"] });
      setSelectedSuiteId(suite.id);
      setSuiteEditorOpen(false);
    },
  });
  const createRevision = useMutation({
    mutationFn: (payload: Omit<BenchmarkSuiteCreatePayload, "name" | "description">) => {
      if (!selectedSuiteId) throw new Error("Choose a suite before creating a revision.");
      return api.createBenchmarkSuiteRevision(selectedSuiteId, payload);
    },
    onSuccess: (revision) => {
      queryClient.invalidateQueries({ queryKey: ["benchmark-suites"] });
      queryClient.invalidateQueries({ queryKey: ["benchmark-suites", selectedSuiteId] });
      setSelectedRevisionId(revision.id);
      setRevisionEditorOpen(false);
    },
  });

  const launch = useMutation({
    mutationFn: async () => {
      if (!selectedRevision?.id) throw new Error("Choose a suite revision before launching.");
      if (!baseValue.trim()) throw new Error("Choose one base subject for a bounded comparison batch.");
      if (!candidateValue.trim()) throw new Error("Add a candidate model, run, or checkpoint.");
      const candidates = [candidateValue, ...additionalCandidateValues].map((value) => value.trim()).filter(Boolean).slice(0, 4);
      const batch = await api.launchEvaluationBatch({
        suite_revision_id: selectedRevision.id,
        base: subject(baseKind, baseValue),
        candidates: candidates.map((value) => subject(candidateKind, value)),
        verifier_profile_revision_id: effectiveVerifierRevisionId || undefined,
        reuse_completed: true,
      });
      return batch.evaluations ?? [];
    },
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["evaluations"] });
      queryClient.invalidateQueries({ queryKey: ["evaluation-jobs"] });
      if (created.length >= 2) {
        setBaseEvaluationId(created[0].id);
        setComparisonCandidateIds(created.slice(1, 5).map((item) => item.id));
      } else if (created.length === 1) {
        setCandidateEvaluationId(created[0].id);
      }
    },
  });

  const comparison = useQuery({
    queryKey: ["evaluations", "compare", baseEvaluationId, candidateEvaluationId, search.classification, search.page],
    queryFn: () => api.compareEvaluations(baseEvaluationId, candidateEvaluationId, ((search.page || 1) - 1) * 100, 100, { classification: search.classification }),
    enabled: comparisonReady,
    retry: 1,
  });
  const history = useQuery({
    queryKey: ["evaluations", "history", selectedRevision?.id],
    queryFn: () => api.evaluationHistory({ suiteRevisionId: selectedRevision?.id, limit: 40 }),
    enabled: Boolean(selectedRevision?.id),
    retry: false,
  });
  const drift = useQuery({
    queryKey: ["evaluations", "drift", baseEvaluationId, candidateEvaluationId],
    queryFn: () => api.evaluationDrift({ baseId: baseEvaluationId, candidateId: candidateEvaluationId, practicalDelta: 0 }),
    enabled: comparisonReady,
    retry: false,
  });

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Evaluate"
        subtitle="Build benchmark evidence, compare exact subjects, calibrate verifiers, and review consequential failures."
        actions={
          <>
            {activeSection === "results" ? <Button variant={view === "lab" ? "secondary" : "ghost"} size="sm" onClick={() => setView("lab")}><FlaskConical />Persistent</Button> : null}
            {activeSection === "results" ? <Button variant={view === "legacy" ? "secondary" : "ghost"} size="sm" onClick={() => setView("legacy")}><BarChart3 />Legacy</Button> : null}
            <Button variant="ghost" size="icon" asChild aria-label="Back to runs"><Link to="/runs"><ArrowLeft /></Link></Button>
          </>
        }
        statusBar={
          <>
            <Readout label="SUITES" value={String(suites.data?.items.length ?? 0)} />
            <span className="text-fg-disabled">·</span>
            <Readout label="EVALUATIONS" value={String(evaluations.data?.items.length ?? 0)} />
            <span className="text-fg-disabled">·</span>
            <Readout label="ACTIVE" value={String((jobs.data?.items ?? []).filter((job) => ["queued", "running"].includes(job.status)).length)} />
          </>
        }
      />
      <EvaluateTabs active={activeSection} />

      {activeSection === "environments" ? (
        <EnvironmentWorkspace
          selectedEnvironmentId={search.environment}
          selectedEpisodeId={search.episode}
          onEnvironment={(environment) => navigate({ to: "/eval", search: { ...search, section: "environments", environment, episode: undefined }, replace: true })}
          onEpisode={(episode) => navigate({ to: "/eval", search: { ...search, section: "environments", episode }, replace: true })}
        />
      ) : activeSection === "verifiers" ? (
        <VerifierReliabilityWorkspace
          view={search.verifierView || "catalog"}
          selectedProfileId={search.profile}
          selectedCalibrationId={search.calibration}
          onView={(verifierView) => navigate({ to: "/eval", search: { ...search, section: "verifiers", verifierView }, replace: true })}
          onProfile={(profile) => navigate({ to: "/eval", search: { ...search, section: "verifiers", profile }, replace: true })}
          onCalibration={(calibration) => navigate({ to: "/eval", search: { ...search, section: "verifiers", calibration }, replace: true })}
          auditView={search.auditView || "profiles"}
          selectedAuditId={search.audit}
          baseAuditId={search.auditBase}
          candidateAuditId={search.auditCandidate}
          selectedAuditSampleId={search.auditSample}
          auditPage={search.auditPage || 1}
          auditClassification={search.auditClassification}
          onAuditView={(auditView) => navigate({ to: "/eval", search: { ...search, section: "verifiers", verifierView: "training-audits", auditView, auditSample: undefined, auditPage: 1 }, replace: true })}
          onAudit={(audit) => navigate({ to: "/eval", search: { ...search, section: "verifiers", verifierView: "training-audits", audit, auditSample: undefined, auditPage: 1 }, replace: true })}
          onAuditCompare={(auditBase, auditCandidate) => navigate({ to: "/eval", search: { ...search, section: "verifiers", verifierView: "training-audits", auditView: "compare", auditBase, auditCandidate, auditSample: undefined, auditPage: 1 }, replace: true })}
          onAuditSample={(auditSample) => navigate({ to: "/eval", search: { ...search, auditSample }, replace: true })}
          onAuditPage={(auditPage) => navigate({ to: "/eval", search: { ...search, auditPage, auditSample: undefined }, replace: true })}
          onAuditClassification={(auditClassification) => navigate({ to: "/eval", search: { ...search, auditClassification, auditPage: 1, auditSample: undefined }, replace: true })}
        />
      ) : activeSection === "suites" || activeSection === "launch" ? (
        <div className="grid min-h-[calc(100vh-112px)] xl:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="border-b border-border-subtle bg-bg-subtle/30 xl:border-b-0 xl:border-r">
            <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
              <div><div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">Benchmark suites</div><div className="mt-0.5 text-[11px] text-fg-muted">Immutable revisions</div></div>
              <Button variant="ghost" size="icon" onClick={() => setSuiteEditorOpen((open) => !open)} aria-label="Create benchmark suite"><Plus /></Button>
            </div>
            {suiteEditorOpen ? <SuiteEditor mutation={createSuite} /> : null}
            <SuiteList items={suites.data?.items ?? []} selected={selectedSuiteId} onSelect={(suite) => { setSelectedSuiteId(suite); navigate({ to: "/eval", search: { ...search, suite }, replace: true }); }} loading={suites.isLoading} />
          </aside>

          <main className="min-w-0">
            {activeSection === "suites" ? <section>
              <SectionHeading
                eyebrow="SUITE WORKSPACE"
                title={selectedSuite?.name || "Choose a benchmark suite"}
                detail="Publish ordered items, typed evaluators, generation settings, and a direction-aware primary metric."
              />
              {selectedRevision ? <div className="grid gap-px border-y border-border bg-border md:grid-cols-4"><MetricBlock label="Purpose" value={selectedSuite?.purpose || "unspecified"} /><MetricBlock label="Revision" value={`r${selectedRevision.revision}`} /><MetricBlock label="Ordered items" value={String(selectedRevision.items.length)} /><MetricBlock label="Primary metric" value={`${selectedRevision.primary_metric} ${selectedRevision.direction === "minimize" ? "↓" : "↑"}`} /></div> : <Empty label="Create or select an immutable benchmark suite revision." />}
              <div className="flex flex-wrap items-center gap-2 px-5 py-4"><Button size="sm" variant="primary" onClick={() => navigate({ to: "/eval", search: { ...search, section: "launch", suite: selectedSuiteId || undefined } })} disabled={!selectedRevision}><Play />Launch this suite</Button><Button size="sm" variant="secondary" onClick={() => setRevisionEditorOpen((open) => !open)} disabled={!selectedRevision}><Plus />New revision</Button></div>
              {revisionEditorOpen && selectedRevision ? <RevisionEditor revision={selectedRevision} mutation={createRevision} /> : null}
            </section> : <section>
              <SectionHeading
                eyebrow="LAUNCH"
                title={selectedSuite?.name || "Choose a benchmark suite"}
                detail={selectedRevision ? `${selectedRevision.items.length} items · ${selectedRevision.primary_metric} · ${selectedRevision.direction}` : "Create or select an immutable suite revision."}
              />
              <EvaluationLauncher
                suite={selectedSuite}
                revision={selectedRevision ?? null}
                revisions={selectedSuite?.revisions ?? (selectedSuite?.latest_revision ? [selectedSuite.latest_revision] : [])}
                onRevision={setSelectedRevisionId}
                baseKind={baseKind}
                baseValue={baseValue}
                candidateKind={candidateKind}
                candidateValue={candidateValue}
                additionalCandidateValues={additionalCandidateValues}
                onBaseKind={setBaseKind}
                onBaseValue={setBaseValue}
                onCandidateKind={setCandidateKind}
                onCandidateValue={setCandidateValue}
                onAdditionalCandidateValues={setAdditionalCandidateValues}
                baseOptions={baseSubjectOptions}
                candidateOptions={candidateSubjectOptions}
                verifierRevisionId={effectiveVerifierRevisionId}
                verifierOptions={verifierOptions}
                verifierRequired={suiteNeedsVerifier && !suiteVerifierRevisionId}
                onVerifierRevision={setVerifierRevisionId}
                onLaunch={() => launch.mutate()}
                onNewRevision={() => setRevisionEditorOpen((open) => !open)}
                launching={launch.isPending}
                error={launch.isError ? (launch.error as Error).message : null}
              />
              {revisionEditorOpen && selectedRevision ? (
                <RevisionEditor revision={selectedRevision} mutation={createRevision} />
              ) : null}
              <div className="border-t border-border-subtle"><SectionHeading eyebrow="ACTIVITY" title="Evaluation queue" detail="One active standalone evaluation at a time; every attempt remains visible in Activity." /><EvaluationJobs items={jobs.data?.items ?? []} loading={jobs.isLoading} /></div>
            </section>}
          </main>
        </div>
      ) : activeSection === "results" ? (
        view === "legacy" ? <LegacyCohort pinned={pinned} /> : <main className="min-w-0"><section><SectionHeading eyebrow="RESULTS" title="Immutable evaluation evidence" detail="Completed subjects and their primary metrics; training loss never counts as evaluation." /><LongitudinalEvidence items={history.data?.items ?? []} drift={drift.data} loading={history.isLoading} /><EvaluationTable items={evaluations.data?.items ?? []} loading={evaluations.isLoading} /></section></main>
      ) : (
        <main className="min-w-0">
          <section>
            <SectionHeading eyebrow={activeSection === "failure-review" ? "FAILURE REVIEW" : "COMPARE"} title={activeSection === "failure-review" ? "Inspect and review consequential evidence" : "One base, up to four candidates"} detail={activeSection === "failure-review" ? "Filter per-example evidence, preserve verifier traces, and create reviewed proposals from development failures only." : "Base / candidate delta across subjects using the same immutable suite revision; every delta respects metric direction."} />
            <ComparisonCandidateBar
              evaluations={completed}
              baseId={baseEvaluationId}
              candidateIds={comparisonCandidateIds}
              onBase={setBaseEvaluationId}
              onCandidates={setComparisonCandidateIds}
            />
            <EvaluationComparisonPanel
              evaluations={completed}
              baseId={baseEvaluationId}
              candidateId={candidateEvaluationId}
              onBase={setBaseEvaluationId}
              onCandidate={setCandidateEvaluationId}
              data={comparison.data}
              loading={comparison.isLoading}
              error={comparison.isError ? (comparison.error as Error).message : null}
            />
          </section>
        </main>
      )}
    </>
  );
}

function EvaluateTabs({ active }: { active: EvalSection }) {
  const navigate = useNavigate();
  const search = Route.useSearch();
  const tabs: Array<{ id: EvalSection; label: string }> = [
    { id: "suites", label: "Suites" },
    { id: "launch", label: "Launch" },
    { id: "results", label: "Results" },
    { id: "compare", label: "Compare" },
    { id: "failure-review", label: "Failure Review" },
    { id: "verifiers", label: "Verifiers" },
    { id: "environments", label: "Environments" },
  ];
  return (
    <div className="flex overflow-x-auto border-b border-border bg-bg-subtle/55 px-4 md:px-5">
      {tabs.map((tab) => <button key={tab.id} type="button" onClick={() => navigate({ to: "/eval", search: { ...search, section: tab.id } })} className={cn("relative h-10 shrink-0 px-3 text-[11.5px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset", active === tab.id ? "font-medium text-fg" : "text-fg-subtle hover:text-fg")}>{tab.label}{active === tab.id ? <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" /> : null}</button>)}
    </div>
  );
}

function EnvironmentWorkspace({
  selectedEnvironmentId,
  selectedEpisodeId,
  onEnvironment,
  onEpisode,
}: {
  selectedEnvironmentId?: string;
  selectedEpisodeId?: string;
  onEnvironment: (id: string) => void;
  onEpisode: (id: string) => void;
}) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("Deterministic task environment");
  const [goal, setGoal] = useState("Set the reviewed state and finish.");
  const [stateKey, setStateKey] = useState("status");
  const [stateValue, setStateValue] = useState("complete");
  const [preparedSuiteRevisionId, setPreparedSuiteRevisionId] = useState("");
  const [preparedEnvironmentRevisionId, setPreparedEnvironmentRevisionId] = useState("");
  const [subjectRef, setSubjectRef] = useState("halo-forge");
  const [serveUrl, setServeUrl] = useState("http://127.0.0.1:8001/v1");
  const environments = useQuery({
    queryKey: ["agent-environments"],
    queryFn: () => api.agentEnvironments({ limit: 200 }),
    retry: false,
  });
  const episodes = useQuery({
    queryKey: ["agent-episodes"],
    queryFn: () => api.agentEpisodes({ limit: 200 }),
    refetchInterval: 4_000,
    retry: false,
  });
  const selectedEnvironment = (environments.data?.items ?? []).find((item) => item.id === selectedEnvironmentId)
    ?? environments.data?.items?.[0]
    ?? null;
  const selectedEpisode = (episodes.data?.items ?? []).find((item) => item.id === selectedEpisodeId)
    ?? episodes.data?.items?.[0]
    ?? null;
  const steps = useQuery({
    queryKey: ["agent-episodes", selectedEpisode?.id, "steps"],
    queryFn: () => api.agentEpisodeSteps(selectedEpisode!.id, { limit: 200 }),
    enabled: Boolean(selectedEpisode?.id),
    retry: false,
  });
  const permissions = useQuery({
    queryKey: ["environment-permissions", preparedEnvironmentRevisionId],
    queryFn: () => api.environmentPermissions(preparedEnvironmentRevisionId),
    enabled: Boolean(preparedEnvironmentRevisionId),
    retry: false,
  });
  const prepare = useMutation({
    mutationFn: async () => {
      const environment = await api.createAgentEnvironment({
        name: name.trim(),
        description: "Attempt-local deterministic state-machine environment",
      });
      const revision = await api.createAgentEnvironmentRevision(environment.id, {
        adapter_id: "state_machine",
        initial_state: { [stateKey]: "pending" },
        transitions: {
          set_reviewed_state: {
            state_delta: { [stateKey]: stateValue },
            reward: 1,
            terminal: false,
            result: { ok: true },
          },
          finish: {
            state_delta: {},
            reward: 1,
            terminal: true,
            terminal_reason: "goal_reached",
            result: { ok: true },
          },
        },
        tools: [
          { name: "set_reviewed_state", input_schema: { type: "object" } },
          { name: "finish", input_schema: { type: "object" } },
        ],
        max_steps: 8,
      });
      const revisionId = String(revision.id || "");
      const suite = await api.createAgentEpisodeSuite(revisionId, {
        name: `${name.trim()} development suite`,
        purpose: "development",
      });
      const suiteRevision = await api.createAgentEpisodeSuiteRevision(String(suite.id || ""), {
        environment_revision_id: revisionId,
        max_steps: 8,
        generation: { seed: 42, temperature: 0 },
        items: [
          {
            id: "guided-task",
            goal: goal.trim(),
            initial_state: { [stateKey]: "pending" },
            expected_state: { [stateKey]: stateValue },
          },
        ],
      });
      return { environment, revisionId, suiteRevisionId: String(suiteRevision.id || "") };
    },
    onSuccess: ({ environment, revisionId, suiteRevisionId }) => {
      queryClient.invalidateQueries({ queryKey: ["agent-environments"] });
      onEnvironment(environment.id);
      setPreparedEnvironmentRevisionId(revisionId);
      setPreparedSuiteRevisionId(suiteRevisionId);
    },
  });
  const run = useMutation({
    mutationFn: () => api.launchAgentEpisode(preparedSuiteRevisionId, {
      suite_item_id: "guided-task",
      subject_type: "served_model",
      subject_ref: subjectRef,
      serve_url: serveUrl,
      seed: 42,
    }),
    onSuccess: (episode) => {
      queryClient.invalidateQueries({ queryKey: ["agent-episodes"] });
      onEpisode(episode.id);
    },
  });
  const replay = useMutation({
    mutationFn: () => api.replayAgentEpisode(selectedEpisode!.id),
  });
  const rerun = useMutation({
    mutationFn: () => api.rerunAgentEpisode(selectedEpisode!.id, {
      subject_ref: subjectRef,
      serve_url: serveUrl,
    }),
    onSuccess: (episode) => {
      queryClient.invalidateQueries({ queryKey: ["agent-episodes"] });
      onEpisode(episode.id);
    },
  });
  return (
    <div className="grid min-h-[calc(100vh-112px)] xl:grid-cols-[280px_minmax(0,1fr)_320px]">
      <aside className="border-b border-border-subtle bg-bg-subtle/30 xl:border-b-0 xl:border-r">
        <div className="border-b border-border-subtle px-4 py-3"><div className="text-[10px] uppercase tracking-[0.14em] text-fg-disabled">Local environments</div><div className="mt-1 text-[11px] text-fg-muted">{environments.data?.total ?? 0} immutable definitions</div></div>
        <div className="divide-y divide-border-subtle">{(environments.data?.items ?? []).map((environment) => <button type="button" key={environment.id} onClick={() => onEnvironment(environment.id)} className={cn("w-full px-4 py-3 text-left hover:bg-surface", selectedEnvironment?.id === environment.id && "bg-accent-bg/45")}><div className="text-[11.5px] font-medium text-fg">{environment.name}</div><div className="mt-1 text-[10px] text-fg-subtle">{environment.latest_revision_id ? "Versioned and ready" : "Definition pending"}</div></button>)}</div>
      </aside>
      <main className="min-w-0 border-b border-border-subtle xl:border-b-0 xl:border-r">
        <SectionHeading eyebrow="ENVIRONMENT STUDIO" title={selectedEnvironment?.name || "Test a model in a local environment"} detail="Choose a local task, confirm its permissions, run the model, then compare an exact replay with a fresh model rerun." />
        {!selectedEnvironment || !preparedSuiteRevisionId ? <div className="grid gap-4 px-5 py-5 sm:grid-cols-2"><FieldLabel label="Environment template"><Input value={name} onChange={(event) => setName(event.target.value)} /></FieldLabel><FieldLabel label="Success condition"><Input value={goal} onChange={(event) => setGoal(event.target.value)} /></FieldLabel><FieldLabel label="State to check"><Input value={stateKey} onChange={(event) => setStateKey(event.target.value)} /></FieldLabel><FieldLabel label="Successful value"><Input value={stateValue} onChange={(event) => setStateValue(event.target.value)} /></FieldLabel><div className="sm:col-span-2"><Button onClick={() => prepare.mutate()} disabled={!name.trim() || !goal.trim() || !stateKey.trim() || prepare.isPending}>{prepare.isPending ? <Loader2 className="animate-spin" /> : <Plus />}Review environment</Button>{prepare.error instanceof Error ? <p className="mt-2 text-[10px] text-danger">{prepare.error.message}</p> : null}</div></div> : <div className="space-y-4 px-5 py-5"><div className="grid gap-px border-y border-border-subtle bg-border-subtle sm:grid-cols-3"><MetricBlock label="Runtime" value="Deterministic local" /><MetricBlock label="Time limit" value={`${permissions.data?.timeout_seconds ?? 60}s`} /><MetricBlock label="External writes" value={permissions.data?.external_writes ? "Enabled" : "Disabled"} /></div><div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="text-xs font-medium text-fg">Permission summary</div><div className="mt-1 text-[10.5px] leading-5 text-fg-muted">Local files: {permissions.data?.local_files ? "allowed in the temporary workspace" : "off"} · local SQLite: {permissions.data?.local_sqlite ? "allowed" : "off"} · loopback test services: {permissions.data?.loopback_services ? "allowed" : "off"} · external writes: disabled.</div></div><div className="grid gap-4 sm:grid-cols-2"><FieldLabel label="Model subject"><Input value={subjectRef} onChange={(event) => setSubjectRef(event.target.value)} /></FieldLabel><FieldLabel label="Local serving endpoint"><Input value={serveUrl} onChange={(event) => setServeUrl(event.target.value)} /></FieldLabel></div><div className="flex flex-wrap gap-2"><Button onClick={() => run.mutate()} disabled={run.isPending || !subjectRef.trim()}>{run.isPending ? <Loader2 className="animate-spin" /> : <Play />}Run model in environment</Button>{selectedEpisode ? <Button variant="secondary" onClick={() => replay.mutate()} disabled={replay.isPending}><RefreshCw className={cn(replay.isPending && "animate-spin")} />Replay the same actions</Button> : null}{selectedEpisode ? <Button variant="secondary" onClick={() => rerun.mutate()} disabled={rerun.isPending}><RefreshCw className={cn(rerun.isPending && "animate-spin")} />Run the model again</Button> : null}</div></div>}
        {selectedEpisode ? <section className="border-t border-border-subtle"><SectionHeading eyebrow="EPISODE EVIDENCE" title={`Episode ${selectedEpisode.status}`} detail={`${selectedEpisode.suite_item_id} · seed ${selectedEpisode.seed} · ${String(selectedEpisode.metrics.task_success ?? "—")} task success`} /><div className="divide-y divide-border-subtle">{(steps.data?.items ?? []).map((step, index) => <div key={index} className="grid gap-2 px-5 py-3 sm:grid-cols-[80px_160px_minmax(0,1fr)]"><div className="text-[10px] text-fg-disabled">Step {String(step.ordinal ?? index)}</div><div className="font-mono text-[10px] text-fg-muted">{String((step.action as Record<string, unknown> | undefined)?.name ?? "action")}</div><div className="text-[10.5px] text-fg-subtle">{step.error ? String(step.error) : `State ${String(step.state_hash ?? "").slice(0, 12)}`}</div></div>)}</div></section> : null}
      </main>
      <aside className="bg-bg-subtle/25 px-4 py-5"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Episode history</div><div className="mt-3 divide-y divide-border-subtle">{(episodes.data?.items ?? []).slice(0, 20).map((episode) => <button type="button" key={episode.id} onClick={() => onEpisode(episode.id)} className={cn("w-full py-3 text-left", selectedEpisode?.id === episode.id && "text-accent")}><div className="text-[10.5px] font-medium">{episode.suite_item_id}</div><div className="mt-1 text-[9.5px] text-fg-subtle">{episode.status} · {String(episode.metrics.step_count ?? 0)} steps</div></button>)}</div><p className="mt-5 text-[10px] leading-5 text-fg-subtle">Trace replay repeats recorded actions exactly. A model rerun invokes the subject again and remains separate evidence.</p></aside>
    </div>
  );
}

function SuiteList({ items, selected, onSelect, loading }: { items: BenchmarkSuite[]; selected: string; onSelect: (id: string) => void; loading: boolean }) {
  if (loading) return <SmallLoading label="Loading suites" />;
  if (!items.length) return <Empty label="No benchmark suites yet. Create one to start persistent evaluation." />;
  return <div className="divide-y divide-border-subtle">{items.map((suite) => <button key={suite.id} type="button" onClick={() => onSelect(suite.id)} className={cn("w-full px-4 py-3 text-left transition-colors", selected === suite.id ? "bg-accent/8" : "hover:bg-surface-hover/35")}><div className="flex items-center justify-between gap-2"><span className={cn("truncate text-[12px] font-medium", selected === suite.id ? "text-accent" : "text-fg")}>{suite.name}</span><Badge tone={selected === suite.id ? "accent" : "neutral"} size="sm">r{suite.latest_revision?.revision ?? suite.revision_count ?? "—"}</Badge></div><div className="mt-1 truncate text-[10.5px] text-fg-muted">{suite.purpose || "unspecified"} · {suite.latest_revision?.primary_metric || "No revision"} · {suite.latest_revision?.direction || "unconfigured"}</div></button>)}</div>;
}

function SuiteEditor({ mutation }: { mutation: ReturnType<typeof useMutation<BenchmarkSuite, Error, BenchmarkSuiteCreatePayload>> }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [purpose, setPurpose] = useState<"development" | "holdout" | "unspecified">("development");
  const [metric, setMetric] = useState("accuracy");
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [items, setItems] = useState<BenchmarkSuiteItem[]>([{ adapter: "lm_eval", task: "gsm8k" }]);
  const [settings, setSettings] = useState<Record<string, unknown>>({ temperature: 0, max_tokens: 512 });
  const [rawItems, setRawItems] = useState(JSON.stringify(items, null, 2));
  const [rawSettings, setRawSettings] = useState(JSON.stringify(settings, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);

  function submit() {
    if (!items.length) { setParseError("Add at least one benchmark item."); return; }
    setParseError(null);
    mutation.mutate({ name: name.trim(), description: description.trim(), purpose, primary_metric: metric.trim(), direction, items, generation_settings: settings });
  }

  function applyRawItems(value: string) { setRawItems(value); try { const parsed = JSON.parse(value); if (!Array.isArray(parsed)) throw new Error("Items must be an array."); setItems(parsed); setParseError(null); } catch (error) { setParseError((error as Error).message); } }
  function applyRawSettings(value: string) { setRawSettings(value); try { const parsed = JSON.parse(value); if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Settings must be an object."); setSettings(parsed); setParseError(null); } catch (error) { setParseError((error as Error).message); } }

  return (
    <div className="space-y-2 border-b border-border-subtle bg-bg px-4 py-3">
      <FieldLabel label="Name"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Core reasoning" /></FieldLabel>
      <FieldLabel label="Description"><Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Stable held-out reasoning suite" /></FieldLabel>
      <FieldLabel label="Purpose"><select value={purpose} onChange={(event) => setPurpose(event.target.value as typeof purpose)} className={selectClass}><option value="development">development · guides selection</option><option value="holdout">holdout · final confirmation only</option><option value="unspecified">unspecified · legacy behavior</option></select></FieldLabel>
      <div className="grid grid-cols-2 gap-2"><FieldLabel label="Primary metric"><Input value={metric} onChange={(event) => setMetric(event.target.value)} /></FieldLabel><FieldLabel label="Direction"><select value={direction} onChange={(event) => setDirection(event.target.value as "maximize" | "minimize")} className={selectClass}><option value="maximize">maximize</option><option value="minimize">minimize</option></select></FieldLabel></div>
      <FieldLabel label="Ordered benchmark items"><OrderedSuiteItems items={items} onChange={(next) => { setItems(next); setRawItems(JSON.stringify(next, null, 2)); }} /></FieldLabel>
      <div className="grid grid-cols-2 gap-2"><FieldLabel label="Temperature"><Input type="number" step="0.1" value={String(settings.temperature ?? 0)} onChange={(event) => { const next = { ...settings, temperature: Number(event.target.value) }; setSettings(next); setRawSettings(JSON.stringify(next, null, 2)); }} /></FieldLabel><FieldLabel label="Maximum output tokens"><Input type="number" value={String(settings.max_tokens ?? 512)} onChange={(event) => { const next = { ...settings, max_tokens: Number(event.target.value) }; setSettings(next); setRawSettings(JSON.stringify(next, null, 2)); }} /></FieldLabel></div>
      <details><summary className="cursor-pointer text-[9.5px] uppercase tracking-wider text-fg-disabled hover:text-fg">Advanced · edit JSON</summary><div className="mt-2 space-y-2"><FieldLabel label="Ordered items JSON"><textarea value={rawItems} onChange={(event) => applyRawItems(event.target.value)} rows={5} className={textareaClass} /></FieldLabel><FieldLabel label="Generation settings JSON"><textarea value={rawSettings} onChange={(event) => applyRawSettings(event.target.value)} rows={3} className={textareaClass} /></FieldLabel></div></details>
      {parseError || mutation.isError ? <div className="text-[10px] text-danger">{parseError || mutation.error?.message || "Suite creation failed."}</div> : null}
      <Button variant="primary" size="sm" className="w-full" disabled={!name.trim() || !metric.trim() || mutation.isPending} onClick={submit}>{mutation.isPending ? <Loader2 className="animate-spin" /> : <Plus />}Create suite</Button>
    </div>
  );
}

function OrderedSuiteItems({ items, onChange }: { items: BenchmarkSuiteItem[]; onChange: (items: BenchmarkSuiteItem[]) => void }) {
  function update(index: number, patch: Partial<BenchmarkSuiteItem>) { onChange(items.map((item, position) => position === index ? { ...item, ...patch } : item)); }
  function move(index: number, delta: number) { const next = [...items]; const target = index + delta; if (target < 0 || target >= next.length) return; [next[index], next[target]] = [next[target], next[index]]; onChange(next); }
  return <div className="overflow-hidden rounded-md border border-border"><div className="divide-y divide-border-subtle">{items.map((item, index) => { const adapter = String(item.adapter || "lm_eval"); const target = String(item.task ?? item.dataset_version_id ?? item.config?.verifier_profile_revision_id ?? ""); return <div key={index} className="grid gap-2 bg-bg p-2 sm:grid-cols-[24px_110px_minmax(0,1fr)_auto]"><span className="pt-2 text-center font-mono text-[9px] text-fg-disabled">{index + 1}</span><select aria-label={`Item ${index + 1} evaluator`} value={adapter} onChange={(event) => update(index, { adapter: event.target.value })} className={selectClass}><option value="lm_eval">lm-eval task</option><option value="dataset_split">Dataset split</option><option value="verifier">Verifier-backed</option><option value="code">Code / pass@k</option><option value="reasoning">Reasoning</option><option value="tool_use">Tool use</option><option value="vlm">VLM</option><option value="audio">Audio</option></select><Input aria-label={`Item ${index + 1} target`} value={target} onChange={(event) => adapter === "dataset_split" ? update(index, { dataset_version_id: event.target.value }) : adapter === "verifier" ? update(index, { config: { ...(item.config ?? {}), verifier_profile_revision_id: event.target.value } }) : update(index, { task: event.target.value })} placeholder={adapter === "dataset_split" ? "Choose a dataset version in the full editor" : adapter === "verifier" ? "Choose a qualified verifier profile" : "Task or evaluator target"} /><div className="flex items-center"><button type="button" className="px-1 text-fg-disabled hover:text-fg disabled:opacity-30" disabled={index === 0} onClick={() => move(index, -1)} aria-label={`Move item ${index + 1} up`}>↑</button><button type="button" className="px-1 text-fg-disabled hover:text-fg disabled:opacity-30" disabled={index === items.length - 1} onClick={() => move(index, 1)} aria-label={`Move item ${index + 1} down`}>↓</button><button type="button" className="px-1 text-fg-disabled hover:text-danger disabled:opacity-30" disabled={items.length === 1} onClick={() => onChange(items.filter((_, position) => position !== index))} aria-label={`Remove item ${index + 1}`}>×</button></div></div>; })}</div><button type="button" className="w-full border-t border-border-subtle px-3 py-2 text-left text-[9.5px] text-accent hover:bg-accent/5" onClick={() => onChange([...items, { adapter: "lm_eval", task: "" }])}><Plus className="mr-1 inline h-3 w-3" />Add ordered item</button></div>;
}

function EvaluationLauncher({ suite, revision, revisions, onRevision, baseKind, baseValue, candidateKind, candidateValue, additionalCandidateValues, onBaseKind, onBaseValue, onCandidateKind, onCandidateValue, onAdditionalCandidateValues, baseOptions, candidateOptions, verifierRevisionId, verifierOptions, verifierRequired, onVerifierRevision, onLaunch, onNewRevision, launching, error }: { suite: BenchmarkSuite | null; revision: BenchmarkSuiteRevision | null; revisions: BenchmarkSuiteRevision[]; onRevision: (revisionId: string) => void; baseKind: SubjectKind; baseValue: string; candidateKind: SubjectKind; candidateValue: string; additionalCandidateValues: string[]; onBaseKind: (kind: SubjectKind) => void; onBaseValue: (value: string) => void; onCandidateKind: (kind: SubjectKind) => void; onCandidateValue: (value: string) => void; onAdditionalCandidateValues: (values: string[]) => void; baseOptions: SubjectPickerOption[]; candidateOptions: SubjectPickerOption[]; verifierRevisionId: string; verifierOptions: SubjectPickerOption[]; verifierRequired: boolean; onVerifierRevision: (value: string) => void; onLaunch: () => void; onNewRevision: () => void; launching: boolean; error: string | null }) {
  return (
    <div className="grid gap-px border-t border-border-subtle bg-border-subtle lg:grid-cols-[1fr_1fr_180px]">
      <SubjectEditor label="Base" kind={baseKind} value={baseValue} options={baseOptions} onKind={onBaseKind} onValue={onBaseValue} />
      <div className="bg-bg"><SubjectEditor label="Candidate 1" kind={candidateKind} value={candidateValue} options={candidateOptions} onKind={onCandidateKind} onValue={onCandidateValue} />{additionalCandidateValues.map((value, index) => <div key={index} className="relative border-t border-border-subtle px-4 py-3"><FieldLabel label={`Candidate ${index + 2}`}><SearchPicker value={value} onChange={(next) => onAdditionalCandidateValues(additionalCandidateValues.map((item, position) => position === index ? next : item))} options={candidateOptions.filter((item) => item.value === value || ![candidateValue, ...additionalCandidateValues].includes(item.value))} placeholder="Search compatible subjects" emptyLabel="No additional compatible subject" /></FieldLabel><button type="button" aria-label={`Remove candidate ${index + 2}`} onClick={() => onAdditionalCandidateValues(additionalCandidateValues.filter((_, position) => position !== index))} className="absolute right-4 top-2 text-[9.5px] text-fg-disabled hover:text-danger">Remove</button></div>)}{additionalCandidateValues.length < 3 ? <button type="button" onClick={() => onAdditionalCandidateValues([...additionalCandidateValues, ""])} className="mx-4 mb-3 inline-flex items-center gap-1 text-[10px] text-accent hover:underline"><Plus className="h-3 w-3" />Add candidate</button> : null}</div>
      <div className="flex flex-col justify-between gap-3 bg-bg px-4 py-4"><div><div className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled">Suite revision</div><select aria-label="Suite revision" value={revision?.id || ""} onChange={(event) => onRevision(event.target.value)} disabled={!revisions.length} className={`${selectClass} mt-1 w-full font-mono text-[10px]`}>{!revisions.length ? <option value="">—</option> : revisions.map((item) => <option key={item.id} value={item.id}>r{item.revision} · {item.id}</option>)}</select><button type="button" disabled={!revision || !suite} onClick={onNewRevision} className="mt-2 text-[10px] text-accent hover:underline disabled:text-fg-disabled">Edit as new revision</button><div className="mt-4"><FieldLabel label={verifierRequired ? "Qualified verifier · required" : "Qualified verifier · optional"}><SearchPicker value={verifierRevisionId} onChange={onVerifierRevision} options={verifierOptions} allowEmpty={!verifierRequired} placeholder="Choose an exact revision" emptyLabel="No compatible qualified verifier is available" /></FieldLabel><p className="text-[8.5px] leading-4 text-fg-disabled">The same immutable verifier revision is bound to every subject in this comparison.</p></div></div><Button variant="primary" size="sm" disabled={!revision || !baseValue.trim() || !candidateValue.trim() || additionalCandidateValues.some((value) => !value.trim()) || (verifierRequired && !verifierRevisionId) || launching} onClick={onLaunch}>{launching ? <Loader2 className="animate-spin" /> : <Play />}Launch {1 + additionalCandidateValues.length} candidate{additionalCandidateValues.length ? "s" : ""}</Button>{error ? <div className="text-[10px] leading-4 text-danger">{error}</div> : null}</div>
    </div>
  );
}

function RevisionEditor({ revision, mutation }: { revision: BenchmarkSuiteRevision; mutation: ReturnType<typeof useMutation<BenchmarkSuiteRevision, Error, Omit<BenchmarkSuiteCreatePayload, "name" | "description">>> }) {
  const [metric, setMetric] = useState(revision.primary_metric);
  const [direction, setDirection] = useState<"maximize" | "minimize">(revision.direction);
  const [items, setItems] = useState<BenchmarkSuiteItem[]>(revision.items);
  const [settings, setSettings] = useState<Record<string, unknown>>(revision.generation_settings);
  const [rawItems, setRawItems] = useState(JSON.stringify(revision.items, null, 2));
  const [rawSettings, setRawSettings] = useState(JSON.stringify(revision.generation_settings, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);
  function submit() {
    if (!items.length) { setParseError("Add at least one ordered item."); return; }
    setParseError(null);
    mutation.mutate({ primary_metric: metric, direction, items, generation_settings: settings });
  }
  function applyItems(value: string) { setRawItems(value); try { const parsed = JSON.parse(value); if (!Array.isArray(parsed)) throw new Error("Items must be an array."); setItems(parsed); setParseError(null); } catch (error) { setParseError((error as Error).message); } }
  function applySettings(value: string) { setRawSettings(value); try { const parsed = JSON.parse(value); if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Settings must be an object."); setSettings(parsed); setParseError(null); } catch (error) { setParseError((error as Error).message); } }
  return <div className="border-t border-border-subtle bg-accent/5 px-5 py-4"><div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4"><FieldLabel label="Primary metric"><Input value={metric} onChange={(event) => setMetric(event.target.value)} /></FieldLabel><FieldLabel label="Direction"><select value={direction} onChange={(event) => setDirection(event.target.value as "maximize" | "minimize")} className={selectClass}><option value="maximize">maximize</option><option value="minimize">minimize</option></select></FieldLabel><FieldLabel label="Temperature"><Input type="number" step="0.1" value={String(settings.temperature ?? 0)} onChange={(event) => { const next = { ...settings, temperature: Number(event.target.value) }; setSettings(next); setRawSettings(JSON.stringify(next, null, 2)); }} /></FieldLabel><FieldLabel label="Maximum output tokens"><Input type="number" value={String(settings.max_tokens ?? 512)} onChange={(event) => { const next = { ...settings, max_tokens: Number(event.target.value) }; setSettings(next); setRawSettings(JSON.stringify(next, null, 2)); }} /></FieldLabel></div><div className="mt-3"><FieldLabel label="Ordered benchmark items"><OrderedSuiteItems items={items} onChange={(next) => { setItems(next); setRawItems(JSON.stringify(next, null, 2)); }} /></FieldLabel></div><details className="mt-3"><summary className="cursor-pointer text-[9.5px] uppercase tracking-wider text-fg-disabled hover:text-fg">Advanced · edit JSON</summary><div className="mt-2 grid gap-3 lg:grid-cols-2"><FieldLabel label="Ordered items JSON"><textarea value={rawItems} onChange={(event) => applyItems(event.target.value)} rows={6} className={textareaClass} /></FieldLabel><FieldLabel label="Generation settings JSON"><textarea value={rawSettings} onChange={(event) => applySettings(event.target.value)} rows={6} className={textareaClass} /></FieldLabel></div></details><div className="mt-3 flex items-center justify-end gap-3">{parseError || mutation.isError ? <div className="text-[9.5px] text-danger">{parseError || mutation.error?.message || "Revision failed."}</div> : null}<Button variant="primary" size="sm" disabled={mutation.isPending || !metric.trim() || Boolean(parseError)} onClick={submit}>{mutation.isPending ? <Loader2 className="animate-spin" /> : <Plus />}Publish revision</Button></div></div>;
}

function SubjectEditor({ label, kind, value, options, optional, onKind, onValue }: { label: string; kind: SubjectKind; value: string; options: SubjectPickerOption[]; optional?: boolean; onKind: (kind: SubjectKind) => void; onValue: (value: string) => void }) {
  return <div className="bg-bg px-5 py-4"><div className="mb-2 text-[10px] font-medium uppercase tracking-[0.12em] text-fg-subtle">{label}</div><div className="grid gap-2 sm:grid-cols-[130px_1fr]"><select value={kind} onChange={(event) => { onKind(event.target.value as SubjectKind); onValue(""); }} className={selectClass}><option value="model">Pinned model</option><option value="run">Completed run</option><option value="final_model">Final model</option><option value="checkpoint">Checkpoint</option></select><SearchPicker value={value} onChange={onValue} options={options} allowEmpty={optional} placeholder={`Search ${friendlySubjectKind(kind).toLowerCase()}s`} emptyLabel={`No compatible ${friendlySubjectKind(kind).toLowerCase()} is available`} /></div><details className="mt-2"><summary className="cursor-pointer text-[9.5px] uppercase tracking-wider text-fg-disabled hover:text-fg">Advanced · use an unlisted identifier</summary><Input className="mt-2" value={value} onChange={(event) => onValue(event.target.value)} placeholder={kind === "model" ? "Repository@revision" : kind === "checkpoint" ? "run:checkpoint path" : "Run or local path"} /></details></div>;
}

function EvaluationJobs({ items, loading }: { items: Evaluation[]; loading: boolean }) {
  const queryClient = useQueryClient();
  const cancel = useMutation({ mutationFn: api.cancelEvaluation, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["evaluation-jobs"] }) });
  const retry = useMutation({ mutationFn: api.retryEvaluation, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["evaluation-jobs"] }) });
  if (loading) return <SmallLoading label="Loading queue" />;
  if (!items.length) return <Empty label="No evaluation jobs are queued or running." />;
  return <div className="divide-y divide-border-subtle border-t border-border-subtle">{items.map((job) => <div key={job.id} className="grid gap-3 px-5 py-3 md:grid-cols-[minmax(180px,1fr)_150px_120px_auto] md:items-center"><div className="min-w-0"><div className="truncate font-mono text-[10.5px] text-fg">{job.id}</div><div className="mt-0.5 truncate text-[10px] text-fg-muted">{job.stage || job.subject.kind} · {job.subject.value}</div>{job.logs?.length ? <div className="mt-0.5 truncate font-mono text-[9px] text-fg-disabled" title={job.logs.at(-1)}>{job.logs.at(-1)}</div> : null}</div><Progress value={job.progress_percent} /><div className="font-mono text-[10px] text-fg-subtle">{job.processed_samples ?? 0}/{job.total_samples ?? "?"}</div><div className="flex justify-end gap-1">{["queued", "running"].includes(job.status) ? <Button variant="ghost" size="icon" onClick={() => cancel.mutate(job.id)} aria-label="Cancel evaluation"><Square /></Button> : null}{["failed", "cancelled", "interrupted"].includes(job.status) ? <Button variant="ghost" size="icon" onClick={() => retry.mutate(job.id)} aria-label="Retry evaluation"><RefreshCw /></Button> : null}<Badge tone={evaluationTone(job.status)} dot size="sm">{job.status}</Badge></div></div>)}</div>;
}

function LongitudinalEvidence({ items, drift, loading }: { items: EvaluationHistoryItem[]; drift?: EvaluationDrift; loading: boolean }) {
  if (loading) return <SmallLoading label="Loading longitudinal evidence" />;
  const grouped = new Map<string, EvaluationHistoryItem[]>();
  items.forEach((item) => {
    const key = `${item.subject.kind}:${item.subject.value}`;
    grouped.set(key, [...(grouped.get(key) ?? []), item].sort((a, b) => a.history_ordinal - b.history_ordinal));
  });
  return <div className="border-t border-border-subtle"><div className="grid border-b border-border-subtle lg:grid-cols-[240px_minmax(0,1fr)]"><div className="border-b border-border-subtle bg-bg-subtle/25 px-5 py-4 lg:border-b-0 lg:border-r"><div className="text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">Selected pair drift</div>{drift ? <><div className="mt-2 flex items-center gap-2"><Badge tone={driftClassificationTone(drift.classification)} size="sm">{drift.classification.replaceAll("_", " ")}</Badge><span className={cn("font-mono text-[11px]", drift.classification === "improved" ? "text-success" : drift.classification === "regressed" ? "text-danger" : "text-fg-muted")}>{formatSigned(directionNormalizedDelta(drift.delta, drift.direction))}</span></div><p className="mt-2 text-[10px] leading-relaxed text-fg-subtle">Direction-normalized change · {drift.primary_metric} · {drift.direction}. The comparison uses the same immutable suite revision.</p></> : <p className="mt-2 text-[10.5px] leading-relaxed text-fg-subtle">Choose a compatible base and candidate below to classify drift.</p>}</div><div className="min-w-0 px-5 py-4"><div className="mb-3 flex items-center justify-between"><div className="text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">Immutable subject history</div><div className="font-mono text-[9px] text-fg-disabled">{items.length} completed point{items.length === 1 ? "" : "s"}</div></div>{grouped.size ? <div className="divide-y divide-border-subtle border-y border-border-subtle">{[...grouped.entries()].slice(0, 8).map(([key, values]) => <HistorySubjectRow key={key} items={values} />)}</div> : <p className="py-4 text-[10.5px] text-fg-muted">Complete at least two evaluations under this suite revision to expose longitudinal drift.</p>}</div></div></div>;
}

function HistorySubjectRow({ items }: { items: EvaluationHistoryItem[] }) {
  const visible = items.slice(-6);
  const latest = visible.at(-1);
  const direction = latest?.primary_metric?.direction ?? "maximize";
  return <div className="grid gap-3 py-3 sm:grid-cols-[180px_minmax(0,1fr)] sm:items-center"><div className="min-w-0"><div className="truncate text-[10.5px] font-medium text-fg">{latest?.subject.value ?? "Unknown subject"}</div><div className="mt-0.5 text-[9px] text-fg-disabled">{latest?.primary_metric?.name ?? "primary metric"} · {direction} · oldest → newest</div></div><div className="relative flex min-w-[300px] items-start justify-between gap-2 overflow-x-auto before:absolute before:inset-x-4 before:top-2 before:h-px before:bg-border-strong">{visible.map((item, index) => { const previous = index ? visible[index - 1]?.primary_value : null; const normalized = previous == null || item.primary_value == null ? null : directionNormalizedDelta(item.primary_value - previous, direction); return <div key={item.id} className="relative z-10 min-w-12 text-center"><span className={cn("mx-auto block h-4 w-4 rounded-full border-2 border-bg", item.primary_value == null ? "bg-fg-disabled" : normalized == null || normalized === 0 ? "bg-fg-muted" : normalized > 0 ? "bg-success" : "bg-danger")} /><span className="mt-1.5 block font-mono text-[9px] text-fg-muted">{item.primary_value == null ? "unavailable" : formatMetric(item.primary_value)}</span><span className="block text-[8.5px] text-fg-disabled">#{item.history_ordinal + 1}</span></div>; })}</div></div>;
}

function EvaluationTable({ items, loading }: { items: Evaluation[]; loading: boolean }) {
  if (loading) return <SmallLoading label="Loading evaluation history" />;
  if (!items.length) return <Empty label="No persistent evaluation evidence has been recorded." />;
  return <div className="overflow-x-auto border-t border-border-subtle"><table className="w-full min-w-[760px] text-[11px]"><thead><tr className="border-b border-border-subtle text-left text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled"><Th>Evaluation</Th><Th>Suite</Th><Th>Subject</Th><Th>Primary metric</Th><Th>Status</Th><Th>Finished</Th></tr></thead><tbody>{items.map((item) => <tr key={item.id} className="border-b border-border-subtle last:border-0"><Td mono>{item.id}</Td><Td>{item.suite_name || item.suite_revision_id}</Td><Td><div>{item.subject.kind}</div><div className="max-w-[32ch] truncate font-mono text-[9.5px] text-fg-subtle">{item.subject.value}</div></Td><Td mono>{metricLabel(item)}</Td><Td><Badge tone={evaluationTone(item.status)} dot size="sm">{item.status}</Badge></Td><Td>{formatDate(item.finished_at)}</Td></tr>)}</tbody></table></div>;
}

function ComparisonCandidateBar({ evaluations, baseId, candidateIds, onBase, onCandidates }: { evaluations: Evaluation[]; baseId: string; candidateIds: string[]; onBase: (id: string) => void; onCandidates: (ids: string[]) => void }) {
  const base = evaluations.find((item) => item.id === baseId);
  const options = evaluations.map((item) => ({ value: item.id, label: `${subjectLabel(item)} · ${item.suite_name || "Benchmark"}`, description: metricLabel(item), status: item.status, keywords: `${item.id} ${item.subject.value}` }));
  const candidateOptions = (base ? evaluations.filter((item) => item.id !== base.id && item.suite_revision_id === base.suite_revision_id) : evaluations.filter((item) => item.id !== baseId)).map((item) => ({ value: item.id, label: subjectLabel(item), description: `${item.suite_name || "Benchmark"} · ${metricLabel(item)}`, status: item.status, keywords: `${item.id} ${item.subject.value}` }));
  const visibleCandidates = candidateIds.length ? candidateIds : [""];
  return <div className="border-y border-border-subtle bg-bg-subtle/25 px-5 py-4"><div className="grid gap-3 xl:grid-cols-[minmax(220px,1fr)_minmax(0,3fr)]"><FieldLabel label="Base"><SearchPicker value={baseId} onChange={onBase} options={options} placeholder="Choose the reference evaluation" emptyLabel="No completed evaluations" /></FieldLabel><div><div className="mb-1 flex items-center justify-between"><span className="text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">Candidates · {visibleCandidates.length} of 4</span>{visibleCandidates.length < 4 ? <button type="button" onClick={() => onCandidates([...visibleCandidates, ""])} className="text-[10px] text-accent hover:underline"><Plus className="mr-1 inline h-3 w-3" />Add candidate</button> : null}</div><div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">{visibleCandidates.map((id, index) => <div key={`${index}-${id}`} className="relative"><SearchPicker value={id} onChange={(value) => onCandidates(visibleCandidates.map((item, position) => position === index ? value : item).filter((item, position, values) => item || position === values.length - 1).slice(0, 4))} options={candidateOptions.filter((item) => !visibleCandidates.includes(item.value) || item.value === id)} placeholder={`Candidate ${index + 1}`} emptyLabel="No compatible subject" />{visibleCandidates.length > 1 ? <button type="button" aria-label={`Remove candidate ${index + 1}`} onClick={() => onCandidates(visibleCandidates.filter((_, position) => position !== index))} className="absolute -right-1.5 -top-1.5 grid h-4 w-4 place-items-center rounded-full border border-border bg-bg text-fg-disabled hover:text-danger">×</button> : null}</div>)}</div></div></div>{baseId && candidateIds.some(Boolean) ? <div className="mt-3 grid gap-px overflow-hidden rounded-md border border-border bg-border sm:grid-cols-2 xl:grid-cols-4">{candidateIds.filter(Boolean).map((id) => <CandidateDeltaSummary key={id} baseId={baseId} candidateId={id} evaluation={evaluations.find((item) => item.id === id)} />)}</div> : null}</div>;
}

function CandidateDeltaSummary({ baseId, candidateId, evaluation }: { baseId: string; candidateId: string; evaluation?: Evaluation }) {
  const comparison = useQuery({ queryKey: ["evaluations", "compare", baseId, candidateId, "summary"], queryFn: () => api.compareEvaluations(baseId, candidateId, 0, 1), enabled: Boolean(baseId && candidateId), retry: false });
  const normalized = comparison.data ? directionNormalizedDelta(comparison.data.delta, comparison.data.direction) : null;
  return <div className="bg-bg px-3 py-3"><div className="truncate text-[10px] font-medium text-fg">{evaluation ? subjectLabel(evaluation) : "Candidate"}</div><div className={cn("mt-1 font-mono text-[13px]", normalized == null ? "text-fg-disabled" : normalized > 0 ? "text-success" : normalized < 0 ? "text-danger" : "text-fg-muted")}>{comparison.isLoading ? "…" : formatSigned(normalized)}</div><div className="mt-0.5 text-[8.5px] text-fg-disabled">direction-aware delta{comparison.data ? ` · ${comparison.data.counts.regression ?? 0} regressions` : ""}</div></div>;
}

function EvaluationComparisonPanel({ evaluations: _evaluations, baseId, candidateId, onBase: _onBase, onCandidate: _onCandidate, data, loading, error }: { evaluations: Evaluation[]; baseId: string; candidateId: string; onBase: (id: string) => void; onCandidate: (id: string) => void; data?: EvaluationComparison; loading: boolean; error: string | null }) {
  const search = Route.useSearch();
  const navigate = useNavigate();
  const classification = search.classification || "all";
  const visible = data?.samples ?? [];
  const selectedSample = visible.find((sample) => sample.record_id === search.record);
  const page = search.page || 1;
  const total = data?.sample_total ?? visible.length;
  return (
    <div className="border-t border-border-subtle">
      {loading ? <SmallLoading label="Joining per-example evidence" /> : error ? <div className="px-5 py-5 text-[11px] text-danger">{error}</div> : data ? <><ComparisonSummary data={data} /><div className="flex flex-wrap items-center gap-2 border-y border-border-subtle px-5 py-2">{["all", "regression", "improvement", "unchanged_failure", "unchanged_pass"].map((value) => <button key={value} type="button" onClick={() => navigate({ to: "/eval", search: { ...search, classification: value === "all" ? undefined : value, page: 1, record: undefined }, replace: true })} className={cn("rounded-sm px-2 py-1 text-[10.5px]", classification === value ? "bg-accent-bg text-accent" : "text-fg-muted hover:bg-surface")}>{value.replace(/_/g, " ")}<span className="ml-1.5 font-mono text-[9px] opacity-70">{value === "all" ? total : data.counts[value] ?? 0}</span></button>)}</div><EvidenceTable items={visible} selectedRecord={search.record} onSelect={(record) => navigate({ to: "/eval", search: { ...search, record }, replace: true })} />{selectedSample ? <EvaluationSampleInspector sample={selectedSample} onClose={() => navigate({ to: "/eval", search: { ...search, record: undefined }, replace: true })} /> : null}<div className="flex items-center justify-between border-t border-border-subtle px-5 py-2"><span className="font-mono text-[9px] text-fg-disabled">{total ? `${(page - 1) * 100 + 1}–${Math.min(total, page * 100)} of ${total}` : "0 records"}</span><div className="flex gap-1"><Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => navigate({ to: "/eval", search: { ...search, page: page - 1, record: undefined }, replace: true })}>Previous</Button><Button size="sm" variant="ghost" disabled={page * 100 >= total} onClick={() => navigate({ to: "/eval", search: { ...search, page: page + 1, record: undefined }, replace: true })}>Next</Button></div></div><FailureMiningWorkbench baseId={baseId} candidateId={candidateId} comparison={data} /></> : <Empty label="Choose two completed evaluations from the same suite revision." />}
    </div>
  );
}

function ComparisonSummary({ data }: { data: EvaluationComparison }) {
  const improved = isImprovement(data.delta, data.direction);
  return <div className="grid grid-cols-2 gap-px bg-border-subtle sm:grid-cols-4 lg:grid-cols-8"><MetricBlock label="Metric" value={data.primary_metric} /><MetricBlock label="Direction" value={data.direction} /><MetricBlock label="Base" value={formatMetric(data.base_value)} /><MetricBlock label="Candidate" value={formatMetric(data.candidate_value)} /><MetricBlock label="Delta" value={formatSigned(data.delta)} tone={data.delta == null ? undefined : improved ? "success" : "danger"} /><MetricBlock label="Regressions" value={String(data.counts.regression ?? 0)} tone={(data.counts.regression ?? 0) ? "danger" : undefined} /><MetricBlock label="Valid pairs" value={String(data.sample_total ?? data.samples.length)} /><MetricBlock label="Evidence gaps" value={String(data.evidence_gap_total ?? data.evidence_gaps?.length ?? 0)} tone={(data.evidence_gap_total ?? data.evidence_gaps?.length ?? 0) ? "danger" : undefined} /></div>;
}

function EvidenceTable({ items, selectedRecord, onSelect }: { items: EvaluationSampleDelta[]; selectedRecord?: string; onSelect: (recordId: string) => void }) {
  if (!items.length) return <Empty label="No examples match this classification." />;
  return <div className="overflow-x-auto"><table className="w-full min-w-[900px] text-[10.5px]"><thead><tr className="border-b border-border-subtle text-left text-[9px] uppercase tracking-[0.12em] text-fg-disabled"><Th>Class</Th><Th>Record</Th><Th>Input</Th><Th>Expected</Th><Th>Candidate output</Th><Th>Score delta</Th></tr></thead><tbody>{items.map((item) => <tr key={`${item.suite_item_id}-${item.record_id}`} onClick={() => onSelect(item.record_id)} className={cn("cursor-pointer border-b border-border-subtle align-top last:border-0 hover:bg-surface/45", selectedRecord === item.record_id && "bg-accent/7")}><Td><Badge tone={classificationTone(item.classification)} size="sm">{item.classification.replace(/_/g, " ")}</Badge></Td><Td mono>{item.record_id}</Td><Td>{compactValue(item.candidate?.input ?? item.base?.input)}</Td><Td>{compactValue(item.candidate?.expected ?? item.base?.expected)}</Td><Td>{compactValue(item.candidate?.output)}</Td><Td mono>{formatSigned(item.delta)}</Td></tr>)}</tbody></table></div>;
}

function EvaluationSampleInspector({ sample, onClose }: { sample: EvaluationSampleDelta; onClose: () => void }) {
  return <section className="border-t border-accent/25 bg-bg-subtle/30 px-5 py-4"><div className="flex items-start justify-between gap-3"><div><div className="text-[9px] uppercase tracking-[0.12em] text-accent">SELECTED EVIDENCE</div><h3 className="mt-1 font-mono text-[11px] text-fg">{sample.record_id}</h3></div><Button size="sm" variant="ghost" onClick={onClose}>Close</Button></div><div className="mt-3 grid gap-3 lg:grid-cols-4"><EvidenceTrace label="Input" value={sample.candidate?.input ?? sample.base?.input} /><EvidenceTrace label="Expected" value={sample.candidate?.expected ?? sample.base?.expected} /><EvidenceTrace label="Base observation" value={{ output: sample.base?.output, score: sample.base?.score, passed: sample.base?.passed, error: sample.base?.error, verifier_trace: sample.base?.verifier_trace }} /><EvidenceTrace label="Candidate observation" value={{ output: sample.candidate?.output, score: sample.candidate?.score, passed: sample.candidate?.passed, error: sample.candidate?.error, verifier_trace: sample.candidate?.verifier_trace }} /></div></section>;
}

function EvidenceTrace({ label, value }: { label: string; value: unknown }) { return <div><div className="mb-1 text-[8.5px] uppercase tracking-wider text-fg-disabled">{label}</div><pre className="max-h-52 overflow-auto whitespace-pre-wrap rounded border border-border bg-bg p-2 font-mono text-[8.5px] leading-4 text-fg-subtle">{JSON.stringify(value ?? null, null, 2)}</pre></div>; }

function FailureMiningWorkbench({
  baseId,
  candidateId,
  comparison,
}: {
  baseId: string;
  candidateId: string;
  comparison: EvaluationComparison;
}) {
  const [selectorKind, setSelectorKind] =
    useState<FailureMiningSelector["kind"]>("regression");
  const [task, setTask] = useState("");
  const [category, setCategory] = useState("");
  const [failureReason, setFailureReason] = useState("");
  const [minScore, setMinScore] = useState("");
  const [maxScore, setMaxScore] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [parentVersionId, setParentVersionId] = useState("");
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [previewSelectorKey, setPreviewSelectorKey] = useState("");
  const datasets = useQuery({
    queryKey: ["datasets", "failure-parent-picker"],
    queryFn: api.listDatasets,
    retry: false,
  });
  const selectedDataset = useQuery({
    queryKey: ["datasets", datasetId, "failure-parent-picker"],
    queryFn: () => api.datasetDetail(datasetId),
    enabled: Boolean(datasetId),
    retry: false,
  });
  const datasetOptions = (datasets.data?.items ?? []).map((dataset) => ({
    value: dataset.id,
    label: dataset.name,
    description: `${typeof dataset.canonical_schema === "string" ? dataset.canonical_schema : "canonical dataset"} · ${dataset.row_count ?? dataset.latest_version?.row_count ?? "?"} rows`,
    status: dataset.latest_version?.status,
    keywords: `${dataset.description ?? ""} ${dataset.modality ?? ""}`,
  }));
  const versionOptions = (selectedDataset.data?.versions ?? datasets.data?.items.find((dataset) => dataset.id === datasetId)?.versions ?? [])
    .filter((version) => version.status === "ready")
    .map((version) => ({
      value: version.id,
      label: version.label || `Version ${version.version ?? version.id}`,
      description: `${version.row_count ?? "?"} rows · immutable`,
      status: version.status,
      keywords: `${version.content_hash ?? ""} ${version.recipe_hash ?? ""}`,
    }));
  const selector: FailureMiningSelector = {
    kind: selectorKind,
    task: task || undefined,
    category: category || undefined,
    failure_reason: failureReason || undefined,
    min_score:
      minScore.trim() && Number.isFinite(Number(minScore))
        ? Number(minScore)
        : undefined,
    max_score:
      maxScore.trim() && Number.isFinite(Number(maxScore))
        ? Number(maxScore)
        : undefined,
  };
  const selectorKey = JSON.stringify(selector);
  const preview = useMutation({
    mutationFn: () =>
      api.previewFailureMining({
        base_id: baseId || undefined,
        candidate_id: candidateId,
        selector,
        excluded_record_ids: [...excluded],
      }),
    onSuccess: () => setPreviewSelectorKey(selectorKey),
  });
  const build = useMutation({
    mutationFn: () =>
      api.buildFailureMinedDataset({
        dataset_id: datasetId,
        parent_version_id: parentVersionId,
        base_id: baseId || undefined,
        candidate_id: candidateId,
        selector,
        excluded_record_ids: [...excluded],
      }),
  });
  const evidence: FailureMiningPreview | null = preview.data ?? null;
  const previewIsCurrent = Boolean(
    evidence && previewSelectorKey === selectorKey,
  );
  const miningAllowed = comparison.failure_mining_allowed !== false;
  return (
    <div className="border-t border-border-subtle bg-bg-subtle/20">
      <SectionHeading
        eyebrow="FEEDBACK"
        title="Build dataset from failures"
        detail={
          miningAllowed
            ? "Preview and explicitly exclude records before creating an immutable child version."
            : "Holdout evidence is confirmation-only and cannot be recycled into training data."
        }
      />
      <div className="grid gap-3 border-t border-border-subtle px-5 py-4 md:grid-cols-2 xl:grid-cols-4">
        <FieldLabel label="Selector">
          <select
            value={selectorKind}
            disabled={!miningAllowed}
            onChange={(event) => {
              setSelectorKind(event.target.value);
              setExcluded(new Set());
            }}
            className={selectClass}
          >
            <option value="candidate_failure">candidate failure</option>
            <option value="regression">candidate failed / base passed</option>
            <option value="improvement">improvement</option>
            <option value="verifier_disagreement">verifier disagreement</option>
          </select>
        </FieldLabel>
        <FieldLabel label="Task">
          <Input
            disabled={!miningAllowed}
            value={task}
            onChange={(event) => setTask(event.target.value)}
            placeholder="Optional task"
          />
        </FieldLabel>
        <FieldLabel label="Category">
          <Input
            disabled={!miningAllowed}
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            placeholder="Optional category"
          />
        </FieldLabel>
        <FieldLabel label="Failure reason">
          <Input
            disabled={!miningAllowed}
            value={failureReason}
            onChange={(event) => setFailureReason(event.target.value)}
            placeholder="Optional reason"
          />
        </FieldLabel>
        <FieldLabel label="Minimum score / reward">
          <Input
            disabled={!miningAllowed}
            type="number"
            value={minScore}
            onChange={(event) => setMinScore(event.target.value)}
            placeholder="No minimum"
          />
        </FieldLabel>
        <FieldLabel label="Maximum score / reward">
          <Input
            disabled={!miningAllowed}
            type="number"
            value={maxScore}
            onChange={(event) => setMaxScore(event.target.value)}
            placeholder="No maximum"
          />
        </FieldLabel>
        <FieldLabel label="Parent dataset">
          <SearchPicker
            disabled={!miningAllowed}
            value={datasetId}
            onChange={(value) => { setDatasetId(value); setParentVersionId(""); }}
            options={datasetOptions}
            placeholder="Search datasets"
            emptyLabel="No managed dataset is available"
          />
        </FieldLabel>
        <FieldLabel label="Parent version">
          <SearchPicker
            disabled={!miningAllowed || !datasetId}
            value={parentVersionId}
            onChange={setParentVersionId}
            options={versionOptions}
            placeholder="Search ready versions"
            emptyLabel="No ready version belongs to this dataset"
          />
        </FieldLabel>
        <details className="md:col-span-2 xl:col-span-4">
          <summary className="cursor-pointer text-[9.5px] uppercase tracking-wider text-fg-disabled hover:text-fg">Advanced · use unlisted dataset identifiers</summary>
          <div className="mt-2 grid gap-3 md:grid-cols-2">
            <FieldLabel label="Dataset ID"><Input disabled={!miningAllowed} value={datasetId} onChange={(event) => setDatasetId(event.target.value)} /></FieldLabel>
            <FieldLabel label="Version ID"><Input disabled={!miningAllowed} value={parentVersionId} onChange={(event) => setParentVersionId(event.target.value)} /></FieldLabel>
          </div>
        </details>
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-border-subtle px-5 py-3">
        {miningAllowed && candidateId ? (
          <Button variant="secondary" size="sm" asChild>
            <Link
              to="/datasets/review"
              search={{ new: "1", source: "evaluation_comparison", sourceRef: candidateId, baseRef: baseId || undefined }}
            >
              <ClipboardCheck />Review these examples
            </Link>
          </Button>
        ) : null}
        <Button
          variant="secondary"
          size="sm"
          onClick={() => preview.mutate()}
          disabled={!miningAllowed || !candidateId || preview.isPending}
        >
          {preview.isPending ? (
            <Loader2 className="animate-spin" />
          ) : (
            <Wrench />
          )}
          Preview selection
        </Button>
        <Button
          variant="primary"
          size="sm"
          onClick={() => build.mutate()}
          disabled={
            !miningAllowed ||
            !previewIsCurrent ||
            !datasetId ||
            !parentVersionId ||
            build.isPending
          }
        >
          {build.isPending ? (
            <Loader2 className="animate-spin" />
          ) : (
            <Database />
          )}
          Build child version
        </Button>
        {evidence ? (
          <span
            className={cn(
              "font-mono text-[10px]",
              previewIsCurrent ? "text-fg-muted" : "text-warning",
            )}
          >
            {previewIsCurrent
              ? `${evidence.total - excluded.size} accepted · ${excluded.size} excluded`
              : "Filters changed · preview again"}
          </span>
        ) : (
          <span className="text-[10.5px] text-fg-muted">
            {miningAllowed
              ? `Comparison contains ${comparison.samples.length} joined records.`
              : "Mining disabled for this holdout suite."}
          </span>
        )}
        {build.data ? (
          <span className="font-mono text-[10px] text-success">
            job {build.data.job_id || build.data.id}
          </span>
        ) : null}
        {preview.isError || build.isError ? (
          <span className="text-[10px] text-danger">
            {(preview.error as Error)?.message ||
              (build.error as Error)?.message}
          </span>
        ) : null}
      </div>
      {evidence?.items.length ? (
        <div className="max-h-80 overflow-auto border-t border-border-subtle">
          {evidence.items.map((item) => {
            const checked = excluded.has(item.record_id);
            return (
              <label
                key={`${item.suite_item_id}-${item.record_id}`}
                className="grid cursor-pointer grid-cols-[18px_120px_120px_1fr] gap-3 border-b border-border-subtle px-5 py-2.5 text-[10.5px] hover:bg-surface"
              >
                <input
                  type="checkbox"
                  checked={!checked}
                  onChange={() =>
                    setExcluded((current) => {
                      const next = new Set(current);
                      if (next.has(item.record_id)) next.delete(item.record_id);
                      else next.add(item.record_id);
                      return next;
                    })
                  }
                />
                <span className="font-mono text-fg-muted">
                  {item.record_id}
                </span>
                <span className="text-fg-subtle">
                  {item.classification.replace(/_/g, " ")}
                </span>
                <span className="truncate text-fg-muted">
                  {compactValue(item.candidate?.input ?? item.base?.input)}
                </span>
              </label>
            );
          })}
        </div>
      ) : evidence ? (
        <Empty label="The current selector returned no records." />
      ) : null}
    </div>
  );
}

function LegacyCohort({ pinned }: { pinned: string[] }) {
  const data = useQuery<EvalCohortResponse>({
    queryKey: ["eval-cohort", pinned],
    queryFn: () => api.evalCohort(pinned),
    enabled: pinned.length > 0,
    refetchInterval: 30_000,
  });
  if (!pinned.length)
    return (
      <div className="px-5 py-10">
        <Empty label="Pin legacy runs from the runs list to compare lm_eval_summary.json files." />
      </div>
    );
  if (data.isLoading) return <SmallLoading label="Loading legacy summaries" />;
  if (data.isError || !data.data)
    return (
      <div className="px-5 py-8 text-sm text-danger">
        {(data.error as Error)?.message || "Legacy cohort unavailable."}
      </div>
    );
  return (
    <div className="px-5 py-5">
      <LegacyCohortTable data={data.data} />
    </div>
  );
}

function LegacyCohortTable({ data }: { data: EvalCohortResponse }) {
  const availableCount = useMemo(() => data.runs.filter((run) => run.available).length, [data.runs]);
  return <Card><CardHeader><div className="flex items-center gap-2"><CardEyebrow>LEGACY</CardEyebrow><CardTitle>Cohort summaries</CardTitle><BarChart3 className="h-3.5 w-3.5 text-fg-disabled" /></div><span className="text-[11px] text-fg-subtle">{availableCount} of {data.runs.length} have lm_eval_summary.json</span></CardHeader><CardContent className="overflow-x-auto p-0"><table className="w-full min-w-max text-[12px]"><thead><tr className="border-b border-border-subtle"><Th>Run</Th><Th>Status</Th>{data.tasks.map((task) => <Th key={task} right>{task}</Th>)}</tr></thead><tbody>{data.runs.map((run) => <tr key={run.run_id} className="border-b border-border-subtle last:border-0"><Td><Link to="/runs/$runId" params={{ runId: run.run_id }} className="font-mono text-accent hover:underline">{run.run_id}</Link></Td><Td>{run.available ? <Badge tone="success" dot size="sm">ready</Badge> : <span className="text-fg-disabled">no eval</span>}</Td>{data.tasks.map((task) => { const cell = data.cells[run.run_id]?.[task]; const best = data.best_per_task_higher_is_better[task] === run.run_id; return <Td key={task} right mono className={best ? "text-accent" : undefined}>{typeof cell?.value === "number" ? cell.value.toFixed(4) : "—"}</Td>; })}</tr>)}</tbody></table></CardContent></Card>;
}

function SectionHeading({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) { return <div className="flex flex-wrap items-end justify-between gap-3 px-5 py-4"><div><div className="text-[9.5px] uppercase tracking-[0.14em] text-accent">{eyebrow}</div><h2 className="mt-1 text-[13px] font-medium text-fg">{title}</h2></div><p className="max-w-[66ch] text-[10.5px] text-fg-muted">{detail}</p></div>; }
function FieldLabel({ label, children }: { label: string; children: React.ReactNode }) { return <label className="block space-y-1"><span className="block text-[9.5px] uppercase tracking-[0.12em] text-fg-disabled">{label}</span>{children}</label>; }
function Readout({ label, value }: { label: string; value: string }) { return <span className="inline-flex items-center gap-1.5"><span className="tracking-wider text-fg-disabled">{label}</span><span className="text-fg">{value}</span></span>; }
function SmallLoading({ label }: { label: string }) { return <div className="flex items-center justify-center gap-2 px-5 py-10 text-[11px] text-fg-muted"><Loader2 className="h-4 w-4 animate-spin text-accent" />{label}</div>; }
function Empty({ label }: { label: string }) { return <div className="flex flex-col items-center justify-center px-6 py-10 text-center"><FlaskConical className="h-6 w-6 text-fg-disabled" /><p className="mt-2 max-w-md text-[11px] leading-4 text-fg-muted">{label}</p></div>; }
function Progress({ value }: { value?: number | null }) { const percent = Math.max(0, Math.min(100, value ?? 0)); return <div className="flex items-center gap-2"><div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-subtle"><div className="h-full bg-accent transition-all" style={{ width: `${percent}%` }} /></div><span className="w-8 text-right font-mono text-[9.5px] text-fg-subtle">{percent.toFixed(0)}%</span></div>; }
function MetricBlock({ label, value, tone }: { label: string; value: string; tone?: "success" | "danger" }) { return <div className="bg-bg px-4 py-3"><div className={cn("font-mono text-[13px] text-fg", tone === "success" && "text-success", tone === "danger" && "text-danger")}>{value}</div><div className="mt-0.5 text-[9px] uppercase tracking-[0.11em] text-fg-disabled">{label}</div></div>; }
function Th({ children, right }: { children: React.ReactNode; right?: boolean }) { return <th className={cn("px-3.5 py-2 text-left text-[9.5px] font-medium uppercase tracking-[0.12em] text-fg-disabled", right && "text-right")}>{children}</th>; }
function Td({ children, mono, right, className }: { children: React.ReactNode; mono?: boolean; right?: boolean; className?: string }) { return <td className={cn("px-3.5 py-2.5 text-fg-muted", mono && "font-mono tabular-nums", right && "text-right", className)}>{children}</td>; }

const selectClass = "h-9 w-full rounded-md border border-border bg-bg px-2 text-[11px] text-fg";
const textareaClass = "w-full rounded-md border border-border bg-bg px-2 py-2 font-mono text-[10px] leading-4 text-fg outline-none focus:border-accent";
type SubjectPickerOption = { value: string; label: string; description?: string; status?: string; keywords?: string };
function evaluationSubjectOptions(kind: SubjectKind, models: ModelCatalogEntry[], runs: RunListItem[], artifacts: ModelArtifactOccurrence[], evaluations: Evaluation[]): SubjectPickerOption[] {
  const options = new Map<string, SubjectPickerOption>();
  const add = (option: SubjectPickerOption) => { if (option.value && !options.has(option.value)) options.set(option.value, option); };
  evaluations.filter((evaluation) => evaluation.subject.kind === kind).forEach((evaluation) => add({ value: evaluation.subject.value, label: evaluation.subject.kind === "model" ? evaluation.subject.value : evaluation.suite_name ? `${evaluation.suite_name} subject` : friendlySubjectKind(kind), description: `${evaluation.status} evidence · ${evaluation.subject.value}`, status: evaluation.status, keywords: `${evaluation.id} ${evaluation.run_id ?? ""}` }));
  if (kind === "model") models.forEach((model) => add({ value: model.id, label: model.label || model.id, description: `${model.provider || "catalog"} · ${model.memory_tier || "memory unknown"}`, status: model.status, keywords: `${model.id} ${model.trainer_support?.join(" ") ?? ""}` }));
  if (kind === "run" || kind === "final_model") runs.filter((run) => ["completed", "succeeded", "success"].includes(String(run.status).toLowerCase()) && run.final_model_available !== false).forEach((run) => add({ value: run.run_id, label: `${run.model_name || "Completed model"} · ${kind === "run" ? "run" : "final model"}`, description: `${run.modality || "training"} · ${run.run_id}`, status: run.status, keywords: `${run.run_id} ${run.model_name}` }));
  if (kind === "checkpoint") artifacts.filter((artifact) => artifact.kind === "checkpoint" && artifact.run_id).forEach((artifact) => add({ value: `${artifact.run_id}:${artifact.path || artifact.id}`, label: `${artifact.model_name || "Training"} checkpoint`, description: `${artifact.run_id} · ${artifact.path || artifact.id}`, status: artifact.integrity ?? undefined, keywords: `${artifact.id} ${artifact.content_hash}` }));
  return [...options.values()];
}
function friendlySubjectKind(kind: SubjectKind): string { if (kind === "final_model") return "Final model"; if (kind === "checkpoint") return "Checkpoint"; if (kind === "run") return "Completed run"; return "Pinned model"; }
function subject(kind: SubjectKind, value: string): EvaluationSubject {
  const trimmed = value.trim();
  const revisionAt = kind === "model" ? trimmed.lastIndexOf("@") : -1;
  const modelValue = revisionAt > 0 ? trimmed.slice(0, revisionAt) : trimmed;
  return {
    kind,
    value: modelValue,
    revision: revisionAt > 0 ? trimmed.slice(revisionAt + 1) || null : null,
    run_id: kind === "model" ? null : trimmed.split(":")[0],
    checkpoint: kind === "checkpoint" ? trimmed.split(":").slice(1).join(":") || null : null,
  };
}
function metricLabel(item: Evaluation): string { const metric = item.primary_metric ?? item.metrics?.[0]; return metric ? `${metric.name} ${formatMetric(metric.value)} ${metric.direction === "minimize" ? "↓" : "↑"}` : "—"; }
function formatMetric(value?: number | null): string { return typeof value === "number" ? value.toFixed(4) : "—"; }
function formatSigned(value?: number | null): string { return typeof value === "number" ? `${value >= 0 ? "+" : ""}${value.toFixed(4)}` : "—"; }
function isImprovement(delta: number | null | undefined, direction: "maximize" | "minimize"): boolean { return typeof delta === "number" && (direction === "maximize" ? delta > 0 : delta < 0); }
function directionNormalizedDelta(delta: number | null | undefined, direction: "maximize" | "minimize" | string): number | null { return typeof delta === "number" ? delta * (direction === "minimize" ? -1 : 1) : null; }
function compactValue(value: unknown): string { if (value == null) return "—"; const text = typeof value === "string" ? value : JSON.stringify(value); return text.length > 160 ? `${text.slice(0, 157)}…` : text; }
function evaluationTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" { if (status === "completed") return "success"; if (status === "failed") return "danger"; if (status === "cancelled") return "warning"; if (["queued", "running"].includes(status)) return "accent"; return "neutral"; }
function classificationTone(classification: string): "neutral" | "success" | "warning" | "danger" { if (classification === "regression") return "danger"; if (classification === "improvement") return "success"; if (classification === "unchanged_failure") return "warning"; return "neutral"; }
function driftClassificationTone(classification: string): "neutral" | "success" | "warning" | "danger" { if (classification === "regressed") return "danger"; if (classification === "improved") return "success"; if (classification === "unavailable") return "warning"; return "neutral"; }
function formatDate(value?: string | null): string { if (!value) return "—"; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date); }
function subjectLabel(item: Evaluation): string { const kind = friendlySubjectKind(item.subject.kind); const value = item.subject.value || item.run_id || "Unknown subject"; return `${kind} · ${value}`; }
function isEvalSection(value: unknown): value is EvalSection { return typeof value === "string" && ["suites", "launch", "results", "compare", "failure-review", "verifiers", "environments"].includes(value); }
function isVerifierView(value: unknown): value is VerifierStudioView { return typeof value === "string" && ["catalog", "profiles", "calibrate", "compare", "qualification", "training-audits"].includes(value); }
function isRewardAuditView(value: unknown): value is RewardAuditStudioView { return typeof value === "string" && ["profiles", "results", "compare"].includes(value); }
