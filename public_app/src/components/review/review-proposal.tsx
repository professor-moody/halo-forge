import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, AlertTriangle, ArrowLeft, ArrowRight, Check, FileJson, Loader2, Plus, RotateCcw, ShieldCheck, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SearchPicker, type SearchPickerOption } from "@/components/ui/search-picker";
import { StructuredSpecEditor } from "@/components/ui/structured-spec-editor";
import { api, type AcquisitionBatch, type AnnotationModality, type AnnotationTaskType, type ReviewPolicy, type SpecDescriptor, type VerifierProfileRevision } from "@/lib/api";
import { useWorkspaceDraft } from "@/lib/workspace-draft";
import { cn } from "@/lib/utils";

type ProposalDraft = {
  name: string;
  sourceKind: string;
  sourceRef: string;
  baseRef: string;
  split: string;
  strategy: string;
  quota: number;
  seed: number;
  embeddingRevision: string;
  verifierSelector: string;
  selectorMarginMode: "range_fraction" | "margin";
  selectorMargin: number;
  selectorRangeFraction: number;
  repeatTolerance: number;
  subgroupKey: string;
  subgroupValue: string;
  componentRevisionId: string;
  schemaRevisionId: string;
  createSchema: boolean;
  schemaName: string;
  modality: AnnotationModality;
  taskType: AnnotationTaskType;
  definition: Record<string, unknown>;
  mapEnabled: boolean;
  mapSpec: Record<string, unknown>;
  policy: ReviewPolicy;
};

type SchemaPickerOption = SearchPickerOption & { taskType: AnnotationTaskType };

type VerifierSelectorDescriptor = {
  id: string;
  label: string;
  description: string;
  tasks?: string[];
  family?: string;
};

const VERIFIER_SELECTOR_DESCRIPTORS: VerifierSelectorDescriptor[] = [
  { id: "false_accept", label: "False accepts", description: "Human reference rejects, verifier passes.", tasks: ["binary"] },
  { id: "false_reject", label: "False rejects", description: "Human reference accepts, verifier fails.", tasks: ["binary"] },
  { id: "high_confidence_disagreement", label: "High-confidence disagreement", description: "Verifier disagrees far from its declared threshold." },
  { id: "repeat_instability", label: "Repeat instability", description: "Fresh-process repeats flip or drift beyond tolerance." },
  { id: "order_flip", label: "Order flips", description: "A/B and B/A orientations disagree.", tasks: ["pairwise"] },
  { id: "ranking_inversion", label: "Ranking inversions", description: "Predicted ranking reverses reviewed ordering.", tasks: ["ranking"] },
  { id: "threshold_adjacent", label: "Threshold-adjacent", description: "Reward lies near the declared decision threshold." },
  { id: "parser_runtime", label: "Parser or runtime failures", description: "Parser, timeout, or runtime evidence failed." },
  { id: "subgroup", label: "Subgroup failures", description: "Select failures within an explicitly recorded subgroup." },
  { id: "chain_component", label: "Chain component failures", description: "A child error or component disagreement is visible in the chain trace.", family: "chain" },
];

const initialDraft: ProposalDraft = {
  name: "Focused quality review",
  sourceKind: "evaluation",
  sourceRef: "",
  baseRef: "",
  split: "train",
  strategy: "candidate_failure",
  quota: 100,
  seed: 42,
  embeddingRevision: "",
  verifierSelector: "false_accept",
  selectorMarginMode: "range_fraction",
  selectorMargin: 0.05,
  selectorRangeFraction: 0.05,
  repeatTolerance: 0,
  subgroupKey: "",
  subgroupValue: "",
  componentRevisionId: "",
  schemaRevisionId: "",
  createSchema: true,
  schemaName: "Quality decision",
  modality: "text",
  taskType: "binary",
  definition: { output_adapter_id: "filter.v1" },
  mapEnabled: false,
  mapSpec: { schema: "sft", fields: { prompt: "input", response: "output" } },
  policy: { mode: "one_pass", blind_second_pass: false, allow_suggestions: false, require_adjudication: true },
};

export function ReviewProposal({ initialSource, initialSourceRef, initialBaseRef, onClose, onCreated }: { initialSource?: string; initialSourceRef?: string; initialBaseRef?: string; onClose: () => void; onCreated: (queueId: string) => void }) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState<ProposalDraft>({ ...initialDraft, sourceKind: initialSource || initialDraft.sourceKind, sourceRef: initialSourceRef || "", baseRef: initialBaseRef || "", strategy: initialSource === "verifier_calibration" ? "explicit" : initialDraft.strategy });
  const [batch, setBatch] = useState<AcquisitionBatch | null>(null);
  const draftState = useWorkspaceDraft({ surface: "review-proposal", draftKey: "new-queue", name: draft.name, value: draft, onRestore: setDraft });
  const catalog = useProposalCatalog();
  const reviewCapabilities = useQuery({ queryKey: ["review-capabilities"], queryFn: api.reviewCapabilities, retry: false });
  const calibration = useQuery({ queryKey: ["verifier-calibration", "review-proposal", draft.sourceRef], queryFn: () => api.verifierCalibration(draft.sourceRef), enabled: draft.sourceKind === "verifier_calibration" && Boolean(draft.sourceRef), retry: false });
  const verifierRevision = useQuery({ queryKey: ["verifier-profile-revision", "review-proposal", calibration.data?.profile_revision_id], queryFn: () => api.verifierProfileRevision(calibration.data!.profile_revision_id), enabled: draft.sourceKind === "verifier_calibration" && Boolean(calibration.data?.profile_revision_id), retry: false });
  const schemas = useSchemaCatalog();
  const mapDescriptor = useQuery({
    queryKey: ["spec-descriptors", "dataset_recipe_step"],
    queryFn: () => api.listSpecDescriptors("dataset_recipe_step"),
    select: (value) => value.items.find((item) => item.id === "map") ?? null,
  });
  const batchStatus = useQuery({
    queryKey: ["acquisition-batch", batch?.id],
    queryFn: () => api.acquisitionBatch(batch!.id),
    enabled: Boolean(batch?.id),
    refetchInterval: (query) => isAcquisitionActive(query.state.data?.status ?? batch?.status) ? 1_500 : false,
    retry: false,
  });
  const candidates = useQuery({
    queryKey: ["acquisition-candidates", batch?.id],
    queryFn: () => api.acquisitionCandidates(batch!.id, { limit: 6 }),
    enabled: Boolean(batch?.id && batch.status === "ready"),
    retry: false,
  });

  useEffect(() => {
    if (batchStatus.data) setBatch(batchStatus.data);
  }, [batchStatus.data]);

  useEffect(() => {
    if (draft.sourceKind !== "verifier_calibration" || !verifierRevision.data) return;
    const available = compatibleVerifierSelectors(verifierRevision.data, reviewCapabilities.data?.verifier_failure_selectors);
    if (available.some((item) => item.id === draft.verifierSelector)) return;
    setDraft((current) => ({ ...current, verifierSelector: available[0]?.id ?? "" }));
  }, [draft.sourceKind, draft.verifierSelector, reviewCapabilities.data?.verifier_failure_selectors, verifierRevision.data]);

  const prepare = useMutation({
    mutationFn: () => api.createAcquisitionBatch({
      name: draft.name,
      seed: draft.seed,
      sources: [sourcePayload(draft)],
      strategies: [{ kind: draft.sourceKind === "verifier_calibration" ? "explicit" : draft.strategy, quota: draft.quota > 0 ? draft.quota : undefined, options: strategyOptions(draft) }],
      metadata: { proposal: "review-studio", projection: draft.mapEnabled ? draft.mapSpec : undefined },
    }),
    onSuccess: (value) => { queryClient.setQueryData(["acquisition-batch", value.id], value); setBatch(value); setStep(3); },
  });
  const cancelBatch = useMutation({
    mutationFn: () => api.cancelAcquisitionBatch(batch!.id),
    onSuccess: (value) => { queryClient.setQueryData(["acquisition-batch", value.id], value); setBatch(value); },
  });
  const retryBatch = useMutation({
    mutationFn: () => api.retryAcquisitionBatch(batch!.id),
    onSuccess: (value) => { queryClient.setQueryData(["acquisition-batch", value.id], value); setBatch(value); },
  });
  const create = useMutation({
    mutationFn: async () => {
      let schemaRevisionId = draft.schemaRevisionId;
      if (draft.createSchema) {
        await api.validateAnnotationSchema({ name: draft.schemaName, modality: draft.modality, task_type: draft.taskType, definition: draft.definition });
        const created = await api.createAnnotationSchema({ name: draft.schemaName, modality: draft.modality, task_type: draft.taskType, definition: draft.definition });
        schemaRevisionId = created.revision.id;
      }
      if (!batch) throw new Error("Prepare the candidate batch first.");
      return api.createReviewQueue({ batch_id: batch.id, schema_revision_id: schemaRevisionId, name: draft.name, policy: draft.policy });
    },
    onSuccess: async (queue) => { await draftState.clear().catch(() => undefined); onCreated(queue.id); },
  });

  const sourceOptions = (catalog.data as Record<string, SearchPickerOption[]> | undefined)?.[draft.sourceKind] ?? [];
  const verifierSourceReady = draft.sourceKind !== "verifier_calibration" || Boolean(verifierRevision.data && selectorDraftValid(draft));
  const canContinue = step === 0 ? Boolean(draft.name.trim() && draft.sourceRef.trim() && (draft.sourceKind !== "evaluation_comparison" || draft.baseRef.trim()) && (draft.strategy !== "diversity" || draft.embeddingRevision.trim()) && verifierSourceReady) : step === 1 ? Boolean(draft.createSchema ? draft.schemaName.trim() : draft.schemaRevisionId) : true;

  return (
    <section aria-label="New review queue proposal" className="flex min-h-0 flex-1 flex-col bg-bg">
      <header className="border-b border-border px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-accent">New review queue</div>
            <h2 className="mt-1 text-lg font-semibold tracking-tight text-fg">Turn evidence into a focused human decision</h2>
            <p className="mt-1 max-w-2xl text-[11px] leading-5 text-fg-subtle">Choose eligible records, bind an immutable annotation schema, then confirm the review policy before anyone labels an item.</p>
          </div>
          <Button size="sm" variant="ghost" onClick={onClose}>Close</Button>
        </div>
        <ol className="mt-4 flex max-w-2xl items-center" aria-label="Proposal progress">
          {["Source", "Annotation", "Review policy", "Confirm"].map((label, index) => (
            <li key={label} className="flex flex-1 items-center last:flex-none">
              <button type="button" disabled={index > step || Boolean(batch)} onClick={() => setStep(index)} className={cn("flex items-center gap-2 text-[10px]", index === step ? "text-fg" : index < step ? "text-accent" : "text-fg-disabled")}>
                <span className={cn("grid h-5 w-5 place-items-center rounded-full border font-mono text-[9px]", index < step ? "border-accent bg-accent text-accent-fg" : index === step ? "border-accent" : "border-border")}>{index < step ? <Check className="h-3 w-3" /> : index + 1}</span>{label}
              </button>
              {index < 3 ? <span className="mx-2 h-px flex-1 bg-border" /> : null}
            </li>
          ))}
        </ol>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
        {draftState.candidate ? <RestoreDraftBanner name={draftState.candidate.name} onRestore={draftState.restore} onDiscard={draftState.discard} /> : null}
        {step === 0 ? <SourceStep draft={draft} setDraft={setDraft} sourceOptions={sourceOptions} loading={catalog.isLoading} verifierRevision={verifierRevision.data} verifierLoading={calibration.isLoading || verifierRevision.isLoading} verifierError={calibration.error?.message || verifierRevision.error?.message} availableSourceKinds={reviewCapabilities.data?.acquisition_source_kinds} availableVerifierSelectors={reviewCapabilities.data?.verifier_failure_selectors} /> : null}
        {step === 1 ? <AnnotationStep draft={draft} setDraft={setDraft} schemas={schemas.data ?? []} mapDescriptor={mapDescriptor.data ?? null} /> : null}
        {step === 2 ? <PolicyStep draft={draft} setDraft={setDraft} /> : null}
        {step === 3 ? <ConfirmStep draft={draft} batch={batch} candidates={candidates.data?.items ?? []} loading={candidates.isLoading} refreshing={batchStatus.isFetching} onCancel={() => cancelBatch.mutate()} onRetry={() => retryBatch.mutate()} actionPending={cancelBatch.isPending || retryBatch.isPending} /> : null}
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-bg-subtle/50 px-5 py-3">
        <div className="text-[9.5px] text-fg-disabled">{draftState.isSaving ? "Saving proposal…" : draftState.savedAt ? "Proposal autosaved" : draftState.unavailable ? "Autosave unavailable" : "Changes save automatically"}</div>
        <div className="flex items-center gap-2">
          {step > 0 && !batch ? <Button size="sm" variant="ghost" onClick={() => setStep((value) => value - 1)}><ArrowLeft />Back</Button> : null}
          {step < 2 ? <Button size="sm" variant="primary" disabled={!canContinue} onClick={() => setStep((value) => value + 1)}>Continue<ArrowRight /></Button> : null}
          {step === 2 ? <Button size="sm" variant="primary" disabled={!canContinue || prepare.isPending} onClick={() => prepare.mutate()}>{prepare.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}Prepare candidates</Button> : null}
          {step === 3 ? <Button size="sm" variant="primary" disabled={!batch || batch.status !== "ready" || !batch.row_count || create.isPending} onClick={() => create.mutate()}>{create.isPending ? <Loader2 className="animate-spin" /> : <Plus />}Create review queue</Button> : null}
        </div>
        {prepare.isError || batchStatus.isError || cancelBatch.isError || retryBatch.isError || create.isError ? <p role="alert" className="w-full text-right text-[10px] text-danger">{prepare.error?.message || batchStatus.error?.message || cancelBatch.error?.message || retryBatch.error?.message || create.error?.message}</p> : null}
      </footer>
    </section>
  );
}

function SourceStep({ draft, setDraft, sourceOptions, loading, verifierRevision, verifierLoading, verifierError, availableSourceKinds, availableVerifierSelectors }: { draft: ProposalDraft; setDraft: (value: ProposalDraft) => void; sourceOptions: SearchPickerOption[]; loading: boolean; verifierRevision?: VerifierProfileRevision; verifierLoading: boolean; verifierError?: string; availableSourceKinds?: string[]; availableVerifierSelectors?: string[] }) {
  const verifierSource = draft.sourceKind === "verifier_calibration";
  const verifierSourceAvailable = !availableSourceKinds?.length || availableSourceKinds.includes("verifier_calibration");
  const rewardAuditSource = draft.sourceKind === "reward_integrity_audit";
  const rewardAuditSourceAvailable = !availableSourceKinds?.length || availableSourceKinds.includes("reward_integrity_audit");
  const selectors = compatibleVerifierSelectors(verifierRevision, availableVerifierSelectors);
  return <div className="mx-auto max-w-3xl space-y-5">
    <SectionIntro number="01" title="Select the evidence" copy="Review only records that can improve a development dataset. Operational and holdout evidence stays isolated." />
    <div className="grid gap-4 rounded-lg border border-border bg-surface/40 p-4 sm:grid-cols-2">
      <Field label="Queue name"><Input value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} placeholder="e.g. July regression review" /></Field>
      <Field label="Source type"><select value={draft.sourceKind} onChange={(event) => { const sourceKind = event.target.value; setDraft({ ...draft, sourceKind, sourceRef: "", baseRef: "", strategy: sourceKind === "verifier_calibration" ? "explicit" : sourceKind.startsWith("evaluation") || sourceKind === "reward_integrity_audit" ? "candidate_failure" : "explicit" }); }} className={selectClass}>
        <option value="evaluation">Completed evaluation</option>
        <option value="evaluation_comparison">Evaluation comparison</option>
        {verifierSourceAvailable ? <option value="verifier_calibration">Verifier calibration failures</option> : null}
        {rewardAuditSourceAvailable ? <option value="reward_integrity_audit">Training audit evidence</option> : null}
        <option value="dataset_version">Dataset version</option>
        <option value="run_samples">Run samples</option>
        <option value="playground_session">Playground session</option>
        <option value="jsonl">Local JSONL</option>
      </select></Field>
      <div className="sm:col-span-2"><Field label={draft.sourceKind === "evaluation_comparison" ? "Candidate evaluation" : verifierSource ? "Development calibration" : rewardAuditSource ? "Completed training audit" : "Source"}>
        {draft.sourceKind === "jsonl" ? <Input value={draft.sourceRef} onChange={(event) => setDraft({ ...draft, sourceRef: event.target.value })} placeholder="/path/to/review-records.jsonl" /> : <SearchPicker value={draft.sourceRef} onChange={(sourceRef) => setDraft({ ...draft, sourceRef })} options={sourceOptions} disabled={loading} placeholder={loading ? "Loading eligible sources…" : verifierSource ? "Search completed development calibrations" : rewardAuditSource ? "Search completed same-output audits" : "Search eligible sources"} />}
      </Field></div>
      {draft.sourceKind === "evaluation_comparison" ? <div className="sm:col-span-2"><Field label="Base evaluation"><SearchPicker value={draft.baseRef} onChange={(baseRef) => setDraft({ ...draft, baseRef })} options={sourceOptions.filter((option) => option.value !== draft.sourceRef)} placeholder="Choose the matching base evaluation" /></Field></div> : null}
      {draft.sourceKind === "dataset_version" ? <Field label="Split"><select value={draft.split} onChange={(event) => setDraft({ ...draft, split: event.target.value })} className={selectClass}><option value="train">Train</option><option value="validation">Validation</option></select></Field> : null}
      {verifierSource ? <div className="sm:col-span-2"><div className="border-l-2 border-accent/40 pl-3 text-[9.5px] leading-4 text-fg-subtle"><span className="font-medium text-fg">Development-only evidence.</span> Only the calibration partition is eligible. Confirmation, operational, holdout, test, canary, leaked, and protected-lineage records are refused.</div>{verifierLoading ? <p className="mt-2 text-[9.5px] text-fg-disabled">Resolving exact verifier task and family…</p> : verifierError ? <p role="alert" className="mt-2 text-[9.5px] text-danger">{verifierError}</p> : verifierRevision ? <p className="mt-2 font-mono text-[9px] text-fg-disabled">{humanize(verifierRevision.family)} · {humanize(verifierRevision.task_type)} · exact revision {shortId(verifierRevision.id)}</p> : null}</div> : null}
      {rewardAuditSource ? <div className="sm:col-span-2 border-l-2 border-accent/40 pl-3 text-[9.5px] leading-4 text-fg-subtle"><span className="font-medium text-fg">Reviewed proposal only.</span> Halo Forge copies the exact captured output and paired verifier evidence into an immutable acquisition batch. Protected purposes, test/canary splits, and protected lineage are refused. Creating the queue does not resolve a training pause.</div> : null}
    </div>
    {verifierSource ? <VerifierSelectorFields draft={draft} setDraft={setDraft} selectors={selectors} revision={verifierRevision} /> : <div className="grid gap-3 sm:grid-cols-[1fr_140px_120px]">
      <Field label="Acquisition strategy"><select value={draft.strategy} onChange={(event) => setDraft({ ...draft, strategy: event.target.value })} className={selectClass}>
        <option value="candidate_failure">Candidate failures</option><option value="regression">Base passed, candidate failed</option><option value="verifier_disagreement">Verifier disagreement</option><option value="low_score">Lowest scores</option><option value="low_margin">Lowest confidence margin</option><option value="coverage_gap">Coverage gaps</option><option value="diversity">Diverse sample</option><option value="random">Seeded random</option><option value="explicit">All eligible records</option>
      </select></Field>
      <Field label="Maximum items"><Input type="number" min={1} value={draft.quota} onChange={(event) => setDraft({ ...draft, quota: Number(event.target.value) })} /></Field>
      <Field label="Seed"><Input type="number" value={draft.seed} onChange={(event) => setDraft({ ...draft, seed: Number(event.target.value) })} /></Field>
      {draft.strategy === "diversity" ? <div className="sm:col-span-3"><Field label="Pinned embedding revision"><Input value={draft.embeddingRevision} onChange={(event) => setDraft({ ...draft, embeddingRevision: event.target.value })} placeholder="image:organization/model@commit" /><p className="mt-1 text-[9.5px] leading-4 text-fg-disabled">Halo Forge uses stored vectors when present or schedules the pinned compatible model. It never substitutes synthetic diversity evidence.</p></Field></div> : null}
    </div>}
    <p className="text-[10px] leading-4 text-fg-disabled"><ShieldCheck className="mr-1 inline h-3 w-3" />Eligibility is checked before the candidate batch is published. Held-out, canary, operational, and final-holdout records are refused.</p>
  </div>;
}

function VerifierSelectorFields({ draft, setDraft, selectors, revision }: { draft: ProposalDraft; setDraft: (value: ProposalDraft) => void; selectors: VerifierSelectorDescriptor[]; revision?: VerifierProfileRevision }) {
  const selected = selectors.find((item) => item.id === draft.verifierSelector);
  const componentOptions = (revision?.components ?? []).map((component) => ({ value: component.child_revision_id, label: `Component ${component.ordinal + 1}`, description: component.child?.profile_id || shortId(component.child_revision_id), keywords: component.child_revision_id }));
  return <div className="rounded-lg border border-border bg-surface/40 p-4"><div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_140px_120px]"><Field label="Reliability failure selector"><select value={draft.verifierSelector} onChange={(event) => { const verifierSelector = event.target.value; setDraft({ ...draft, verifierSelector, selectorRangeFraction: verifierSelector === "high_confidence_disagreement" ? 0.25 : verifierSelector === "threshold_adjacent" ? 0.05 : draft.selectorRangeFraction }); }} className={selectClass}>{selectors.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></Field><Field label="Maximum items"><Input type="number" min={1} value={draft.quota} onChange={(event) => setDraft({ ...draft, quota: Number(event.target.value) })} /></Field><Field label="Seed"><Input type="number" value={draft.seed} onChange={(event) => setDraft({ ...draft, seed: Number(event.target.value) })} /></Field></div>{selected ? <p className="mt-2 text-[9.5px] leading-4 text-fg-subtle">{selected.description}</p> : null}
    {["high_confidence_disagreement", "threshold_adjacent"].includes(draft.verifierSelector) ? <div className="mt-4 grid gap-3 sm:grid-cols-2"><Field label="Margin definition"><select value={draft.selectorMarginMode} onChange={(event) => setDraft({ ...draft, selectorMarginMode: event.target.value as ProposalDraft["selectorMarginMode"] })} className={selectClass}><option value="range_fraction">Fraction of declared reward range</option><option value="margin">Absolute reward margin</option></select></Field>{draft.selectorMarginMode === "range_fraction" ? <Field label="Range fraction"><Input type="number" min={0} max={1} step={0.01} value={draft.selectorRangeFraction} onChange={(event) => setDraft({ ...draft, selectorRangeFraction: Number(event.target.value) })} /></Field> : <Field label="Absolute margin"><Input type="number" min={0} step={0.001} value={draft.selectorMargin} onChange={(event) => setDraft({ ...draft, selectorMargin: Number(event.target.value) })} /></Field>}<p className="sm:col-span-2 text-[9px] leading-4 text-fg-disabled">The exact reward contract, threshold, and resolved margin remain attached to every selected record. No confidence is invented.</p></div> : null}
    {draft.verifierSelector === "repeat_instability" ? <div className="mt-4"><Field label="Maximum tolerated reward drift"><Input type="number" min={0} step={0.000001} value={draft.repeatTolerance} onChange={(event) => setDraft({ ...draft, repeatTolerance: Number(event.target.value) })} /><p className="mt-1 text-[9px] leading-4 text-fg-disabled">Pass, parsed-value, and error flips are always selected; this value controls numeric reward drift.</p></Field></div> : null}
    {draft.verifierSelector === "subgroup" ? <div className="mt-4 grid gap-3 sm:grid-cols-2"><Field label="Subgroup field"><Input value={draft.subgroupKey} onChange={(event) => setDraft({ ...draft, subgroupKey: event.target.value })} placeholder="category" /></Field><Field label="Subgroup value · optional"><Input value={draft.subgroupValue} onChange={(event) => setDraft({ ...draft, subgroupValue: event.target.value })} placeholder="reasoning" /></Field></div> : null}
    {draft.verifierSelector === "chain_component" ? <div className="mt-4"><Field label="Chain component · optional"><SearchPicker allowEmpty value={draft.componentRevisionId} onChange={(componentRevisionId) => setDraft({ ...draft, componentRevisionId })} options={componentOptions} placeholder="Any component with an error or disagreement" emptyLabel="No child component is available on this revision" /><p className="mt-1 text-[9px] leading-4 text-fg-disabled">Leave empty to include failures from any child while preserving the full ordered component trace.</p></Field></div> : null}
  </div>;
}

function AnnotationStep({ draft, setDraft, schemas, mapDescriptor }: { draft: ProposalDraft; setDraft: (value: ProposalDraft) => void; schemas: SchemaPickerOption[]; mapDescriptor: SpecDescriptor | null }) {
  const descriptor = useMemo(() => annotationDescriptor(draft.taskType), [draft.taskType]);
  return <div className="mx-auto max-w-3xl space-y-5">
    <SectionIntro number="02" title="Define the decision" copy="A pinned schema keeps every label interpretable after the queue is complete." />
    <div className="flex gap-1 rounded-md bg-bg-subtle p-1">
      <button type="button" onClick={() => setDraft({ ...draft, createSchema: false })} className={segmentedClass(!draft.createSchema)}>Existing schema</button>
      <button type="button" onClick={() => setDraft({ ...draft, createSchema: true })} className={segmentedClass(draft.createSchema)}>New schema</button>
    </div>
    {draft.createSchema ? <div className="space-y-4 rounded-lg border border-border bg-surface/40 p-4">
      <div className="grid gap-3 sm:grid-cols-3"><Field label="Schema name"><Input value={draft.schemaName} onChange={(event) => setDraft({ ...draft, schemaName: event.target.value })} /></Field><Field label="Modality"><select value={draft.modality} onChange={(event) => { const modality = event.target.value; const available = taskOptions(modality); const taskType = available.includes(draft.taskType) ? draft.taskType : "binary"; const twoPass = defaultTwoPassTask(taskType); setDraft({ ...draft, modality, taskType, definition: defaultDefinition(taskType, modality), policy: { ...draft.policy, mode: twoPass ? "two_pass" : "one_pass", blind_second_pass: twoPass } }); }} className={selectClass}><option value="text">Text</option><option value="preference">Preference</option><option value="tool">Tool use</option><option value="vlm">Vision + language</option><option value="audio">Audio</option></select></Field><Field label="Task"><select value={draft.taskType} onChange={(event) => { const taskType = event.target.value; const twoPass = defaultTwoPassTask(taskType); setDraft({ ...draft, taskType, definition: defaultDefinition(taskType, draft.modality), policy: { ...draft.policy, mode: twoPass ? "two_pass" : "one_pass", blind_second_pass: twoPass } }); }} className={selectClass}>{taskOptions(draft.modality).map((task) => <option key={task} value={task}>{humanize(task)}</option>)}</select></Field></div>
      <StructuredSpecEditor descriptor={descriptor} value={draft.definition} onChange={(definition) => setDraft({ ...draft, definition })} validateRemotely={false} />
    </div> : <Field label="Schema revision"><SearchPicker value={draft.schemaRevisionId} onChange={(schemaRevisionId) => { const selected = schemas.find((value) => value.value === schemaRevisionId); const taskType = selected?.taskType ?? draft.taskType; const twoPass = defaultTwoPassTask(taskType); setDraft({ ...draft, schemaRevisionId, taskType, policy: { ...draft.policy, mode: twoPass ? "two_pass" : "one_pass", blind_second_pass: twoPass } }); }} options={schemas} placeholder="Search immutable schema revisions" /></Field>}
    {mapDescriptor ? <details className="rounded-lg border border-border-subtle bg-bg-subtle/45 p-4"><summary className="cursor-pointer text-[11px] font-medium text-fg">Candidate field projection <span className="ml-1 font-normal text-fg-disabled">optional</span></summary><label className="mt-3 flex items-center gap-2 text-[10.5px] text-fg-muted"><input type="checkbox" checked={draft.mapEnabled} onChange={(event) => setDraft({ ...draft, mapEnabled: event.target.checked })} />Project source-specific fields into a canonical preview</label>{draft.mapEnabled ? <StructuredSpecEditor className="mt-4" descriptor={mapDescriptor} value={draft.mapSpec} onChange={(mapSpec) => setDraft({ ...draft, mapSpec })} /> : null}</details> : null}
  </div>;
}

function PolicyStep({ draft, setDraft }: { draft: ProposalDraft; setDraft: (value: ProposalDraft) => void }) {
  const policy = draft.policy;
  const update = (next: Partial<ReviewPolicy>) => setDraft({ ...draft, policy: { ...policy, ...next } });
  return <div className="mx-auto max-w-3xl space-y-5"><SectionIntro number="03" title="Set the review policy" copy="Keep simple decisions fast; reserve independent second passes and adjudication for judgment-heavy work." />
    <div className="grid gap-3 sm:grid-cols-2"><PolicyCard selected={policy.mode === "one_pass"} title="One pass" copy="One reviewer decision per item. Best for filtering and clear categorical labels." onClick={() => update({ mode: "one_pass", blind_second_pass: false })} /><PolicyCard selected={policy.mode === "two_pass"} title="Two independent passes" copy="Compare two decisions and send disagreements to adjudication." onClick={() => update({ mode: "two_pass", blind_second_pass: true })} /></div>
    <div className="divide-y divide-border-subtle rounded-lg border border-border bg-surface/40">
      <ToggleRow label="Blind second pass" copy="Hide first-pass decisions until the second decision is submitted." checked={Boolean(policy.blind_second_pass)} disabled={policy.mode !== "two_pass"} onChange={(blind_second_pass) => update({ blind_second_pass })} />
      <ToggleRow label="Require adjudication" copy="Conflicting pass decisions need a final reviewed resolution." checked={Boolean(policy.require_adjudication)} disabled={policy.mode !== "two_pass"} onChange={(require_adjudication) => update({ require_adjudication })} />
      <ToggleRow label="Allow model suggestions" copy="Reviewers may request a suggestion; provenance is retained and suggestions never submit themselves." checked={Boolean(policy.allow_suggestions)} onChange={(allow_suggestions) => update({ allow_suggestions })} />
    </div>
  </div>;
}

function ConfirmStep({ draft, batch, candidates, loading, refreshing, onCancel, onRetry, actionPending }: { draft: ProposalDraft; batch: AcquisitionBatch | null; candidates: Array<{ id: string; record_id: string; record?: Record<string, unknown>; score?: number | null; stratum?: string | null }>; loading: boolean; refreshing: boolean; onCancel: () => void; onRetry: () => void; actionPending: boolean }) {
  const active = isAcquisitionActive(batch?.status);
  const retryable = isAcquisitionRetryable(batch?.status);
  const progress = acquisitionProgress(batch);
  return <div className="mx-auto max-w-3xl space-y-5"><SectionIntro number="04" title="Confirm the review set" copy="The acquisition batch is immutable. Inspect the sample before opening the queue." />
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-4"><Stat label="Status" value={batch?.status ?? "preparing"} /><Stat label="Eligible items" value={String(batch?.row_count ?? 0)} /><Stat label="Passes" value={draft.policy.mode === "two_pass" ? "2" : "1"} /><Stat label="Seed" value={String(draft.seed)} /></div>
    {active ? <div role="status" aria-live="polite" className="rounded-lg border border-accent/30 bg-accent/7 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex min-w-0 items-start gap-3"><span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-accent/12 text-accent"><Activity className="h-4 w-4" /></span><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><h3 className="text-[11.5px] font-medium text-fg">Preparing review candidates</h3>{refreshing ? <Loader2 className="h-3 w-3 animate-spin text-accent" /> : null}</div><p className="mt-1 text-[10px] leading-4 text-fg-subtle">{acquisitionStageCopy(batch)} You can leave this screen; the Activity Center will keep tracking queue position, resources, progress, and logs.</p></div></div>
        <Button size="sm" variant="ghost" disabled={actionPending} onClick={onCancel}>Cancel</Button>
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-bg-subtle"><div className={cn("h-full rounded-full bg-accent transition-[width] duration-500", progress == null && "w-1/3 animate-pulse")} style={progress == null ? undefined : { width: `${progress}%` }} /></div>
      <div className="mt-2 flex flex-wrap justify-between gap-2 font-mono text-[9px] text-fg-disabled"><span>{batch?.processed_records != null && batch?.total_records ? `${batch.processed_records.toLocaleString()} / ${batch.total_records.toLocaleString()} records` : humanize(batch?.stage || batch?.status || "queued")}</span><span>{progress == null ? "Waiting for progress" : `${Math.round(progress)}%`}{batch?.work_item_id ? ` · activity ${shortId(batch.work_item_id)}` : ""}</span></div>
    </div> : null}
    {retryable ? <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-danger/30 bg-danger/5 p-4"><div className="flex min-w-0 items-start gap-3"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-danger" /><div><h3 className="text-[11.5px] font-medium text-fg">Candidate preparation needs attention</h3><p className="mt-1 max-w-xl text-[10px] leading-4 text-fg-subtle">{batch?.error || `The acquisition attempt is ${humanize(batch?.status || "interrupted").toLowerCase()}. Review its Activity Center history, then retry without changing this proposal.`}</p></div></div><Button size="sm" variant="secondary" disabled={actionPending} onClick={onRetry}>{actionPending ? <Loader2 className="animate-spin" /> : <RotateCcw />}Retry preparation</Button></div> : null}
    {batch?.eligibility && Object.keys(batch.eligibility).length ? <div className="rounded-md border border-border-subtle bg-bg-subtle p-3 text-[10px] text-fg-muted"><span className="font-medium text-fg">Eligibility report</span><pre className="mt-2 overflow-auto font-mono text-[9.5px] leading-4">{JSON.stringify(batch.eligibility, null, 2)}</pre></div> : null}
    <div><div className="mb-2 flex items-center justify-between"><h3 className="text-[11px] font-medium text-fg">Candidate preview</h3><span className="font-mono text-[9px] text-fg-disabled">{batch?.status === "ready" ? `first ${candidates.length}` : "available when ready"}</span></div>{batch?.status !== "ready" ? <div className="rounded-lg border border-dashed border-border py-8 text-center text-[10px] text-fg-disabled">The immutable sample will appear after acquisition and eligibility checks finish.</div> : loading ? <div className="py-8 text-center text-[10px] text-fg-disabled">Loading immutable candidates…</div> : candidates.length ? <div className="divide-y divide-border-subtle overflow-hidden rounded-lg border border-border">{candidates.map((candidate) => <div key={candidate.id} className="grid gap-2 bg-surface/30 px-3 py-2.5 sm:grid-cols-[110px_1fr_auto]"><span className="truncate font-mono text-[9px] text-fg-disabled">{candidate.record_id}</span><span className="truncate text-[10.5px] text-fg-muted">{recordSummary(candidate.record)}</span><span className="text-[9px] text-fg-disabled">{candidate.stratum || (candidate.score == null ? "selected" : candidate.score.toFixed(3))}</span></div>)}</div> : <div className="rounded-lg border border-dashed border-border py-8 text-center text-[10px] text-fg-disabled">No eligible candidates were selected. Adjust the source or strategy and prepare a new batch.</div>}</div>
  </div>;
}

function useProposalCatalog() {
  return useQuery({ queryKey: ["review-proposal-catalog"], queryFn: async () => {
    const [evaluations, datasets, runs, sessions, calibrations, rewardAudits] = await Promise.all([api.evaluationHistory({ limit: 150 }), api.listDatasets(), api.listRuns({ limit: 150 }), api.listPlaygroundSessions({ limit: 150 }), api.listVerifierCalibrations({ status: "completed", limit: 150 }).catch(() => ({ items: [], total: 0, limit: 150, offset: 0 })), api.listRewardIntegrityAudits({ status: "completed", limit: 150 }).catch(() => ({ items: [], total: 0, limit: 150, offset: 0 }))]);
    const versionPages = await Promise.all((datasets.items ?? []).map(async (dataset) => ({ dataset, versions: (await api.datasetVersions(dataset.id)).items })));
    const evaluationOptions = evaluations.items.filter((item) => item.status === "completed").map((item) => ({ value: item.id, label: item.suite_name || item.id, description: `${item.subject?.kind || "subject"} · ${item.primary_value ?? "no primary metric"}`, status: item.created_at ? formatDate(item.created_at) : undefined, keywords: item.id }));
    const calibrationOptions = calibrations.items.filter((item) => item.status === "completed" && ["development", "unspecified", ""].includes(item.source_purpose ?? "unspecified")).map((item) => ({ value: item.id, label: item.source_name || `Verifier calibration ${shortId(item.id)}`, description: `${humanize(item.qualification?.decision || "unqualified")} · ${humanize(item.source_purpose || "unspecified")} · ${(item.total_records ?? item.processed_records ?? 0).toLocaleString()} stable records`, status: item.completed_at ? formatDate(item.completed_at) : undefined, keywords: `${item.id} ${item.profile_revision_id} ${item.source_hash || ""}` }));
    const rewardAuditOptions = rewardAudits.items.map((item) => ({ value: item.id, label: `${item.run_id} · ${item.boundary_unit === "final" ? "final boundary" : `${humanize(item.boundary_unit || "boundary")} ${item.boundary_value ?? ""}`}`, description: `${humanize(item.decision?.decision || item.status)} · ${humanize(item.capture_fidelity || "capture unavailable")}`, status: item.completed_at ? formatDate(item.completed_at) : undefined, keywords: `${item.id} ${item.run_id} ${item.reward_system_revision_id}` }));
    return {
      evaluation: evaluationOptions,
      evaluation_comparison: evaluationOptions,
      verifier_calibration: calibrationOptions,
      reward_integrity_audit: rewardAuditOptions,
      dataset_version: versionPages.flatMap(({ dataset, versions }) => versions.filter((version) => version.status === "ready").map((version) => ({ value: version.id, label: `${dataset.name} · ${version.label || `v${version.version || ""}`}`, description: `${version.row_count ?? 0} rows · ${dataset.modality || "text"}`, keywords: `${dataset.id} ${version.content_hash || ""}` }))),
      run_samples: (runs.items ?? []).map((run) => ({ value: run.run_id, label: run.model_name || run.run_id, description: `${run.modality || "run"} · ${run.status || "unknown"}`, keywords: run.run_id })),
      playground_session: (sessions.items ?? []).filter((session) => !session.archived).map((session) => ({ value: session.id, label: session.name, description: `${session.messages.length} messages`, keywords: session.id })),
      jsonl: [],
    } satisfies Record<string, SearchPickerOption[]>;
  }, staleTime: 20_000 });
}

function useSchemaCatalog() {
  return useQuery({ queryKey: ["annotation-schema-revisions"], queryFn: async () => {
    const schemas = await api.listAnnotationSchemas({ limit: 200 });
    const revisions = await Promise.all(schemas.items.filter((schema) => !schema.archived).map(async (schema) => ({ schema, revisions: (await api.listAnnotationSchemaRevisions(schema.id, { limit: 200 })).items })));
    return revisions.flatMap(({ schema, revisions: values }) => values.map((revision) => ({ value: revision.id, label: `${schema.name} · r${revision.revision_number}`, description: `${humanize(revision.modality)} · ${humanize(revision.task_type)}`, keywords: `${schema.id} ${revision.id}`, taskType: revision.task_type })));
  }, staleTime: 20_000 });
}

function annotationDescriptor(taskType: string): SpecDescriptor {
  const common = [{ name: "output_adapter_id", label: "Dataset output", value_type: "select", options: ["filter.v1", "metadata.v1", "sft_correction.v1", "preference.v1", "tool_trace.v1", "vlm_annotation.v1", "audio_annotation.v1"], description: "Controls how reviewed labels render during dataset handoff." }];
  const fields = taskType === "categorical" || taskType === "multi_label" ? [{ name: "labels", label: "Allowed labels", value_type: "array", required: true, description: "A JSON list of stable label names." }, ...common] : taskType === "scalar" ? [{ name: "minimum", label: "Minimum", value_type: "number", default: 0 }, { name: "maximum", label: "Maximum", value_type: "number", default: 1 }, ...common] : common;
  return { kind: "annotation_schema", id: taskType, version: "1", label: `${humanize(taskType)} definition`, description: "The form and Advanced JSON produce the same immutable schema definition.", fields };
}

function sourcePayload(draft: ProposalDraft) {
  if (draft.sourceKind === "evaluation_comparison") return { kind: draft.sourceKind, candidate_id: draft.sourceRef, base_id: draft.baseRef };
  if (draft.sourceKind === "dataset_version") return { kind: draft.sourceKind, ref: draft.sourceRef, split: draft.split };
  if (draft.sourceKind === "verifier_calibration") return { kind: draft.sourceKind, ref: draft.sourceRef, selector: draft.verifierSelector, options: verifierSelectorOptions(draft) };
  return { kind: draft.sourceKind, ref: draft.sourceRef };
}

function strategyOptions(draft: ProposalDraft) {
  if (draft.strategy === "low_score") return { direction: "maximize" };
  if (draft.strategy === "diversity") return { embedding_revision: draft.embeddingRevision };
  return {};
}

function compatibleVerifierSelectors(revision?: VerifierProfileRevision, available?: string[]) {
  const advertised = new Set(available?.length ? available : VERIFIER_SELECTOR_DESCRIPTORS.map((item) => item.id));
  return VERIFIER_SELECTOR_DESCRIPTORS.filter((item) => advertised.has(item.id) && (!item.tasks || item.tasks.includes(revision?.task_type ?? "")) && (!item.family || item.family === revision?.family));
}

function selectorDraftValid(draft: ProposalDraft) {
  if (!draft.verifierSelector) return false;
  if (draft.verifierSelector === "subgroup") return Boolean(draft.subgroupKey.trim());
  if (["high_confidence_disagreement", "threshold_adjacent"].includes(draft.verifierSelector)) {
    const value = draft.selectorMarginMode === "range_fraction" ? draft.selectorRangeFraction : draft.selectorMargin;
    return Number.isFinite(value) && value >= 0 && (draft.selectorMarginMode !== "range_fraction" || value <= 1);
  }
  if (draft.verifierSelector === "repeat_instability") return Number.isFinite(draft.repeatTolerance) && draft.repeatTolerance >= 0;
  return true;
}

function verifierSelectorOptions(draft: ProposalDraft) {
  if (["high_confidence_disagreement", "threshold_adjacent"].includes(draft.verifierSelector)) return draft.selectorMarginMode === "range_fraction" ? { range_fraction: draft.selectorRangeFraction } : { margin: draft.selectorMargin };
  if (draft.verifierSelector === "repeat_instability") return { tolerance: draft.repeatTolerance };
  if (draft.verifierSelector === "subgroup") return { key: draft.subgroupKey.trim(), value: draft.subgroupValue.trim() || undefined };
  if (draft.verifierSelector === "chain_component") return draft.componentRevisionId ? { component_revision_id: draft.componentRevisionId } : {};
  return {};
}

function defaultDefinition(task: string, modality: string) { if (task === "binary") return { output_adapter_id: "filter.v1" }; if (task === "categorical" || task === "multi_label") return { labels: ["good", "needs_work"], output_adapter_id: modality === "vlm" ? "vlm_annotation.v1" : modality === "audio" ? "audio_annotation.v1" : "metadata.v1" }; if (task === "scalar") return { minimum: 0, maximum: 1, output_adapter_id: modality === "vlm" ? "vlm_annotation.v1" : modality === "audio" ? "audio_annotation.v1" : "metadata.v1" }; if (task === "structured_correction") return { output_adapter_id: "tool_trace.v1" }; if (task === "text_correction") return { output_adapter_id: modality === "vlm" ? "vlm_annotation.v1" : modality === "audio" ? "audio_annotation.v1" : "sft_correction.v1" }; if (task === "pairwise" || task === "ranking") return { output_adapter_id: "preference.v1" }; return { output_adapter_id: "metadata.v1" }; }
function defaultTwoPassTask(task: string) { return ["pairwise", "ranking", "text_correction", "structured_correction"].includes(task); }
function taskOptions(modality: string) { const shared = ["binary", "categorical", "multi_label", "scalar"]; if (modality === "preference") return ["binary", "categorical", "scalar", "pairwise", "ranking"]; if (modality === "tool") return ["binary", "categorical", "scalar", "structured_correction"]; if (modality === "audio") return [...shared, "text_correction"]; return [...shared, "text_correction", "pairwise", "ranking"]; }
function SectionIntro({ number, title, copy }: { number: string; title: string; copy: string }) { return <div className="grid gap-2 sm:grid-cols-[64px_1fr]"><span className="font-mono text-[10px] text-accent">{number}</span><div><h3 className="text-sm font-semibold text-fg">{title}</h3><p className="mt-1 max-w-xl text-[10.5px] leading-5 text-fg-subtle">{copy}</p></div></div>; }
function Field({ label, children }: { label: string; children: React.ReactNode }) { return <div className="space-y-1.5"><Label>{label}</Label>{children}</div>; }
function PolicyCard({ selected, title, copy, onClick }: { selected: boolean; title: string; copy: string; onClick: () => void }) { return <button type="button" aria-pressed={selected} onClick={onClick} className={cn("rounded-lg border p-4 text-left transition-colors", selected ? "border-accent bg-accent/7" : "border-border bg-surface/40 hover:border-border-strong")}><span className="flex items-center gap-2 text-[11.5px] font-medium text-fg">{selected ? <Check className="h-3.5 w-3.5 text-accent" /> : <span className="h-3.5 w-3.5 rounded-full border border-border-strong" />}{title}</span><span className="mt-1.5 block text-[10px] leading-4 text-fg-subtle">{copy}</span></button>; }
function ToggleRow({ label, copy, checked, disabled, onChange }: { label: string; copy: string; checked: boolean; disabled?: boolean; onChange: (checked: boolean) => void }) { return <label className={cn("flex items-center justify-between gap-4 px-4 py-3", disabled && "opacity-45")}><span><span className="block text-[11px] font-medium text-fg">{label}</span><span className="mt-0.5 block text-[9.5px] leading-4 text-fg-subtle">{copy}</span></span><input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} className="h-4 w-4 accent-[var(--color-accent)]" /></label>; }
function Stat({ label, value }: { label: string; value: string }) { return <div className="bg-surface/60 px-3 py-3"><div className="text-[9px] uppercase tracking-wider text-fg-disabled">{label}</div><div className="mt-1 truncate font-mono text-[11px] text-fg">{value}</div></div>; }
function RestoreDraftBanner({ name, onRestore, onDiscard }: { name: string; onRestore: () => void; onDiscard: () => void }) { return <div className="mx-auto mb-5 flex max-w-3xl items-center justify-between gap-3 rounded-md border border-accent/35 bg-accent/7 px-3 py-2"><div className="flex items-center gap-2 text-[10.5px] text-fg"><FileJson className="h-3.5 w-3.5 text-accent" />A saved “{name}” proposal is available.</div><div className="flex gap-1"><Button size="sm" variant="ghost" onClick={onDiscard}>Discard</Button><Button size="sm" onClick={onRestore}>Restore</Button></div></div>; }
function recordSummary(record?: Record<string, unknown>) { if (!record) return "Record details unavailable"; const value = record.prompt ?? record.input ?? record.question ?? record.transcript ?? record.messages ?? record; return typeof value === "string" ? value : JSON.stringify(value); }
function isAcquisitionActive(status?: string | null) { return ["queued", "pending", "preparing", "selecting", "running", "blocked"].includes(status || ""); }
function isAcquisitionRetryable(status?: string | null) { return ["failed", "interrupted", "cancelled", "needs_reconciliation"].includes(status || ""); }
function acquisitionProgress(batch: AcquisitionBatch | null) { if (!batch) return null; if (batch.progress_percent != null && Number.isFinite(batch.progress_percent)) return Math.max(0, Math.min(100, batch.progress_percent)); if (batch.total_records && batch.processed_records != null) return Math.max(0, Math.min(100, (100 * batch.processed_records) / batch.total_records)); return null; }
function acquisitionStageCopy(batch: AcquisitionBatch | null) { const stage = humanize(batch?.stage || batch?.status || "queued").toLowerCase(); return stage === "queued" ? "The request is queued for the supervised worker." : stage === "blocked" ? "The request is waiting for a dependency or workstation resource." : `The worker is ${stage}.`; }
function shortId(value: string) { return value.length > 14 ? `${value.slice(0, 7)}…${value.slice(-5)}` : value; }
function humanize(value: string) { return value.replace(/[_-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(); }
function segmentedClass(active: boolean) { return cn("flex-1 rounded px-3 py-2 text-[10.5px] font-medium transition-colors", active ? "bg-surface text-fg shadow-sm" : "text-fg-subtle hover:text-fg"); }
const selectClass = "h-8 w-full rounded-md border border-border bg-surface px-2.5 text-[11.5px] text-fg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20";
