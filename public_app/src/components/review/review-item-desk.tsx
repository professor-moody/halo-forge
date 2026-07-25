import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Bot, Check, CheckCircle2, CircleSlash2, Database, Eye, Flag, Image as ImageIcon, Keyboard, Loader2, Music2, Plus, RotateCcw, Save, ShieldCheck, Sparkles, Trash2, X, ZoomIn } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ApiError, api, type AnnotationSchemaRevision, type DatasetBuildPreview, type DatasetRecord, type LabelSetPublicationAccepted, type ReviewEvent, type ReviewItem, type ReviewQueue } from "@/lib/api";
import { useWorkspaceDraft } from "@/lib/workspace-draft";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SearchPicker } from "@/components/ui/search-picker";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type ReviewDraft = { annotation: Record<string, unknown>; reason: string; note: string };
type ToolParameterDraft = { name: string; type: string; description: string; required: boolean; schema: Record<string, unknown> };
type ToolDetails = { name: string; description: string; parameters: ToolParameterDraft[] };

export function ReviewItemDesk({ queue, item, schema, position, total, onPrevious, onNext }: { queue: ReviewQueue; item: ReviewItem; schema: AnnotationSchemaRevision; position: number; total: number; onPrevious: () => void; onNext: () => void }) {
  const queryClient = useQueryClient();
  const baseDraft = useMemo<ReviewDraft>(() => ({ annotation: initialAnnotation(schema, item, queue.current_pass, shouldFlipCandidates(queue, item)), reason: "", note: "" }), [schema, item, queue]);
  const [draft, setDraft] = useState(baseDraft);
  const [conflict, setConflict] = useState<string | null>(null);
  const draftState = useWorkspaceDraft({ surface: "review", draftKey: `${queue.id}:${item.id}:pass-${queue.current_pass}`, name: `Item ${position + 1}`, value: draft, onRestore: setDraft });
  const events = useQuery({ queryKey: ["review-item-events", item.id], queryFn: () => api.reviewItemEvents(item.id, { limit: 100 }) });
  const suggestions = useQuery({ queryKey: ["review-item-suggestions", item.id], queryFn: () => api.reviewItemSuggestions(item.id), enabled: Boolean(queue.policy.allow_suggestions) });
  const currentDecision = item.projection?.[`pass_${queue.current_pass}`] as { event_id?: string } | undefined;
  const isAdjudication = item.status === "conflict" || item.status === "needs_adjudication";
  const isCorrection = Boolean(currentDecision?.event_id) && !isAdjudication;
  const flipCandidates = shouldFlipCandidates(queue, item);
  const submit = useMutation({
    mutationFn: async ({ eventType, payload, reason }: { eventType: string; payload: Record<string, unknown>; reason?: string }) => {
      const effectiveType = eventType === "label" ? isAdjudication ? "adjudicate" : isCorrection ? "correct" : "label" : eventType;
      const decisionPayload = ["label", "correct", "adjudicate"].includes(effectiveType)
        ? { ...payload, ...(draft.note.trim() ? { note: draft.note.trim() } : {}), ...(effectiveType === "correct" ? { reason: reason || draft.reason } : {}) }
        : payload;
      return api.submitReviewEvent(item.id, {
        event_type: effectiveType,
        pass_number: queue.current_pass,
        payload: decisionPayload,
        reason: isAdjudication ? reason || draft.reason : undefined,
        idempotency_key: eventKey(queue.id, item.id, effectiveType),
        expected_active_event_id: item.active_event_id ?? null,
        supersedes_event_id: effectiveType === "correct" ? currentDecision?.event_id : undefined,
      });
    },
    onSuccess: async () => {
      setConflict(null);
      await draftState.clear().catch(() => undefined);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["review-queue", queue.id] }),
        queryClient.invalidateQueries({ queryKey: ["review-queue-items", queue.id] }),
        queryClient.invalidateQueries({ queryKey: ["review-queue-statistics", queue.id] }),
        queryClient.invalidateQueries({ queryKey: ["review-item-events", item.id] }),
        queryClient.invalidateQueries({ queryKey: ["review-item", item.id] }),
      ]);
      onNext();
    },
    onError: (error) => { if (error instanceof ApiError && error.status === 409) setConflict("This item changed after you opened it. Your draft is safe; reload the latest decision before submitting."); },
  });

  const submitLabel = (annotation = draft.annotation) => submit.mutate({ eventType: "label", payload: annotation, reason: draft.reason });
  const defer = () => submit.mutate({ eventType: "defer", payload: {} });
  useReviewKeyboard({ schema, item, flipCandidates, onPrevious, onNext, onSkip: defer, onLabel: (annotation) => { setDraft((value) => ({ ...value, annotation })); submitLabel(annotation); }, disabled: submit.isPending || queue.status !== "active" });

  return (
    <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_280px]">
      <main className="flex min-h-0 flex-col bg-bg" aria-label={`Review item ${position + 1} of ${total}`}>
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border-subtle px-4 py-2.5">
          <div className="flex items-center gap-2"><span className="font-mono text-[10px] text-fg-disabled">{position + 1} / {total}</span><Badge tone={statusVariant(item.status)}>{humanize(item.status)}</Badge>{isAdjudication ? <Badge tone="warning">Adjudication</Badge> : <Badge tone="neutral">Pass {queue.current_pass}</Badge>}</div>
          <div className="flex items-center gap-1"><Button size="icon" variant="ghost" onClick={onPrevious} disabled={position === 0} aria-label="Previous review item"><ArrowLeft /></Button><Button size="icon" variant="ghost" onClick={onNext} disabled={position >= total - 1} aria-label="Next review item"><ArrowRight /></Button></div>
        </div>

        {queue.policy.mode === "two_pass" && queue.current_pass === 2 && queue.policy.blind_second_pass ? <div className="flex items-center gap-2 border-b border-accent/25 bg-accent/7 px-4 py-2 text-[10px] text-fg-muted"><Eye className="h-3.5 w-3.5 text-accent" />Blind second pass · the first decision remains hidden until you submit.</div> : null}
        {conflict ? <div role="alert" className="flex items-start justify-between gap-3 border-b border-warning/30 bg-warning/8 px-4 py-3 text-[10.5px] text-fg"><span className="flex gap-2"><AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />{conflict}</span><Button size="sm" variant="ghost" onClick={() => window.location.reload()}><RotateCcw />Reload</Button></div> : null}
        {draftState.candidate ? <div className="flex items-center justify-between gap-3 border-b border-accent/25 bg-accent/7 px-4 py-2 text-[10px] text-fg"><span>A saved draft for this item is available.</span><span className="flex gap-1"><Button size="sm" variant="ghost" onClick={draftState.discard}>Discard</Button><Button size="sm" onClick={draftState.restore}>Restore</Button></span></div> : null}

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 pb-36 lg:pb-5">
          <div className="mx-auto max-w-4xl">
            <RecordEvidence item={item} schema={schema} flipCandidates={flipCandidates} />
            <section className="mt-5 border-t border-border pt-5" aria-labelledby="review-decision-title">
              <div className="mb-3 flex items-center justify-between gap-3"><div><div className="text-[9px] font-medium uppercase tracking-[0.14em] text-accent">Your decision</div><h2 id="review-decision-title" className="mt-1 text-sm font-semibold text-fg">{decisionTitle(schema.task_type, isAdjudication, isCorrection)}</h2></div><span className="hidden items-center gap-1 font-mono text-[9px] text-fg-disabled sm:flex"><Keyboard className="h-3.5 w-3.5" />Shortcuts active</span></div>
              <AnnotationControl schema={schema} item={item} annotation={draft.annotation} flipCandidates={flipCandidates} onChange={(annotation) => setDraft({ ...draft, annotation })} />
              {(isAdjudication || isCorrection) ? <div className="mt-3"><Label htmlFor="review-reason">{isAdjudication ? "Adjudication rationale" : "Correction reason"} <span className="text-accent">*</span></Label><textarea id="review-reason" value={draft.reason} onChange={(event) => setDraft({ ...draft, reason: event.target.value })} rows={2} placeholder="Explain why the final decision differs…" className={textareaClass} /></div> : <details className="mt-3"><summary className="cursor-pointer text-[10px] text-fg-subtle">Add a reviewer note</summary><textarea aria-label="Reviewer note" value={draft.note} onChange={(event) => setDraft({ ...draft, note: event.target.value })} rows={2} className={textareaClass} /></details>}
            </section>
            <div className="mt-5 space-y-2 lg:hidden">
              <details className="rounded-md border border-border bg-bg-subtle/45 p-3"><summary className="cursor-pointer text-[10.5px] font-medium text-fg">Rubric and source context</summary><div className="mt-3"><GuidePanel queue={queue} item={item} schema={schema} suggestions={suggestions.data?.items ?? []} /></div></details>
              <details className="rounded-md border border-border bg-bg-subtle/45 p-3"><summary className="cursor-pointer text-[10.5px] font-medium text-fg">Decision history</summary><div className="mt-3"><HistoryPanel events={events.data?.items ?? []} /></div></details>
              <details className="rounded-md border border-border bg-bg-subtle/45 p-3"><summary className="cursor-pointer text-[10.5px] font-medium text-fg">Publish and build dataset</summary><div className="mt-3"><LabelSetHandoff queue={queue} schema={schema} /></div></details>
            </div>
          </div>
        </div>

        <ActionDock draft={draft} schema={schema} pending={submit.isPending} disabled={queue.status !== "active" || (isAdjudication || isCorrection) && !draft.reason.trim()} skipDisabled={submit.isPending || queue.status !== "active"} saving={draftState.isSaving} onSubmit={() => submitLabel()} onSkip={defer} onExclude={(reason) => submit.mutate({ eventType: "exclude", payload: { reason } })} onFlag={(reason) => submit.mutate({ eventType: "flag", payload: { reason } })} />
      </main>

      <ReviewInspector queue={queue} item={item} schema={schema} events={events.data?.items ?? []} suggestions={suggestions.data?.items ?? []} />
    </div>
  );
}

function RecordEvidence({ item, schema, flipCandidates }: { item: ReviewItem; schema: AnnotationSchemaRevision; flipCandidates: boolean }) {
  const record = item.record ?? {};
  const image = firstString(record, ["image_url", "image", "image_reference"]);
  const audio = firstString(record, ["audio_url", "audio", "audio_reference"]);
  const messages = Array.isArray(record.messages) ? record.messages as Array<Record<string, unknown>> : null;
  const alternatives = (Array.isArray(record.alternatives) ? record.alternatives : Array.isArray(record.candidates) ? record.candidates : Array.isArray(record.responses) ? record.responses : null) as unknown[] | null;
  const chosen = record.chosen;
  const rejected = record.rejected;
  const toolDefinitions = record.tools ?? record.tool_definitions;
  const calls = record.expected_calls ?? record.tool_calls;
  return <article>
    <div className="mb-4 flex items-start justify-between gap-3"><div><div className="text-[9px] font-medium uppercase tracking-[0.14em] text-fg-disabled">Evidence</div><h1 className="mt-1 text-base font-semibold text-fg">{recordHeading(record, schema.modality)}</h1></div><span className="font-mono text-[9px] text-fg-disabled">{item.record_id || item.id}</span></div>
    {schema.modality === "vlm" || image ? <MediaImage src={image} record={record} /> : null}
    {schema.modality === "audio" || audio ? <MediaAudio src={audio} record={record} /> : null}
    {messages ? <div className="space-y-2">{messages.map((message, index) => <div key={index} className={cn("rounded-md border px-3 py-2.5", message.role === "assistant" ? "ml-5 border-accent/20 bg-accent/5" : "mr-5 border-border bg-surface/45")}><div className="mb-1 text-[8.5px] font-medium uppercase tracking-wider text-fg-disabled">{String(message.role || "message")}</div><div className="whitespace-pre-wrap text-[12px] leading-5 text-fg">{renderValue(message.content)}</div></div>)}</div> : null}
    {!messages ? <div className="space-y-3"><EvidenceBlock label="Prompt / input" value={record.prompt ?? record.input ?? record.question ?? record.instruction} prominent /><EvidenceBlock label="Response / output" value={record.response ?? record.output ?? record.answer ?? record.transcript} /></div> : null}
    {chosen !== undefined || rejected !== undefined || alternatives ? <div className="mt-4 grid gap-3 sm:grid-cols-2">{(schema.task_type === "pairwise" ? pairwiseCandidates(item, flipCandidates) : alternatives ?? [chosen, rejected]).map((value, index) => <EvidenceBlock key={index} label={schema.task_type === "pairwise" ? ["Option A", "Option B"][index] : `Option ${index + 1}`} value={value} />)}</div> : null}
    {toolDefinitions !== undefined ? <div className="mt-4 grid gap-3 sm:grid-cols-2"><EvidenceBlock label="Tool definitions" value={toolDefinitions} code /><EvidenceBlock label="Expected calls / results" value={calls} code /></div> : null}
  </article>;
}

function MediaImage({ src, record }: { src: string | null; record: Record<string, unknown> }) {
  const [zoomed, setZoomed] = useState(false);
  const alt = String(record.image_alt ?? record.alt ?? "Review image");
  useEffect(() => {
    if (!zoomed) return;
    const close = (event: KeyboardEvent) => { if (event.key === "Escape") setZoomed(false); };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [zoomed]);
  return <><figure className="mb-4 overflow-hidden rounded-lg border border-border bg-black/25">{src ? <button type="button" onClick={() => setZoomed(true)} className="group relative block w-full cursor-zoom-in" aria-label="Open image in a larger view"><img src={src} alt={alt} className="max-h-[440px] w-full object-contain" /><span className="absolute bottom-2 right-2 flex items-center gap-1.5 rounded-md border border-white/15 bg-black/70 px-2 py-1 text-[9.5px] text-white opacity-90 backdrop-blur transition-opacity group-hover:opacity-100 group-focus-visible:ring-2 group-focus-visible:ring-accent"><ZoomIn className="h-3 w-3" />Enlarge</span></button> : <div className="grid h-56 place-items-center text-fg-disabled"><ImageIcon className="h-7 w-7" /><span className="sr-only">Image reference unavailable</span></div>}<figcaption className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border bg-bg-subtle px-3 py-2 font-mono text-[9px] text-fg-disabled"><span>{String(record.width ?? "?")} × {String(record.height ?? "?")}</span>{record.mime_type ? <span>{String(record.mime_type)}</span> : null}{src ? <span className="truncate">{src}</span> : null}</figcaption></figure>{zoomed && src ? <div className="fixed inset-0 z-[90] grid place-items-center bg-black/85 p-3 backdrop-blur-sm sm:p-6" role="presentation" onMouseDown={() => setZoomed(false)}><section role="dialog" aria-modal="true" aria-label="Enlarged review image" className="relative flex max-h-full max-w-full flex-col overflow-hidden rounded-lg border border-white/15 bg-black shadow-2xl" onMouseDown={(event) => event.stopPropagation()}><Button autoFocus size="icon" variant="secondary" onClick={() => setZoomed(false)} className="absolute right-2 top-2 z-10 bg-black/70 text-white hover:bg-black" aria-label="Close enlarged image"><X /></Button><img src={src} alt={alt} className="max-h-[calc(100vh-4rem)] max-w-[calc(100vw-2rem)] object-contain sm:max-w-[calc(100vw-5rem)]" /><div className="border-t border-white/10 bg-black px-3 py-2 text-center font-mono text-[9px] text-white/60">{String(record.width ?? "?")} × {String(record.height ?? "?")} · press Escape to close</div></section></div> : null}</>;
}

function MediaAudio({ src, record }: { src: string | null; record: Record<string, unknown> }) {
  return <figure className="mb-4 rounded-lg border border-border bg-surface/40 p-4"><div className="flex items-center gap-2 text-[10px] font-medium text-fg"><Music2 className="h-4 w-4 text-accent" />Audio evidence</div>{src ? <audio className="mt-3 w-full" controls preload="metadata" src={src}>Your browser cannot play this audio evidence.</audio> : <div className="mt-3 rounded-md border border-dashed border-border px-3 py-6 text-center text-[10px] text-fg-disabled">The audio reference is not currently available.</div>}<figcaption className="mt-2 flex flex-wrap gap-3 font-mono text-[9px] text-fg-disabled"><span>{String(record.duration_seconds ?? record.duration ?? "?")} sec</span><span>{String(record.sample_rate ?? "?")} Hz</span>{record.channels ? <span>{String(record.channels)} channels</span> : null}</figcaption></figure>;
}

function EvidenceBlock({ label, value, prominent, code }: { label: string; value: unknown; prominent?: boolean; code?: boolean }) { if (value === undefined || value === null || value === "") return null; return <section className={cn("rounded-md border border-border bg-surface/35 px-3 py-3", prominent && "border-border-strong")}><div className="mb-1.5 text-[8.5px] font-medium uppercase tracking-wider text-fg-disabled">{label}</div><div className={cn("whitespace-pre-wrap text-[11.5px] leading-5 text-fg", code && "overflow-auto font-mono text-[9.5px] leading-4")}>{renderValue(value)}</div></section>; }

function AnnotationControl({ schema, item, annotation, flipCandidates, onChange }: { schema: AnnotationSchemaRevision; item: ReviewItem; annotation: Record<string, unknown>; flipCandidates: boolean; onChange: (value: Record<string, unknown>) => void }) {
  const task = schema.task_type;
  const labels = Array.isArray(schema.definition.labels) ? schema.definition.labels.map(String) : [];
  if (task === "binary") return <div className="grid gap-2 sm:grid-cols-2"><DecisionButton active={annotation.accepted === true} tone="accept" shortcut="A" label="Accept" onClick={() => onChange({ accepted: true })} /><DecisionButton active={annotation.accepted === false} tone="reject" shortcut="R" label="Reject" onClick={() => onChange({ accepted: false })} /></div>;
  if (task === "categorical") return <div className="flex flex-wrap gap-2">{labels.map((label, index) => <DecisionButton key={label} active={annotation.label === label} shortcut={index < 9 ? String(index + 1) : undefined} label={humanize(label)} onClick={() => onChange({ label })} />)}</div>;
  if (task === "multi_label") { const selected = Array.isArray(annotation.labels) ? annotation.labels.map(String) : []; return <div className="flex flex-wrap gap-2">{labels.map((label) => <label key={label} className={cn("flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-[11px]", selected.includes(label) ? "border-accent bg-accent/7 text-fg" : "border-border bg-surface text-fg-muted")}><input type="checkbox" checked={selected.includes(label)} onChange={(event) => onChange({ labels: event.target.checked ? [...selected, label] : selected.filter((value) => value !== label) })} />{humanize(label)}</label>)}</div>; }
  if (task === "scalar") { const min = Number(schema.definition.minimum ?? 0); const max = Number(schema.definition.maximum ?? 1); const score = Number(annotation.score ?? min); return <div className="grid gap-3 sm:grid-cols-[1fr_90px]"><input aria-label="Review score" type="range" min={min} max={max} step={(max - min) / 100} value={score} onChange={(event) => onChange({ score: Number(event.target.value) })} /><Input aria-label="Exact review score" type="number" min={min} max={max} value={score} onChange={(event) => onChange({ score: Number(event.target.value) })} /></div>; }
  if (task === "pairwise") { const [optionA, optionB] = pairwiseCandidates(item, flipCandidates); return <div className="grid gap-2 sm:grid-cols-3"><DecisionButton active={choiceMatches(annotation.chosen, optionA)} shortcut="1" label="Choose A" onClick={() => onChange({ chosen: optionA, rejected: optionB })} /><DecisionButton active={choiceMatches(annotation.chosen, optionB)} shortcut="2" label="Choose B" onClick={() => onChange({ chosen: optionB, rejected: optionA })} /><DecisionButton active={annotation.chosen === "tie"} shortcut="T" label="Tie / equal" onClick={() => onChange({ chosen: "tie" })} /></div>; }
  if (task === "ranking") return <RankingEditor value={annotation.ranking} candidates={rankingCandidates(item, flipCandidates)} onChange={(ranking) => onChange({ ranking })} />;
  if (task === "structured_correction") return <StructuredCorrectionEditor value={annotation.correction} onChange={(correction) => onChange({ correction })} />;
  return <textarea aria-label="Corrected response" value={String(annotation.corrected_text ?? "")} onChange={(event) => onChange({ corrected_text: event.target.value })} rows={7} className={textareaClass} placeholder="Write the corrected target response…" />;
}

function RankingEditor({ value, candidates, onChange }: { value: unknown; candidates: unknown[]; onChange: (ranking: unknown[]) => void }) {
  const ranking = Array.isArray(value) && value.length ? value : candidates;
  const move = (from: number, to: number) => {
    if (to < 0 || to >= ranking.length) return;
    const next = [...ranking];
    const [candidate] = next.splice(from, 1);
    next.splice(to, 0, candidate);
    onChange(next);
  };
  if (ranking.length < 2) return <div role="alert" className="rounded-md border border-warning/30 bg-warning/8 px-3 py-3 text-[10.5px] leading-5 text-fg-muted">This item needs at least two candidates before it can be ranked.</div>;
  return <div>
    <div className="mb-2 flex items-baseline justify-between gap-3"><p className="text-[10.5px] text-fg-subtle">Arrange from best to worst. The first candidate becomes the winner.</p><span className="shrink-0 font-mono text-[9px] text-fg-disabled">{ranking.length} candidates</span></div>
    <ol aria-label="Candidate ranking" className="divide-y divide-border overflow-hidden rounded-md border border-border bg-surface/35">
      {ranking.map((candidate, index) => <li key={`${stableValueKey(candidate)}-${index}`} className="grid grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-2 px-2 py-2.5 sm:grid-cols-[2.25rem_minmax(0,1fr)_auto]">
        <span aria-label={`Rank ${index + 1}`} className={cn("grid h-7 w-7 place-items-center rounded-full font-mono text-[10px] font-semibold", index === 0 ? "bg-accent text-white" : "bg-bg-subtle text-fg-muted")}>{index + 1}</span>
        <div className="min-w-0 whitespace-pre-wrap text-[11px] leading-5 text-fg">{renderValue(candidate)}</div>
        <div className="flex gap-1">
          <Button type="button" size="icon" variant="ghost" className="min-h-11 min-w-11" disabled={index === 0} onClick={() => move(index, index - 1)} aria-label={`Move candidate ${index + 1} up`}><ArrowUp /></Button>
          <Button type="button" size="icon" variant="ghost" className="min-h-11 min-w-11" disabled={index === ranking.length - 1} onClick={() => move(index, index + 1)} aria-label={`Move candidate ${index + 1} down`}><ArrowDown /></Button>
        </div>
      </li>)}
    </ol>
    <p aria-live="polite" className="mt-2 text-[9.5px] text-fg-disabled">Use the arrow controls to change the order. Every candidate remains in the final ranking.</p>
  </div>;
}

function StructuredCorrectionEditor({ value, onChange }: { value: unknown; onChange: (correction: Record<string, unknown>) => void }) {
  const correction = asRecord(value);
  const messages = objectArray(correction.messages);
  const tools = objectArray(correction.tools);
  const calls = objectArray(correction.expected_calls);
  const results = objectArray(correction.expected_results);
  const setSection = (key: "messages" | "tools" | "expected_calls" | "expected_results", next: Record<string, unknown>[]) => onChange({ ...correction, [key]: next });
  return <div className="space-y-6">
    <p className="text-[10.5px] leading-5 text-fg-subtle">Correct the trace using the fields below. Empty sections are omitted from the training target; no JSON editing is required.</p>
    <StructuredListSection title="Messages" description="The ordered conversation that leads to the tool decision." count={messages.length} onAdd={() => setSection("messages", [...messages, { role: "assistant", content: "" }])}>
      {messages.map((message, index) => <StructuredRow key={index} label={`Message ${index + 1}`} onRemove={() => setSection("messages", removeAt(messages, index))}>
        <div className="grid gap-3 sm:grid-cols-2"><GuidedField label="Role"><select aria-label={`Message ${index + 1} role`} value={String(message.role ?? "assistant")} onChange={(event) => setSection("messages", replaceAt(messages, index, { ...message, role: event.target.value }))} className={guidedSelectClass}><option value="system">System</option><option value="user">User</option><option value="assistant">Assistant</option><option value="tool">Tool</option></select></GuidedField><GuidedField label="Name (optional)"><Input aria-label={`Message ${index + 1} name`} value={String(message.name ?? "")} onChange={(event) => setSection("messages", replaceAt(messages, index, withOptionalString(message, "name", event.target.value)))} placeholder="tool or participant name" /></GuidedField></div>
        <GuidedField label="Content"><textarea aria-label={`Message ${index + 1} content`} value={String(message.content ?? "")} onChange={(event) => setSection("messages", replaceAt(messages, index, { ...message, content: event.target.value }))} rows={3} className={guidedTextareaClass} placeholder="Corrected message content…" /></GuidedField>
        {message.role === "tool" ? <GuidedField label="Tool call ID (optional)"><Input aria-label={`Message ${index + 1} tool call ID`} value={String(message.tool_call_id ?? "")} onChange={(event) => setSection("messages", replaceAt(messages, index, withOptionalString(message, "tool_call_id", event.target.value)))} placeholder="call_…" /></GuidedField> : null}
      </StructuredRow>)}
    </StructuredListSection>

    <StructuredListSection title="Tools" description="Functions available to the model, including their argument fields." count={tools.length} onAdd={() => setSection("tools", [...tools, newToolDefinition()])}>
      {tools.map((tool, index) => <ToolDefinitionEditor key={index} index={index} tool={tool} onChange={(next) => setSection("tools", replaceAt(tools, index, next))} onRemove={() => setSection("tools", removeAt(tools, index))} />)}
    </StructuredListSection>

    <StructuredListSection title="Expected calls" description="The tool calls the model should make, in order." count={calls.length} onAdd={() => setSection("expected_calls", [...calls, { name: "", arguments: {} }])}>
      {calls.map((call, index) => <ExpectedCallEditor key={index} index={index} call={call} onChange={(next) => setSection("expected_calls", replaceAt(calls, index, next))} onRemove={() => setSection("expected_calls", removeAt(calls, index))} />)}
    </StructuredListSection>

    <StructuredListSection title="Expected results" description="The tool responses that should return to the model." count={results.length} onAdd={() => setSection("expected_results", [...results, { name: "", content: "" }])}>
      {results.map((result, index) => <StructuredRow key={index} label={`Result ${index + 1}`} onRemove={() => setSection("expected_results", removeAt(results, index))}>
        <div className="grid gap-3 sm:grid-cols-2"><GuidedField label="Tool name"><Input aria-label={`Expected result ${index + 1} tool name`} value={String(result.name ?? result.tool_name ?? "")} onChange={(event) => setSection("expected_results", replaceAt(results, index, { ...result, name: event.target.value }))} placeholder="get_weather" /></GuidedField><GuidedField label="Tool call ID (optional)"><Input aria-label={`Expected result ${index + 1} tool call ID`} value={String(result.tool_call_id ?? "")} onChange={(event) => setSection("expected_results", replaceAt(results, index, withOptionalString(result, "tool_call_id", event.target.value)))} placeholder="call_…" /></GuidedField></div>
        <GuidedField label="Result content"><textarea aria-label={`Expected result ${index + 1} content`} value={String(result.content ?? result.result ?? result.output ?? "")} onChange={(event) => setSection("expected_results", replaceAt(results, index, setResultContent(result, event.target.value)))} rows={3} className={guidedTextareaClass} placeholder="Correct tool output…" /></GuidedField>
      </StructuredRow>)}
    </StructuredListSection>
  </div>;
}

function StructuredListSection({ title, description, count, onAdd, children }: { title: string; description: string; count: number; onAdd: () => void; children: ReactNode }) {
  return <section aria-labelledby={`structured-${title.toLowerCase().replace(/\s+/g, "-")}`}>
    <div className="mb-2 flex items-start justify-between gap-3"><div><h3 id={`structured-${title.toLowerCase().replace(/\s+/g, "-")}`} className="text-[11px] font-semibold text-fg">{title} <span className="font-mono text-[9px] font-normal text-fg-disabled">{count}</span></h3><p className="mt-0.5 text-[9.5px] leading-4 text-fg-disabled">{description}</p></div><Button type="button" size="sm" variant="ghost" className="min-h-11 shrink-0" onClick={onAdd}><Plus />Add</Button></div>
    {count ? <div className="space-y-2">{children}</div> : <button type="button" onClick={onAdd} className="min-h-16 w-full rounded-md border border-dashed border-border px-3 text-[10px] text-fg-disabled transition-colors hover:border-border-strong hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30">No {title.toLowerCase()} · add one</button>}
  </section>;
}

function StructuredRow({ label, onRemove, children }: { label: string; onRemove: () => void; children: ReactNode }) {
  return <section className="rounded-md border border-border bg-surface/35 p-3"><div className="mb-3 flex items-center justify-between gap-3"><h4 className="font-mono text-[9px] font-medium uppercase tracking-wider text-fg-muted">{label}</h4><Button type="button" size="icon" variant="ghost" className="min-h-11 min-w-11 text-fg-disabled hover:text-danger" onClick={onRemove} aria-label={`Remove ${label}`}><Trash2 /></Button></div><div className="space-y-3">{children}</div></section>;
}

function ToolDefinitionEditor({ index, tool, onChange, onRemove }: { index: number; tool: Record<string, unknown>; onChange: (tool: Record<string, unknown>) => void; onRemove: () => void }) {
  const details = toolDetails(tool);
  const updateDetails = (next: ToolDetails) => onChange(buildToolDefinition(tool, next));
  return <StructuredRow label={`Tool ${index + 1}`} onRemove={onRemove}>
    <div className="grid gap-3 sm:grid-cols-2"><GuidedField label="Function name"><Input aria-label={`Tool ${index + 1} function name`} value={details.name} onChange={(event) => updateDetails({ ...details, name: event.target.value })} placeholder="get_weather" /></GuidedField><GuidedField label="Description"><Input aria-label={`Tool ${index + 1} description`} value={details.description} onChange={(event) => updateDetails({ ...details, description: event.target.value })} placeholder="What this tool does" /></GuidedField></div>
    <div><div className="mb-2 flex items-center justify-between gap-3"><Label className="text-[9.5px]">Arguments</Label><Button type="button" size="sm" variant="ghost" className="min-h-11" onClick={() => updateDetails({ ...details, parameters: [...details.parameters, { name: `argument_${details.parameters.length + 1}`, type: "string", description: "", required: false, schema: {} }] })}><Plus />Add argument</Button></div>{details.parameters.length ? <div className="space-y-2">{details.parameters.map((parameter, parameterIndex) => <div key={parameterIndex} className="grid gap-2 rounded border border-border-subtle bg-bg/55 p-2 sm:grid-cols-[minmax(0,1fr)_7rem_auto_auto]"><Input aria-label={`Tool ${index + 1} argument ${parameterIndex + 1} name`} value={parameter.name} onChange={(event) => updateDetails({ ...details, parameters: replaceAt(details.parameters, parameterIndex, { ...parameter, name: event.target.value }) })} placeholder="city" /><select aria-label={`Tool ${index + 1} argument ${parameterIndex + 1} type`} value={parameter.type} onChange={(event) => updateDetails({ ...details, parameters: replaceAt(details.parameters, parameterIndex, { ...parameter, type: event.target.value }) })} className={guidedSelectClass}><option value="string">Text</option><option value="number">Number</option><option value="integer">Integer</option><option value="boolean">True / false</option><option value="array">List</option><option value="object">Object</option></select><label className="flex min-h-11 items-center gap-2 px-1 text-[9.5px] text-fg-muted"><input type="checkbox" checked={parameter.required} onChange={(event) => updateDetails({ ...details, parameters: replaceAt(details.parameters, parameterIndex, { ...parameter, required: event.target.checked }) })} />Required</label><Button type="button" size="icon" variant="ghost" className="min-h-11 min-w-11 text-fg-disabled hover:text-danger" onClick={() => updateDetails({ ...details, parameters: removeAt(details.parameters, parameterIndex) })} aria-label={`Remove argument ${parameterIndex + 1}`}><Trash2 /></Button><Input aria-label={`Tool ${index + 1} argument ${parameterIndex + 1} description`} className="sm:col-span-4" value={parameter.description} onChange={(event) => updateDetails({ ...details, parameters: replaceAt(details.parameters, parameterIndex, { ...parameter, description: event.target.value }) })} placeholder="Argument description (optional)" /></div>)}</div> : <p className="rounded border border-dashed border-border px-3 py-3 text-[9.5px] text-fg-disabled">This function takes no arguments.</p>}</div>
  </StructuredRow>;
}

function ExpectedCallEditor({ index, call, onChange, onRemove }: { index: number; call: Record<string, unknown>; onChange: (call: Record<string, unknown>) => void; onRemove: () => void }) {
  const argumentSource = asRecord(call.arguments ?? call.parameters ?? call.args);
  const argumentsList = Object.entries(argumentSource).map(([name, value]) => ({ name, value: formatGuidedValue(value) }));
  const setArguments = (next: Array<{ name: string; value: string }>) => {
    const parsed = Object.fromEntries(next.filter((argument) => argument.name.trim()).map((argument) => [argument.name.trim(), parseGuidedValue(argument.value)]));
    const key = "parameters" in call ? "parameters" : "args" in call ? "args" : "arguments";
    onChange({ ...call, [key]: parsed });
  };
  return <StructuredRow label={`Call ${index + 1}`} onRemove={onRemove}>
    <div className="grid gap-3 sm:grid-cols-2"><GuidedField label="Tool name"><Input aria-label={`Expected call ${index + 1} tool name`} value={String(call.name ?? call.tool_name ?? call.function ?? "")} onChange={(event) => onChange({ ...call, name: event.target.value })} placeholder="get_weather" /></GuidedField><GuidedField label="Tool call ID (optional)"><Input aria-label={`Expected call ${index + 1} tool call ID`} value={String(call.id ?? call.tool_call_id ?? "")} onChange={(event) => onChange(withOptionalString(call, "id", event.target.value))} placeholder="call_…" /></GuidedField></div>
    <div><div className="mb-2 flex items-center justify-between gap-3"><Label className="text-[9.5px]">Arguments</Label><Button type="button" size="sm" variant="ghost" className="min-h-11" onClick={() => setArguments([...argumentsList, { name: `argument_${argumentsList.length + 1}`, value: "" }])}><Plus />Add argument</Button></div>{argumentsList.length ? <div className="space-y-2">{argumentsList.map((argument, argumentIndex) => <div key={argumentIndex} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_auto] gap-2"><Input aria-label={`Expected call ${index + 1} argument ${argumentIndex + 1} name`} value={argument.name} onChange={(event) => setArguments(replaceAt(argumentsList, argumentIndex, { ...argument, name: event.target.value }))} placeholder="argument" /><Input aria-label={`Expected call ${index + 1} argument ${argumentIndex + 1} value`} value={argument.value} onChange={(event) => setArguments(replaceAt(argumentsList, argumentIndex, { ...argument, value: event.target.value }))} placeholder="value" /><Button type="button" size="icon" variant="ghost" className="min-h-11 min-w-11 text-fg-disabled hover:text-danger" onClick={() => setArguments(removeAt(argumentsList, argumentIndex))} aria-label={`Remove call argument ${argumentIndex + 1}`}><Trash2 /></Button></div>)}</div> : <p className="rounded border border-dashed border-border px-3 py-3 text-[9.5px] text-fg-disabled">This call has no arguments.</p>}</div>
  </StructuredRow>;
}

function GuidedField({ label, children }: { label: string; children: ReactNode }) { return <div className="space-y-1.5"><Label className="text-[9.5px]">{label}</Label>{children}</div>; }

function DecisionButton({ active, tone, shortcut, label, onClick }: { active: boolean; tone?: "accept" | "reject"; shortcut?: string; label: string; onClick: () => void }) { return <button type="button" aria-pressed={active} onClick={onClick} className={cn("flex min-h-11 items-center justify-between gap-3 rounded-md border px-3 py-2 text-left text-[11.5px] font-medium transition-colors", active ? tone === "reject" ? "border-danger/60 bg-danger/10 text-fg" : "border-accent bg-accent/8 text-fg" : "border-border bg-surface/45 text-fg-muted hover:border-border-strong hover:text-fg")}><span>{label}</span>{shortcut ? <kbd className="rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[8.5px] text-fg-disabled">{shortcut}</kbd> : null}</button>; }

function ActionDock({ draft, schema, pending, disabled, skipDisabled, saving, onSubmit, onSkip, onExclude, onFlag }: { draft: ReviewDraft; schema: AnnotationSchemaRevision; pending: boolean; disabled: boolean; skipDisabled: boolean; saving: boolean; onSubmit: () => void; onSkip: () => void; onExclude: (reason: string) => void; onFlag: (reason: string) => void }) {
  const [secondary, setSecondary] = useState<"exclude" | "flag" | null>(null); const [reason, setReason] = useState("");
  const valid = annotationReady(schema, draft.annotation);
  return <div className="fixed inset-x-0 bottom-0 z-30 border-t border-border bg-elevated/95 px-3 pb-[max(.75rem,env(safe-area-inset-bottom))] pt-3 shadow-2xl shadow-black/35 backdrop-blur lg:static lg:bg-bg-subtle/80 lg:px-4 lg:pb-3 lg:shadow-none">
    {secondary ? <div className="mx-auto mb-2 flex max-w-4xl items-end gap-2"><div className="min-w-0 flex-1"><Label htmlFor="review-secondary-reason">{secondary === "exclude" ? "Exclusion reason" : "Flag reason"}</Label><Input id="review-secondary-reason" autoFocus value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Required for an auditable decision" /></div><Button size="sm" variant={secondary === "exclude" ? "danger" : "secondary"} disabled={!reason.trim() || pending} onClick={() => { if (secondary === "exclude") onExclude(reason.trim()); else onFlag(reason.trim()); }}>{secondary === "exclude" ? <CircleSlash2 /> : <Flag />}Confirm</Button><Button size="sm" variant="ghost" onClick={() => { setSecondary(null); setReason(""); }}>Cancel</Button></div> : null}
    <div className="mx-auto flex max-w-4xl items-center gap-2 pl-10 sm:pl-0"><div className="hidden flex-1 items-center gap-2 text-[9.5px] text-fg-disabled sm:flex">{saving ? <><Loader2 className="h-3 w-3 animate-spin" />Saving draft…</> : <><Save className="h-3 w-3" />Draft autosaved</>}</div><Button className="px-2" size="sm" variant="ghost" onClick={() => setSecondary("flag")} aria-label="Flag item"><Flag /><span className="hidden sm:inline">Flag</span></Button><Button className="px-2" size="sm" variant="ghost" onClick={() => setSecondary("exclude")} aria-label="Exclude item"><CircleSlash2 /><span className="hidden sm:inline">Exclude</span></Button><Button size="sm" variant="secondary" disabled={skipDisabled} onClick={onSkip}>{pending ? <Loader2 className="animate-spin" /> : null}Skip <span className="font-mono text-[8px] opacity-60">S</span></Button><Button className="min-w-24 flex-1 sm:min-w-28 sm:flex-none" size="sm" variant="primary" disabled={disabled || !valid || pending} onClick={onSubmit}>{pending ? <Loader2 className="animate-spin" /> : <Check />}Submit</Button></div>
  </div>;
}

function ReviewInspector({ queue, item, schema, events, suggestions }: { queue: ReviewQueue; item: ReviewItem; schema: AnnotationSchemaRevision; events: ReviewEvent[]; suggestions: Array<{ id: string; output: Record<string, unknown> | null; model_revision?: string; created_at?: string | null }> }) {
  const [tab, setTab] = useState<"guide" | "history" | "handoff">("guide");
  return <aside className="hidden min-h-0 border-l border-border bg-bg-subtle/45 lg:flex lg:flex-col" aria-label="Review inspector"><div className="flex border-b border-border px-2 pt-2">{(["guide", "history", "handoff"] as const).map((value) => <button key={value} type="button" onClick={() => setTab(value)} className={cn("border-b-2 px-2 py-2 text-[9.5px] font-medium uppercase tracking-wider", tab === value ? "border-accent text-fg" : "border-transparent text-fg-disabled hover:text-fg")}>{value}</button>)}</div><div className="min-h-0 flex-1 overflow-y-auto p-3">{tab === "guide" ? <GuidePanel queue={queue} item={item} schema={schema} suggestions={suggestions} /> : tab === "history" ? <HistoryPanel events={events} /> : <LabelSetHandoff queue={queue} schema={schema} />}</div></aside>;
}

function GuidePanel({ queue, item, schema, suggestions }: { queue: ReviewQueue; item: ReviewItem; schema: AnnotationSchemaRevision; suggestions: Array<{ id: string; output: Record<string, unknown> | null; model_revision?: string }> }) {
  const queryClient = useQueryClient(); const [model, setModel] = useState("");
  const suggestion = useMutation({ mutationFn: () => api.generateReviewSuggestion(item.id, { provider: "openai_compatible", model_revision: model, pass_number: queue.current_pass }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["review-item-suggestions", item.id] }) });
  const reveal = useMutation({ mutationFn: (suggestionId: string) => api.submitReviewEvent(item.id, { event_type: "reveal_suggestion", pass_number: queue.current_pass, payload: { suggestion_id: suggestionId }, idempotency_key: eventKey(queue.id, item.id, `reveal-${suggestionId}`), expected_active_event_id: item.active_event_id ?? null }), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["review-item-suggestions", item.id] }); queryClient.invalidateQueries({ queryKey: ["review-item", item.id] }); queryClient.invalidateQueries({ queryKey: ["review-queue-items", queue.id] }); } });
  const queuedSuggestionId = suggestion.data?.status === "queued" ? suggestion.data.id : null;
  const queuedSuggestionReady = Boolean(queuedSuggestionId && suggestions.some((value) => value.id === queuedSuggestionId));
  useEffect(() => {
    if (!queuedSuggestionId || queuedSuggestionReady) return;
    const timer = window.setInterval(() => queryClient.invalidateQueries({ queryKey: ["review-item-suggestions", item.id] }), 2_000);
    return () => window.clearInterval(timer);
  }, [item.id, queryClient, queuedSuggestionId, queuedSuggestionReady]);
  return <div className="space-y-4"><InspectorSection icon={<ShieldCheck />} title="Pinned rubric"><DefinitionRows definition={schema.definition} /><div className="mt-2 font-mono text-[8.5px] text-fg-disabled">schema {schema.id} · r{schema.revision_number}</div></InspectorSection><InspectorSection icon={<Database />} title="Source evidence"><DefinitionRows definition={item.source ?? {}} />{item.record_hash ? <div className="mt-2 break-all font-mono text-[8.5px] text-fg-disabled">{item.record_hash}</div> : null}</InspectorSection>{queue.policy.allow_suggestions ? <InspectorSection icon={<Bot />} title="Model suggestion"><p className="mb-2 text-[9.5px] leading-4 text-fg-subtle">Suggestions are optional, provenance-stamped, and never submit a label.</p>{suggestions.map((value) => <div key={value.id} className="mb-2 rounded border border-border bg-bg p-2"><div className="text-[9px] text-fg-disabled">{value.model_revision}</div>{value.output ? <pre className="mt-1 overflow-auto whitespace-pre-wrap font-mono text-[9px] text-fg">{JSON.stringify(value.output, null, 2)}</pre> : <Button className="mt-2 w-full" size="sm" variant="ghost" disabled={reveal.isPending} onClick={() => reveal.mutate(value.id)}><Eye />Reveal suggestion</Button>}</div>)}{queuedSuggestionId && !queuedSuggestionReady ? <div className="mb-2 flex items-center gap-2 rounded border border-accent/25 bg-accent/7 px-2.5 py-2 text-[9.5px] text-fg-muted"><Loader2 className="h-3 w-3 animate-spin text-accent" /><span>Suggestion queued in Activity. This panel will update when it is ready.</span></div> : null}<Input value={model} onChange={(event) => setModel(event.target.value)} placeholder="Pinned model revision" /><Button className="mt-2 w-full" size="sm" variant="ghost" disabled={!model.trim() || suggestion.isPending || Boolean(queuedSuggestionId && !queuedSuggestionReady)} onClick={() => suggestion.mutate()}>{suggestion.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}Request suggestion</Button>{suggestion.error ? <p role="alert" className="mt-2 text-[9.5px] leading-4 text-danger">{suggestion.error.message}</p> : null}</InspectorSection> : null}<InspectorSection icon={<Keyboard />} title="Keyboard"><Shortcut label="Previous / next" keys="K / J" /><Shortcut label="Accept / reject" keys="A / R" /><Shortcut label="Skip" keys="S" /><Shortcut label="Choose option" keys="1 / 2" /></InspectorSection></div>;
}

function HistoryPanel({ events }: { events: ReviewEvent[] }) { return <div>{events.length ? <ol className="space-y-3">{events.map((event) => <li key={event.id} className="relative border-l border-border pl-3"><span className="absolute -left-1 top-1 h-2 w-2 rounded-full bg-accent" /><div className="flex items-center justify-between gap-2"><span className="text-[10px] font-medium text-fg">{humanize(event.event_type)}</span><span className="font-mono text-[8px] text-fg-disabled">pass {event.pass_number}</span></div><pre className="mt-1 overflow-auto whitespace-pre-wrap font-mono text-[8.5px] leading-4 text-fg-subtle">{JSON.stringify(event.payload, null, 2)}</pre><div className="mt-1 text-[8px] text-fg-disabled">{event.created_at ? new Date(event.created_at).toLocaleString() : ""}</div></li>)}</ol> : <div className="py-8 text-center text-[10px] text-fg-disabled">No submitted decisions yet.</div>}</div>; }

function LabelSetHandoff({ queue, schema }: { queue: ReviewQueue; schema: AnnotationSchemaRevision }) {
  const queryClient = useQueryClient();
  const [publication, setPublication] = useState<(LabelSetPublicationAccepted & { previous_revision_id: string | null }) | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [parentVersionId, setParentVersionId] = useState("");
  const [mode, setMode] = useState("");
  const [materialize, setMaterialize] = useState(false);
  const capabilities = useQuery({ queryKey: ["review-capabilities"], queryFn: api.reviewCapabilities });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const versions = useQuery({ queryKey: ["dataset-versions", datasetId], queryFn: () => api.datasetVersions(datasetId), enabled: Boolean(datasetId) });
  const liveQueue = useQuery({
    queryKey: ["review-queue", queue.id],
    queryFn: () => api.reviewQueue(queue.id),
    initialData: queue,
    refetchInterval: (query) => {
      if (!publication) return false;
      const latest = (query.state.data as ReviewQueue | undefined)?.latest_label_set_revision_id ?? null;
      return latest && latest !== publication.previous_revision_id ? false : 2_000;
    },
  });
  const latestRevisionId = liveQueue.data?.latest_label_set_revision_id ?? queue.latest_label_set_revision_id ?? null;
  const publicationReady = Boolean(publication && latestRevisionId && latestRevisionId !== publication.previous_revision_id);
  const existingRevision = useQuery({ queryKey: ["label-set-revision", latestRevisionId], queryFn: () => api.labelSetRevision(latestRevisionId!), enabled: Boolean(latestRevisionId) });
  const activeRevision = existingRevision.data ?? null;
  const requestedAdapter = String(schema.definition.output_adapter_id || "");
  const adapter = capabilities.data?.output_adapters.find((value) => value.id === requestedAdapter) ?? capabilities.data?.output_adapters.find((value) => value.modalities.includes(schema.modality) && value.task_types.includes(schema.task_type));
  const buildModes = adapter?.build_modes ?? ["annotate"];
  const selectedMode = buildModes.includes(mode) ? mode : adapter?.default_build_mode ?? buildModes[0];
  const publish = useMutation({ mutationFn: () => api.publishReviewLabelSet(queue.id, { name: `${queue.name} labels` }), onSuccess: (accepted) => { setPublication({ ...accepted, previous_revision_id: latestRevisionId }); queryClient.invalidateQueries({ queryKey: ["review-queue", queue.id] }); queryClient.invalidateQueries({ queryKey: ["activity"] }); } });
  const verify = useMutation({ mutationFn: () => api.verifyLabelSetRevision(activeRevision!.id) });
  const preview = useMutation({ mutationFn: () => api.previewLabelSetDataset(activeRevision!.id, handoffPayload()) });
  const build = useMutation({ mutationFn: () => api.buildLabelSetDataset(activeRevision!.id, handoffPayload()) });
  function handoffPayload() { return { output_adapter_id: adapter?.id || requestedAdapter || undefined, build_mode: selectedMode, dataset_id: datasetId || undefined, parent_version_id: parentVersionId || undefined, name: `${queue.name} reviewed`, materialize_assets: materialize }; }
  const error = publish.error || verify.error || preview.error || build.error;
  return <div className="space-y-4"><InspectorSection icon={<CheckCircle2 />} title="Immutable label set"><p className="text-[9.5px] leading-4 text-fg-subtle">Publish the current reviewed decisions, verify checksums, then preview their dataset effect.</p>{publication && !publicationReady ? <div role="status" className="mt-2 rounded border border-accent/25 bg-accent/7 p-2.5"><div className="flex items-center gap-2 text-[9.5px] font-medium text-fg"><Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />Publishing label set</div><p className="mt-1 text-[9px] leading-4 text-fg-subtle">Queued in Activity. This panel will load the new revision when checksums and evidence files are ready.</p><div className="mt-1 break-all font-mono text-[8px] text-fg-disabled">work {publication.work_item_id}</div></div> : null}{activeRevision ? <div className="mt-2 rounded border border-border bg-bg p-2"><div className="font-mono text-[9px] text-fg">r{activeRevision.revision_number} · {activeRevision.row_count} labels</div><div className="mt-1 break-all font-mono text-[8px] text-fg-disabled">{activeRevision.content_hash}</div></div> : null}<Button className="mt-2 w-full" size="sm" onClick={() => publish.mutate()} disabled={publish.isPending || Boolean(publication && !publicationReady)}>{publish.isPending || publication && !publicationReady ? <Loader2 className="animate-spin" /> : <ShieldCheck />}{activeRevision ? "Publish new revision" : "Publish label set"}</Button>{activeRevision ? <Button className="mt-2 w-full" size="sm" variant="ghost" onClick={() => verify.mutate()} disabled={verify.isPending}>{verify.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}{verify.data?.valid ? "Verified" : "Verify integrity"}</Button> : null}</InspectorSection>
    {activeRevision ? <InspectorSection icon={<Database />} title="Dataset handoff"><FieldSmall label="Destination dataset"><SearchPicker value={datasetId} onChange={(value) => { setDatasetId(value); setParentVersionId(""); }} options={(datasets.data?.items ?? []).map((dataset: DatasetRecord) => ({ value: dataset.id, label: dataset.name, description: `${dataset.modality || "text"} · ${dataset.latest_version?.row_count ?? dataset.row_count ?? 0} rows` }))} allowEmpty placeholder="New dataset or choose existing" /></FieldSmall>{datasetId ? <FieldSmall label="Parent version"><SearchPicker value={parentVersionId} onChange={setParentVersionId} options={(versions.data?.items ?? []).map((value) => ({ value: value.id, label: value.label || `v${value.version || ""}`, description: `${value.row_count ?? 0} rows · ${value.status}` }))} allowEmpty placeholder="Append without parent" /></FieldSmall> : null}<FieldSmall label="Build mode"><select value={selectedMode} onChange={(event) => setMode(event.target.value)} className={selectClass}>{buildModes.map((value) => <option key={value} value={value}>{buildModeLabel(value)}</option>)}</select></FieldSmall><div className="font-mono text-[8px] text-fg-disabled">{adapter?.id || "Compatible adapter resolves at preview"}</div><label className="flex items-center gap-2 text-[9.5px] text-fg-muted"><input type="checkbox" checked={materialize} onChange={(event) => setMaterialize(event.target.checked)} />Copy referenced media into managed storage</label><div className="grid grid-cols-2 gap-2"><Button size="sm" variant="ghost" disabled={preview.isPending} onClick={() => preview.mutate()}>{preview.isPending ? <Loader2 className="animate-spin" /> : <Eye />}Preview</Button><Button size="sm" disabled={build.isPending} onClick={() => build.mutate()}>{build.isPending ? <Loader2 className="animate-spin" /> : <Database />}Build</Button></div>{preview.data ? <SemanticDatasetPreview preview={preview.data} /> : null}{build.data ? <div className="rounded border border-success/30 bg-success/8 p-2 text-[9.5px] text-fg">Dataset build queued · {build.data.job_id || build.data.id}</div> : null}</InspectorSection> : null}
    {error ? <p role="alert" className="text-[9.5px] leading-4 text-danger">{error.message}</p> : null}
  </div>;
}

function InspectorSection({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) { return <section className="border-b border-border-subtle pb-4 last:border-0"><h3 className="mb-2 flex items-center gap-2 text-[9.5px] font-medium uppercase tracking-wider text-fg-muted"><span className="text-accent [&_svg]:h-3.5 [&_svg]:w-3.5">{icon}</span>{title}</h3>{children}</section>; }
function DefinitionRows({ definition }: { definition: Record<string, unknown> }) { const values = Object.entries(definition).filter(([, value]) => value !== undefined && value !== null); return <dl className="space-y-1.5">{values.map(([key, value]) => <div key={key} className="flex items-start justify-between gap-3 text-[9px]"><dt className="text-fg-disabled">{humanize(key)}</dt><dd className="max-w-[60%] break-words text-right font-mono text-fg-muted">{renderValue(value)}</dd></div>)}</dl>; }
function Shortcut({ label, keys }: { label: string; keys: string }) { return <div className="flex items-center justify-between border-b border-border-subtle py-1.5 text-[9px] last:border-0"><span className="text-fg-subtle">{label}</span><kbd className="rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[8px] text-fg-disabled">{keys}</kbd></div>; }
function FieldSmall({ label, children }: { label: string; children: ReactNode }) { return <div className="space-y-1"><Label className="text-[9.5px]">{label}</Label>{children}</div>; }
function SemanticDatasetPreview({ preview }: { preview: DatasetBuildPreview }) { const contamination = preview.contamination ?? {}; const affected = Object.values(preview.moved_from_splits ?? {}).reduce((sum, value) => sum + Number(value || 0), 0); const mediaIssues = Number(contamination.media_overlap_count ?? contamination.media_contamination_count ?? 0); return <div className="mt-2 overflow-hidden rounded border border-border bg-border"><div className="grid grid-cols-3 gap-px"><PreviewCount label="Added" value={preview.added_count} /><PreviewCount label="Removed" value={preview.removed_count} /><PreviewCount label="Replaced" value={preview.replaced_count} /><PreviewCount label="Quarantined" value={preview.quarantined_count} tone={preview.quarantined_count ? "warning" : undefined} /><PreviewCount label="Split affected" value={affected} /><PreviewCount label="Media overlap" value={mediaIssues} tone={mediaIssues ? "danger" : undefined} /></div>{preview.warnings?.length ? <ul className="border-t border-border bg-bg px-2.5 py-2 text-[8.5px] leading-4 text-warning">{preview.warnings.map((warning) => <li key={warning}>• {warning}</li>)}</ul> : <div className="border-t border-border bg-bg px-2.5 py-2 text-[8.5px] text-fg-disabled">No contamination or split warnings in this preview.</div>}</div>; }
function PreviewCount({ label, value, tone }: { label: string; value?: number; tone?: "warning" | "danger" }) { return <div className="bg-bg p-2"><div className={cn("font-mono text-[11px] text-fg", tone === "warning" && "text-warning", tone === "danger" && "text-danger")}>{value ?? 0}</div><div className="text-[7.5px] uppercase tracking-wider text-fg-disabled">{label}</div></div>; }

function useReviewKeyboard({ schema, item, flipCandidates, onPrevious, onNext, onSkip, onLabel, disabled }: { schema: AnnotationSchemaRevision; item: ReviewItem; flipCandidates: boolean; onPrevious: () => void; onNext: () => void; onSkip: () => void; onLabel: (value: Record<string, unknown>) => void; disabled: boolean }) {
  useEffect(() => { const handler = (event: KeyboardEvent) => { const target = event.target as HTMLElement | null; if (disabled || target?.closest("input, textarea, select, button, [contenteditable='true']")) return; const key = event.key.toLowerCase(); const [optionA, optionB] = pairwiseCandidates(item, flipCandidates); if (key === "j" || event.key === "ArrowRight") { event.preventDefault(); onNext(); } else if (key === "k" || event.key === "ArrowLeft") { event.preventDefault(); onPrevious(); } else if (key === "s") { event.preventDefault(); onSkip(); } else if (schema.task_type === "binary" && key === "a") { event.preventDefault(); onLabel({ accepted: true }); } else if (schema.task_type === "binary" && key === "r") { event.preventDefault(); onLabel({ accepted: false }); } else if (schema.task_type === "pairwise" && key === "1") { event.preventDefault(); onLabel({ chosen: optionA, rejected: optionB }); } else if (schema.task_type === "pairwise" && key === "2") { event.preventDefault(); onLabel({ chosen: optionB, rejected: optionA }); } else if (schema.task_type === "pairwise" && key === "t") { event.preventDefault(); onLabel({ chosen: "tie" }); } }; window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler); }, [disabled, flipCandidates, item, onLabel, onNext, onPrevious, onSkip, schema.task_type]);
}

function initialAnnotation(schema: AnnotationSchemaRevision, item: ReviewItem, passNumber: number, flipCandidates: boolean): Record<string, unknown> { const pass = item.projection?.[`pass_${passNumber}`] as { annotation?: unknown } | undefined; const adjudication = item.projection?.adjudication as { annotation?: unknown } | undefined; const existing = adjudication?.annotation ?? pass?.annotation; if (existing && typeof existing === "object" && !Array.isArray(existing)) return existing as Record<string, unknown>; if (schema.task_type === "binary") return { accepted: undefined }; if (schema.task_type === "categorical") return { label: "" }; if (schema.task_type === "multi_label") return { labels: [] }; if (schema.task_type === "scalar") return { score: Number(schema.definition.minimum ?? 0) }; if (schema.task_type === "pairwise") return { chosen: "" }; if (schema.task_type === "ranking") return { ranking: rankingCandidates(item, flipCandidates) }; if (schema.task_type === "structured_correction") return { correction: structuredCorrectionSeed(item.record ?? {}) }; return { corrected_text: String(item.record?.response ?? item.record?.output ?? item.record?.transcript ?? "") }; }
function annotationReady(schema: AnnotationSchemaRevision, value: Record<string, unknown>) { if (schema.task_type === "binary") return typeof value.accepted === "boolean"; if (schema.task_type === "categorical") return Boolean(value.label); if (schema.task_type === "multi_label") return Array.isArray(value.labels); if (schema.task_type === "scalar") return Number.isFinite(Number(value.score)); if (schema.task_type === "pairwise") return Boolean(value.chosen); if (schema.task_type === "ranking") return Array.isArray(value.ranking) && value.ranking.length >= 2; if (schema.task_type === "structured_correction") return Object.values(asRecord(value.correction)).some((entry) => Array.isArray(entry) ? entry.length > 0 : entry !== undefined && entry !== null && entry !== ""); return Boolean(String(value.corrected_text || "").trim()); }
function decisionTitle(task: string, adjudication: boolean, correction: boolean) { if (adjudication) return "Resolve the conflicting decisions"; if (correction) return "Correct the active decision"; if (task === "pairwise") return "Which response is better?"; if (task.includes("correction")) return "Write the training target"; return "Apply the pinned rubric"; }
function recordHeading(record: Record<string, unknown>, modality: string) { const title = record.title ?? record.task ?? record.category; return title ? String(title) : modality === "vlm" ? "Image and prompt" : modality === "audio" ? "Audio and transcript" : modality === "tool" ? "Tool interaction" : modality === "preference" ? "Response comparison" : "Training example"; }
function shouldFlipCandidates(queue: ReviewQueue, item: ReviewItem) { const presentation = item.projection?.presentation as { pass_2_flip_candidates?: boolean } | undefined; return queue.current_pass === 2 && Boolean(queue.policy.blind_second_pass) && presentation?.pass_2_flip_candidates === true; }
function pairwiseCandidates(item: ReviewItem, flip: boolean): [unknown, unknown] { const record = item.record ?? {}; const alternatives = Array.isArray(record.alternatives) ? record.alternatives : Array.isArray(record.candidates) ? record.candidates : Array.isArray(record.responses) ? record.responses : []; const first = record.chosen ?? alternatives[0]; const second = record.rejected ?? alternatives[1]; return flip ? [second, first] : [first, second]; }
function rankingCandidates(item: ReviewItem, flip: boolean) { const record = item.record ?? {}; const source = Array.isArray(record.alternatives) ? record.alternatives : Array.isArray(record.candidates) ? record.candidates : Array.isArray(record.responses) ? record.responses : [record.chosen, record.rejected]; const unique = source.filter((value) => value !== undefined && value !== null).filter((value, index, values) => values.findIndex((candidate) => choiceMatches(candidate, value)) === index); return flip ? [...unique].reverse() : [...unique]; }
function choiceMatches(left: unknown, right: unknown) { if (Object.is(left, right)) return true; try { return JSON.stringify(left) === JSON.stringify(right); } catch { return false; } }
function stableValueKey(value: unknown) { try { return JSON.stringify(value); } catch { return String(value); } }
function asRecord(value: unknown): Record<string, unknown> { return value !== null && typeof value === "object" && !Array.isArray(value) ? { ...(value as Record<string, unknown>) } : {}; }
function objectArray(value: unknown): Record<string, unknown>[] { return Array.isArray(value) ? value.map((entry) => entry !== null && typeof entry === "object" && !Array.isArray(entry) ? { ...(entry as Record<string, unknown>) } : { content: String(entry ?? "") }) : []; }
function replaceAt<T>(values: T[], index: number, value: T) { return values.map((entry, entryIndex) => entryIndex === index ? value : entry); }
function removeAt<T>(values: T[], index: number) { return values.filter((_, entryIndex) => entryIndex !== index); }
function withOptionalString(value: Record<string, unknown>, key: string, next: string) { const result = { ...value }; if (next.trim()) result[key] = next; else delete result[key]; return result; }
function structuredCorrectionSeed(record: Record<string, unknown>) { return { messages: objectArray(record.messages), tools: objectArray(record.tools ?? record.tool_definitions), expected_calls: objectArray(record.expected_calls ?? record.tool_calls), expected_results: objectArray(record.expected_results ?? record.tool_results) }; }
function newToolDefinition(): Record<string, unknown> { return { type: "function", function: { name: "", description: "", parameters: { type: "object", properties: {}, required: [] } } }; }
function toolDetails(tool: Record<string, unknown>): ToolDetails { const nested = asRecord(tool.function); const definition = Object.keys(nested).length ? nested : tool; const parameters = asRecord(definition.parameters); const properties = asRecord(parameters.properties); const required = new Set(Array.isArray(parameters.required) ? parameters.required.map(String) : []); return { name: String(definition.name ?? ""), description: String(definition.description ?? ""), parameters: Object.entries(properties).map(([name, schema]) => { const field = asRecord(schema); return { name, type: String(field.type ?? "string"), description: String(field.description ?? ""), required: required.has(name), schema: field }; }) }; }
function buildToolDefinition(original: Record<string, unknown>, details: ToolDetails): Record<string, unknown> { const nested = asRecord(original.function); const definition = Object.keys(nested).length ? nested : original; const existingParameters = asRecord(definition.parameters); const properties = Object.fromEntries(details.parameters.map((parameter) => [parameter.name, { ...parameter.schema, type: parameter.type, ...(parameter.description ? { description: parameter.description } : {}) }])); const parameters = { ...existingParameters, type: "object", properties, required: details.parameters.filter((parameter) => parameter.required).map((parameter) => parameter.name) }; const updatedDefinition = { ...definition, name: details.name, description: details.description, parameters }; return Object.keys(nested).length ? { ...original, type: String(original.type ?? "function"), function: updatedDefinition } : updatedDefinition; }
function setResultContent(result: Record<string, unknown>, content: string) { const key = "result" in result ? "result" : "output" in result ? "output" : "content"; return { ...result, [key]: content }; }
function formatGuidedValue(value: unknown) { if (typeof value === "string") return value; if (value === undefined || value === null) return ""; return String(value); }
function parseGuidedValue(value: string): unknown { const trimmed = value.trim(); if (trimmed === "true") return true; if (trimmed === "false") return false; if (trimmed === "null") return null; if (/^-?(?:\d+\.?\d*|\.\d+)$/.test(trimmed)) return Number(trimmed); return value; }
function firstString(value: Record<string, unknown>, keys: string[]) { for (const key of keys) if (typeof value[key] === "string" && value[key]) return String(value[key]); return null; }
function renderValue(value: unknown): string { if (typeof value === "string") return value; if (value === undefined || value === null) return "—"; try { return JSON.stringify(value, null, 2); } catch { return String(value); } }
function humanize(value: string) { return value.replace(/[_-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function buildModeLabel(value: string) { return ({ annotate: "Add review metadata", filter: "Keep accepted records", replace_by_record_id: "Replace matching records", append: "Append reviewed records" } as Record<string, string>)[value] ?? humanize(value); }
function statusVariant(status: string): "success" | "warning" | "danger" | "neutral" { if (["completed", "agreed", "labeled"].includes(status)) return "success"; if (["conflict", "needs_adjudication", "flagged"].includes(status)) return "warning"; if (["excluded", "failed"].includes(status)) return "danger"; return "neutral"; }
function eventKey(queueId: string, itemId: string, type: string) { return `ui-${queueId}-${itemId}-${type}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }
const textareaClass = "mt-1.5 w-full resize-y rounded-md border border-border bg-surface px-3 py-2 text-[11.5px] leading-5 text-fg outline-none placeholder:text-fg-disabled focus:border-accent focus:ring-2 focus:ring-accent/20";
const selectClass = "h-8 w-full rounded-md border border-border bg-bg px-2 text-[10px] text-fg outline-none focus:border-accent";
const guidedTextareaClass = "min-h-20 w-full resize-y rounded-md border border-border bg-bg px-3 py-2 text-[11px] leading-5 text-fg outline-none placeholder:text-fg-disabled focus:border-accent focus:ring-2 focus:ring-accent/20";
const guidedSelectClass = "min-h-11 w-full rounded-md border border-border bg-bg px-2.5 text-[10.5px] text-fg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20";
