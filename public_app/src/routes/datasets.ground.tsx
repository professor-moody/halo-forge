import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, BookOpen, CheckCircle2, Loader2, Play, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { Topbar } from "@/components/shell";
import { DataSectionTabs } from "@/components/data/data-section-tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";

export const Route = createFileRoute("/datasets/ground")({
  validateSearch: (search: Record<string, unknown>): { sourceVersion?: string } => ({
    sourceVersion: typeof search.sourceVersion === "string" ? search.sourceVersion : undefined,
  }),
  component: GroundedDataRoute,
});

function GroundedDataRoute() {
  const navigate = useNavigate();
  const sourceVersion = Route.useSearch().sourceVersion ?? "";
  const [task, setTask] = useState("qa");
  const [destination, setDestination] = useState("training");
  const [preset, setPreset] = useState<"quick" | "standard" | "thorough">("standard");
  const [teacherModel, setTeacherModel] = useState("local-default");
  const source = useQuery({
    queryKey: ["dataset-versions", sourceVersion],
    queryFn: () => api.datasetVersion(sourceVersion),
    enabled: Boolean(sourceVersion),
    retry: false,
  });
  const preview = useMutation({
    mutationFn: async () => {
      const profile = await api.createGroundingProfile({
        name: `${source.data?.id || "Corpus"} · grounded ${task}`,
        description: "Reviewed grounded generation profile",
      });
      const profileId = String(profile.id || "");
      const revision = await api.createGroundingProfileRevision(profileId, {
        source_version_id: sourceVersion,
        task_type: task,
        output_adapter: task === "benchmark" ? "benchmark_item" : task === "preference" ? "preference" : "grounded_sft",
        teacher: {
          endpoint_type: teacherModel === "local-default" ? "local" : "configured",
          model: teacherModel,
        },
        prompt_template_hash: "guided-grounding-v1",
        chunk_selection: { strategy: "coverage", seed: 42 },
        quota: { quick: 50, standard: 250, thorough: 1000 }[preset],
        seed: 42,
        citation_contract: { required: true, exact_span: true },
        intended_destination: destination,
      });
      const revisionId = String(revision.id || "");
      const result = await api.previewGroundedBatch(revisionId, {
        source_version_id: sourceVersion,
        preset,
      });
      return { revisionId, result };
    },
  });
  const launch = useMutation({
    mutationFn: () => api.launchGroundedBatch(preview.data!.revisionId, {
      source_version_id: sourceVersion,
      preset,
      hosted_provider_confirmed: preview.data?.result.request_estimate.hosted_provider === true,
    }),
  });
  const batch = useQuery({
    queryKey: ["grounding-batches", launch.data?.id],
    queryFn: () => api.groundedBatch(launch.data!.id),
    enabled: Boolean(launch.data?.id),
    retry: false,
    refetchInterval: (query) => query.state.data && ["queued", "running"].includes(query.state.data.status) ? 1_500 : false,
  });
  const completedBatch = batch.data?.status === "completed" ? batch.data : null;
  const reviewProposal = useMutation({
    mutationFn: () => api.createGroundingReviewProposal(completedBatch!.id),
    onSuccess: (proposal) => navigate({
      to: "/datasets/review",
      search: {
        new: "1",
        source: "jsonl",
        sourceRef: String(proposal.source_ref || ""),
        baseRef: undefined,
      },
    }),
  });
  return (
    <>
      <Topbar
        eyebrow="Data · Grounded generation"
        title="Create examples from documents"
        subtitle="Preview cited examples first, then generate a reviewed batch in the background."
        actions={<Button variant="ghost" size="sm" asChild><Link to="/datasets"><ArrowLeft />Data</Link></Button>}
      />
      <DataSectionTabs />
      <div className="grid min-h-[calc(100vh-152px)] lg:grid-cols-[minmax(0,1fr)_320px]">
        <main className="px-5 py-6">
          <div className="mx-auto max-w-3xl">
            <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-accent">Source → Task → Coverage → Teacher & verifier → Preview → Generate → Review</div>
            <h2 className="mt-2 text-xl font-medium text-fg">Create cited examples you can review</h2>
            <p className="mt-2 text-[11px] leading-5 text-fg-muted">Halo Forge proposes examples and checks their citations. They do not become training data until you review and publish them.</p>
            {!sourceVersion ? <div className="mt-8 border border-border-subtle bg-bg-subtle/40 px-5 py-8 text-center"><BookOpen className="mx-auto h-6 w-6 text-fg-disabled" /><div className="mt-3 text-sm text-fg">Choose an immutable corpus version</div><p className="mt-1 text-[11px] text-fg-muted">Open a corpus Dataset Version and choose Create examples from documents.</p><Button className="mt-4" variant="secondary" asChild><Link to="/datasets">Browse datasets</Link></Button></div> : <div className="mt-7 grid gap-5"><section className="border-y border-border-subtle"><Readout label="Source" value={source.isLoading ? "Loading corpus" : source.data ? `Immutable version · ${source.data.row_count ?? 0} records` : "Corpus version unavailable"} /><Readout label="Safety" value="Development sources only; protected evidence is refused" /><Readout label="Citations" value="Exact source spans are checked" /></section><div className="grid gap-4 sm:grid-cols-2"><Field label="What should Halo Forge create?"><select value={task} onChange={(event) => { setTask(event.target.value); preview.reset(); launch.reset(); }} className="h-9 w-full rounded-md border border-border bg-surface px-3 text-xs text-fg"><option value="qa">Cited questions and answers</option><option value="instruction">Instruction examples</option><option value="extraction">Extraction examples</option><option value="reasoning">Reasoning examples</option><option value="preference">Preference examples</option><option value="benchmark">Evaluation questions</option></select></Field><Field label="How many suggestions?"><select value={preset} onChange={(event) => { setPreset(event.target.value as typeof preset); preview.reset(); launch.reset(); }} className="h-9 w-full rounded-md border border-border bg-surface px-3 text-xs text-fg"><option value="quick">Quick — 50</option><option value="standard">Standard — 250</option><option value="thorough">Thorough — 1,000</option></select></Field></div>{preview.data ? <section className="border border-border-subtle"><div className="border-b border-border-subtle px-4 py-3"><div className="text-xs font-medium text-fg">Ten examples to check before generation</div><div className="mt-1 text-[10.5px] text-fg-muted">These are deterministic previews from the selected corpus and task.</div></div><div className="divide-y divide-border-subtle">{preview.data.result.preview_items.slice(0, 10).map((item, index) => <div key={index} className="px-4 py-3"><div className="text-[10px] font-medium text-fg">Example {index + 1}</div><div className="mt-1 line-clamp-3 text-[10.5px] leading-5 text-fg-muted">{String(item.question || item.instruction || item.prompt || item.answer || item.response || "Grounded example")}</div></div>)}</div></section> : null}{preview.isError || launch.isError ? <div className="border-l-2 border-danger bg-danger-bg px-4 py-3 text-[11px] text-fg-muted">{String((preview.error || launch.error) instanceof Error ? (preview.error || launch.error)?.message : "The request could not be prepared.")}</div> : null}{launch.data && !completedBatch ? <div className="border-l-2 border-accent bg-accent-bg px-4 py-3"><div className="flex items-center gap-2 text-xs font-medium text-fg"><Loader2 className="animate-spin" />Creating and checking cited examples</div><div className="mt-1 text-[11px] text-fg-muted">You can leave this screen. Progress and retry actions are available in Activity.</div></div> : null}{completedBatch ? <div className="border-l-2 border-success bg-success-bg px-4 py-3"><div className="flex items-center gap-2 text-xs font-medium text-success"><CheckCircle2 />Examples are ready to review</div><div className="mt-1 text-[11px] text-fg-muted">{completedBatch.accepted_count} cited suggestions passed structural checks. Human review is still required.</div><div className="mt-3"><Button size="sm" variant="primary" disabled={reviewProposal.isPending} onClick={() => reviewProposal.mutate()}>{reviewProposal.isPending ? <Loader2 className="animate-spin" /> : null}Review examples</Button></div></div> : null}<details className="border-t border-border-subtle pt-3"><summary className="cursor-pointer text-[10px] font-medium text-fg-subtle">Advanced</summary><div className="mt-3 grid gap-4 sm:grid-cols-2"><Field label="Destination"><select value={destination} onChange={(event) => setDestination(event.target.value)} className="h-9 w-full rounded-md border border-border bg-surface px-3 text-xs text-fg"><option value="training">Training dataset</option><option value="development_evaluation">Development evaluation</option></select></Field><Field label="Teacher model"><Input value={teacherModel} onChange={(event) => setTeacherModel(event.target.value)} placeholder="Local or configured model" /></Field></div></details></div>}
          </div>
        </main>
        <aside className="border-t border-border-subtle bg-bg-subtle/30 px-5 py-6 lg:border-l lg:border-t-0"><div className="text-[10px] font-medium uppercase tracking-[0.12em] text-fg-disabled">Recommended next action</div><div className="mt-4 space-y-4 text-[10.5px] leading-5 text-fg-muted"><p><ShieldCheck className="mr-2 inline h-3.5 w-3.5 text-accent" />Halo Forge checks changed, missing, cross-document, and out-of-range citations.</p><p>Generation creates suggestions only. Review, publication, dataset building, and training stay separate.</p></div>{!preview.data ? <Button className="mt-6 w-full" size="lg" disabled={!source.data || preview.isPending} onClick={() => preview.mutate()}>{preview.isPending ? <Loader2 className="animate-spin" /> : <Play />}Preview 10 examples</Button> : !launch.data ? <Button className="mt-6 w-full" size="lg" disabled={launch.isPending || preview.data.result.blockers.length > 0} onClick={() => launch.mutate()}>{launch.isPending ? <Loader2 className="animate-spin" /> : <Play />}Generate {preview.data.result.candidate_limit} suggestions</Button> : completedBatch ? <Button className="mt-6 w-full" size="lg" disabled={reviewProposal.isPending} onClick={() => reviewProposal.mutate()}>Review examples</Button> : <Button className="mt-6 w-full" size="lg" variant="secondary" disabled><Loader2 className="animate-spin" />Working in Activity</Button>}</aside>
      </div>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="space-y-1.5"><Label className="text-[10.5px] uppercase tracking-wider text-fg-subtle">{label}</Label>{children}</div>;
}

function Readout({ label, value }: { label: string; value: string }) {
  return <div className="grid gap-1 border-b border-border-subtle px-3 py-3 last:border-b-0 sm:grid-cols-[140px_minmax(0,1fr)]"><div className="text-[10px] uppercase tracking-wider text-fg-disabled">{label}</div><div className="text-[11px] text-fg-muted">{value}</div></div>;
}
