import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, ChevronDown, Cpu, HardDrive, Loader2, RefreshCw, Wrench } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

export const Route = createFileRoute("/setup")({ component: SetupRoute });

function SetupRoute() {
  const client = useQueryClient();
  const [confirmRuntime, setConfirmRuntime] = useState(false);
  const readiness = useQuery({
    queryKey: ["workstation-readiness"],
    queryFn: () => api.workstationReadiness(),
    refetchInterval: 20_000,
  });
  const release = useQuery({ queryKey: ["release-status"], queryFn: () => api.releaseStatus(), retry: false });
  const runtimes = useQuery({ queryKey: ["managed-runtimes"], queryFn: () => api.managedRuntimes(), retry: false, refetchInterval: 10_000 });
  const remediation = useMutation({
    mutationFn: (action: string) => api.applySetupRemediation(action),
    onSuccess: (value) => client.setQueryData(["workstation-readiness"], value),
  });
  const value = readiness.data;
  const automatic = value?.remediations.find((item) => item.automatic);
  const activeFamily = value?.capability.supported_backends.some((item) => item.startsWith("rocm"))
    ? "rocm"
    : value?.capability.supported_backends.includes("cuda") ? "cuda" : null;
  const runtime = runtimes.data?.items.find((item) => item.accelerator_family === activeFamily);
  const runtimeReady = runtime?.qualification && ["vendor_supported", "local_verified"].includes(runtime.qualification.status);
  const paths = useQuery({
    queryKey: ["training-paths", activeFamily],
    queryFn: () => api.trainingPaths(activeFamily as "rocm" | "cuda"),
    enabled: Boolean(activeFamily),
    retry: false,
    refetchInterval: 10_000,
  });
  const recommendedPath = paths.data?.paths.find(
    (item) => item.path_revision_id === paths.data?.recommended_path_revision_id,
  );
  const pathVerified = recommendedPath?.state === "path_verified";
  const runtimeNeedsSetup = Boolean(activeFamily && runtime?.revision && !runtimeReady);
  const runtimePreparation = runtime?.preparations.find((item) => ["queued", "running"].includes(item.status));
  const prepareRuntime = useMutation({
    mutationFn: () => api.prepareManagedRuntime(runtime!.revision!.id),
    onSuccess: () => {
      setConfirmRuntime(false);
      void runtimes.refetch();
      void readiness.refetch();
    },
  });
  const certifyPath = useMutation({
    mutationFn: () => api.certifyTrainingPath(recommendedPath!.path_revision_id, runtime!.revision!.id),
    onSuccess: () => {
      void paths.refetch();
      void readiness.refetch();
    },
  });

  return (
    <>
      <Topbar
        eyebrow="Workstation"
        title="Setup"
        subtitle="A focused check of what this computer can run right now."
        actions={
          <Button variant="ghost" size="sm" onClick={() => readiness.refetch()} disabled={readiness.isFetching}>
            <RefreshCw className={readiness.isFetching ? "animate-spin" : undefined} /> Check again
          </Button>
        }
      />
      <main className="mx-auto max-w-4xl space-y-5 px-5 py-6">
        {readiness.isLoading ? (
          <Card><CardContent className="flex items-center gap-2 py-8 text-sm text-fg-muted"><Loader2 className="animate-spin" /> Checking this workstation…</CardContent></Card>
        ) : readiness.isError || !value ? (
          <Card><CardContent className="space-y-3 py-6"><p className="text-sm text-danger">Halo Forge could not complete the setup check.</p><Button asChild variant="primary"><Link to="/diagnostics">Open diagnostics</Link></Button></CardContent></Card>
        ) : (
          <>
            <Card className={value.status === "blocked" ? "border-warning/50" : "border-success/40"}>
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <Badge tone={value.status === "blocked" ? "warning" : "success"}>{value.display_status}</Badge>
                    <CardTitle className="mt-3 text-xl">{value.summary}</CardTitle>
                  </div>
                  <span className="font-mono text-[11px] text-fg-disabled">{value.capability.platform} · {value.capability.architecture}</span>
                </div>
              </CardHeader>
              {runtimeNeedsSetup && confirmRuntime ? (
                <CardContent className="border-t border-border-subtle py-4">
                  <div className="max-w-2xl border-l-2 border-accent pl-4">
                    <p className="text-sm font-medium text-fg">Prepare a verified {activeFamily === "rocm" ? "AMD" : "NVIDIA"} training runtime</p>
                    <p className="mt-1 text-xs leading-5 text-fg-muted">Halo Forge will prepare the pinned runtime, verify core accelerator operations, then queue a real instruction-training check through Dataset Lab and the shipped trainer. It will not stop other GPU work.</p>
                    <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-fg-muted">
                      {runtime?.revision?.download_bytes ? <span className="inline-flex items-center gap-1.5"><HardDrive className="h-3.5 w-3.5" />{formatBytes(runtime.revision.download_bytes)} download</span> : null}
                      {runtime?.revision?.installed_bytes ? <span className="inline-flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" />{formatBytes(runtime.revision.installed_bytes)} installed</span> : null}
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      <Button variant="primary" onClick={() => prepareRuntime.mutate()} disabled={prepareRuntime.isPending}>
                        {prepareRuntime.isPending ? <Loader2 className="animate-spin" /> : <Wrench />}Download and verify
                      </Button>
                      <Button variant="ghost" onClick={() => setConfirmRuntime(false)} disabled={prepareRuntime.isPending}>Not now</Button>
                    </div>
                    {prepareRuntime.isError ? <p role="alert" className="mt-3 text-xs text-danger">{(prepareRuntime.error as Error).message}</p> : null}
                  </div>
                </CardContent>
              ) : null}
              <CardContent className="flex flex-wrap gap-2">
                {runtimeNeedsSetup ? (
                  <Button variant="primary" onClick={() => setConfirmRuntime(true)} disabled={Boolean(runtimePreparation)}>
                    {runtimePreparation ? <Loader2 className="animate-spin" /> : <Cpu />}
                    {runtimePreparation ? "Preparing in Activity" : `Prepare ${activeFamily === "rocm" ? "AMD" : "NVIDIA"} training`}
                  </Button>
                ) : runtimeReady && recommendedPath && !pathVerified ? (
                  <Button variant="primary" onClick={() => certifyPath.mutate()} disabled={certifyPath.isPending || recommendedPath.state === "verification_in_progress"}>
                    {certifyPath.isPending || recommendedPath.state === "verification_in_progress" ? <Loader2 className="animate-spin" /> : <Cpu />}
                    {recommendedPath.state === "verification_in_progress" ? "Verifying in Activity" : "Verify text training"}
                  </Button>
                ) : automatic ? (
                  <Button variant="primary" onClick={() => remediation.mutate(automatic.action)} disabled={remediation.isPending}>
                    {remediation.isPending ? <Loader2 className="animate-spin" /> : <Wrench />}{automatic.label}
                  </Button>
                ) : (
                  <Button asChild variant="primary"><Link to="/datasets/new" search={{ example: "1" }}>Try a working example</Link></Button>
                )}
                <Button asChild variant="secondary"><Link to="/datasets/new" search={{ example: undefined }}>Train on your data</Link></Button>
                {value.status === "blocked" ? <Button asChild variant="ghost"><Link to="/diagnostics">Create support bundle</Link></Button> : null}
              </CardContent>
            </Card>

            {runtimeReady && recommendedPath ? (
              <section aria-labelledby="training-path-heading" className="border-l-2 border-accent px-4 py-3">
                <h2 id="training-path-heading" className="text-sm font-medium text-fg">{recommendedPath.label}</h2>
                <p className="mt-1 text-xs leading-5 text-fg-muted">{recommendedPath.summary}</p>
                <Badge className="mt-2" tone={pathVerified ? "success" : "warning"}>{recommendedPath.display_status}</Badge>
              </section>
            ) : null}

            <section aria-labelledby="checks-heading">
              <h2 id="checks-heading" className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-muted">Readiness checks</h2>
              <div className="divide-y divide-border-subtle border border-border-subtle bg-surface/30">
                {value.checks.map((check) => (
                  <div key={check.id} className="flex items-start gap-3 px-4 py-3">
                    {check.status === "blocked" ? <AlertTriangle className="mt-0.5 h-4 w-4 text-warning" /> : <CheckCircle2 className="mt-0.5 h-4 w-4 text-success" />}
                    <div className="min-w-0 flex-1"><p className="text-sm font-medium text-fg">{check.label}</p><p className="mt-0.5 text-xs text-fg-muted">{check.summary}</p></div>
                    <Badge tone={check.status === "blocked" ? "warning" : "neutral"}>{check.status === "attention" ? "Optional" : check.status === "blocked" ? "Needs action" : "Ready"}</Badge>
                  </div>
                ))}
              </div>
            </section>

            {release.data?.update_available ? <Card><CardContent className="flex flex-wrap items-center justify-between gap-3 py-4"><div><p className="text-sm font-medium text-fg">Update available</p><p className="mt-0.5 text-xs text-fg-muted">{release.data.message} Halo Forge will not install it automatically.</p></div>{release.data.release_url ? <Button asChild variant="secondary"><a href={release.data.release_url} target="_blank" rel="noreferrer">Review release</a></Button> : null}</CardContent></Card> : null}

            <Collapsible>
              <CollapsibleTrigger asChild><Button variant="ghost" size="sm"><ChevronDown /> Technical details</Button></CollapsibleTrigger>
              <CollapsibleContent className="mt-2 rounded border border-border-subtle bg-bg-subtle p-4 font-mono text-[11px] text-fg-muted">
                <p className="mb-3 font-sans leading-5">A generic tensor update is a runtime diagnostic only. Guided training unlocks after the real Dataset Lab renderer, shipped trainer, parameter-delta check, and artifact reload all pass.</p>
                <pre className="whitespace-pre-wrap">{JSON.stringify({ capability: value.capability, checks: value.checks.map(({ technical, ...check }) => ({ ...check, technical })) }, null, 2)}</pre>
              </CollapsibleContent>
            </Collapsible>
          </>
        )}
      </main>
    </>
  );
}

function formatBytes(value: number): string {
  const gib = value / 1024 ** 3;
  return `${gib >= 10 ? gib.toFixed(0) : gib.toFixed(1)} GB`;
}
