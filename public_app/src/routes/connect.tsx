import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, ExternalLink, KeyRound, Plug, ShieldCheck, XCircle } from "lucide-react";
import { type FormEvent, useMemo, useState } from "react";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardEyebrow } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, connectionMode, getApiToken, setApiToken } from "@/lib/api";
import {
  queryKeys,
  useHuggingFaceCheckModel,
  useHuggingFaceClearToken,
  useHuggingFaceSaveToken,
  useHuggingFaceStatus,
} from "@/lib/hooks";

export const Route = createFileRoute("/connect")({
  validateSearch: (search): { from?: string; hfModel?: string; section?: string } => ({
    from: typeof search.from === "string" ? search.from : undefined,
    hfModel: typeof search.hfModel === "string" ? search.hfModel : undefined,
    section: typeof search.section === "string" ? search.section : undefined,
  }),
  component: ConnectRoute,
});

function ConnectRoute() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { from, hfModel } = Route.useSearch();
  const [token, setToken] = useState(getApiToken() ?? "");
  const [hfToken, setHfToken] = useState("");
  const [hfAccess, setHfAccess] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "checking" | "ok" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);
  const mode = connectionMode();
  const origin = useMemo(() => window.location.origin, []);
  const hfStatus = useHuggingFaceStatus();
  const hfSave = useHuggingFaceSaveToken();
  const hfClear = useHuggingFaceClearToken();
  const hfCheck = useHuggingFaceCheckModel();

  async function testConnection(nextToken: string | null) {
    setApiToken(nextToken);
    setStatus("checking");
    setMessage(null);
    try {
      await api.health();
      await queryClient.invalidateQueries({ queryKey: queryKeys.backend });
      await queryClient.invalidateQueries({ queryKey: queryKeys.dashboard });
      setStatus("ok");
      setMessage("Connection verified.");
      if (from && from !== "/connect") {
        navigate({ to: from });
      }
    } catch (e) {
      setStatus("error");
      setMessage(e instanceof Error ? e.message : "Connection failed.");
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const next = token.trim();
    testConnection(next ? next : null);
  }

  function onHfSubmit(e: FormEvent) {
    e.preventDefault();
    setHfAccess(null);
    hfSave.mutate(hfToken, {
      onSuccess: (next) => {
        setHfToken("");
        setHfAccess(next.message || "Hugging Face access saved.");
      },
      onError: (err) => setHfAccess(err.message),
    });
  }

  const hf = hfStatus.data;
  const hfTone = hf?.verified
    ? "success"
    : hf?.present
      ? "warning"
      : "neutral";
  const hfLabel = hf?.verified
    ? hf.username
      ? `Connected as ${hf.username}`
      : "Connected"
    : hf?.source === "env"
      ? "Using HF_TOKEN"
      : hf?.present
        ? "Needs attention"
        : "Not connected";
  const hfModelUrl = hfModel ? `https://huggingface.co/${hfModel}` : null;

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Remote connection"
        subtitle="Connect this browser to one Halo Forge workstation."
      />

      <div className="px-5 py-5 max-w-4xl space-y-4">
        <div className="grid gap-3 lg:grid-cols-3">
          <StatusTile
            label="Mode"
            value={mode === "remote" ? "Remote workstation" : "Local loopback"}
            tone={mode === "remote" ? "warning" : "success"}
          />
          <StatusTile label="Origin" value={origin} mono />
          <StatusTile
            label="Auth"
            value={token.trim() ? "Token stored" : mode === "remote" ? "Token needed" : "Optional"}
            tone={token.trim() ? "success" : mode === "remote" ? "danger" : "neutral"}
          />
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>REMOTE V1</CardEyebrow>
              <CardTitle>Bearer token</CardTitle>
            </div>
            <KeyRound className="h-4 w-4 text-fg-disabled" />
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={onSubmit}>
              <div className="space-y-2">
                <label className="text-[12px] font-medium text-fg-muted" htmlFor="api-token">
                  API token
                </label>
                <Input
                  id="api-token"
                  value={token}
                  type="password"
                  mono
                  autoComplete="off"
                  placeholder="hfk_..."
                  onChange={(e) => setToken(e.target.value)}
                />
                <p className="text-xs text-fg-muted">
                  Local loopback stays zero-config. Remote browser access uses the existing
                  Halo Forge bearer token store.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button type="submit" variant="primary" disabled={status === "checking"}>
                  <Plug />
                  {status === "checking" ? "Checking..." : "Save and test"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    setToken("");
                    testConnection(null);
                  }}
                >
                  Clear token
                </Button>
                <Button asChild type="button" variant="ghost">
                  <Link to="/">Back to overview</Link>
                </Button>
              </div>
              {message ? (
                <div className="flex items-center gap-2 text-[12px]">
                  {status === "ok" ? (
                    <CheckCircle2 className="h-4 w-4 text-success" />
                  ) : (
                    <XCircle className="h-4 w-4 text-danger" />
                  )}
                  <span className={status === "ok" ? "text-success" : "text-danger"}>
                    {message}
                  </span>
                </div>
              ) : null}
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>HUGGING FACE</CardEyebrow>
              <CardTitle>Gated model access</CardTitle>
              <Badge tone={hfTone} dot size="sm">{hfLabel}</Badge>
            </div>
            <KeyRound className="h-4 w-4 text-fg-disabled" />
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={onHfSubmit}>
              <div className="space-y-2">
                <label className="text-[12px] font-medium text-fg-muted" htmlFor="hf-token">
                  Hugging Face token
                </label>
                <Input
                  id="hf-token"
                  value={hfToken}
                  type="password"
                  mono
                  autoComplete="off"
                  placeholder="hf_..."
                  onChange={(e) => setHfToken(e.target.value)}
                />
                <p className="text-xs text-fg-muted">
                  Paste a read token once. Halo Forge stores it on this workstation for
                  serving and training downloads; the dashboard never displays the saved value.
                  Some gated models still require accepting the license on Hugging Face.
                </p>
              </div>
              <div className="rounded-md border border-border-subtle bg-bg-subtle/40 px-3 py-2 text-[12px] text-fg-muted">
                {hfStatus.isLoading ? "Checking Hugging Face access..." : hf?.message ?? "No Hugging Face token is configured."}
              </div>
              {hfModel ? (
                <div className="flex flex-wrap items-center gap-2 rounded-md border border-border-subtle bg-surface/70 px-3 py-2">
                  <span className="font-mono text-[12px] text-fg-muted">{hfModel}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    disabled={hfCheck.isPending}
                    onClick={() =>
                      hfCheck.mutate(hfModel, {
                        onSuccess: (result) => setHfAccess(result.message),
                        onError: (err) => setHfAccess(err.message),
                      })
                    }
                  >
                    Check access
                  </Button>
                  {hfModelUrl ? (
                    <Button asChild type="button" variant="ghost">
                      <a href={hfModelUrl} target="_blank" rel="noreferrer">
                        <ExternalLink />
                        Open model page
                      </a>
                    </Button>
                  ) : null}
                </div>
              ) : null}
              <div className="flex flex-wrap items-center gap-2">
                <Button type="submit" variant="primary" disabled={hfSave.isPending || !hfToken.trim()}>
                  <ShieldCheck />
                  {hfSave.isPending ? "Verifying..." : "Save and verify"}
                </Button>
                {hf?.can_clear ? (
                  <Button
                    type="button"
                    variant="ghost"
                    disabled={hfClear.isPending}
                    onClick={() =>
                      hfClear.mutate(undefined, {
                        onSuccess: (next) => setHfAccess(next.message),
                        onError: (err) => setHfAccess(err.message),
                      })
                    }
                  >
                    Clear stored token
                  </Button>
                ) : null}
                {hf?.source === "env" ? (
                  <span className="text-[12px] text-fg-muted">
                    Environment credentials are managed outside the app.
                  </span>
                ) : null}
              </div>
              {hfAccess ? (
                <div className="flex items-center gap-2 text-[12px] text-fg-muted">
                  <ShieldCheck className="h-4 w-4 text-accent" />
                  <span>{hfAccess}</span>
                </div>
              ) : null}
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>SETUP</CardEyebrow>
              <CardTitle>Remote workstation path</CardTitle>
            </div>
            <ShieldCheck className="h-4 w-4 text-fg-disabled" />
          </CardHeader>
          <CardContent className="grid gap-2 text-[13px] text-fg-muted md:grid-cols-2">
            <Step index="01" text="On the workstation, bind Halo Forge to a network interface." />
            <Step index="02" text="Create a dashboard token with `halo-forge token create dashboard`." />
            <Step index="03" text="Open this app from another machine on the same trusted network." />
            <Step index="04" text="Paste the token here, then launch and monitor runs against that workstation." />
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function StatusTile({
  label,
  value,
  tone = "neutral",
  mono,
}: {
  label: string;
  value: string;
  tone?: "success" | "warning" | "danger" | "neutral";
  mono?: boolean;
}) {
  return (
    <Card className="bg-surface/80">
      <CardContent className="px-3.5 py-3">
        <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled">
          {label}
        </div>
        <div className="mt-1.5 flex items-center gap-2">
          {tone !== "neutral" ? <Badge tone={tone} dot size="sm">{value}</Badge> : null}
          {tone === "neutral" ? (
            <span className={mono ? "font-mono text-[12px] text-fg" : "text-[13px] text-fg"}>
              {value}
            </span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function Step({ index, text }: { index: string; text: string }) {
  return (
    <div className="flex gap-3 rounded-md border border-border-subtle bg-bg-subtle/40 px-3 py-2.5">
      <span className="font-mono text-[11px] text-accent">{index}</span>
      <span>{text}</span>
    </div>
  );
}
