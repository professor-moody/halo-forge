import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowLeft,
  ExternalLink,
  Loader2,
  Send,
  Server,
  Settings,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type ModelCatalogEntry, type PlaygroundChatRequest, type PlaygroundMessage, type ServeStatus } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardEyebrow, CardHeader, CardTitle } from "@/components/ui/card";
import { queryKeys, useModelCatalog, useServeLogs, useServeStart, useServeStatus, useServeStop } from "@/lib/hooks";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/playground")({
  component: PlaygroundRoute,
});

/**
 * Inference playground (Track F-S).
 *
 * Chat UI that forwards through `/api/public/playground/chat` to a
 * `halo-forge serve`-style endpoint. Closes the train→ship loop in
 * the UI: train a model, serve it (`halo-forge serve --model ...`),
 * then talk to it without leaving the dashboard.
 *
 * Defaults to the local serve endpoint at 127.0.0.1:8001/v1, but the
 * settings panel lets you point at any OpenAI-compatible URL — hosted
 * APIs, remote serves, custom inference stacks. Auth still goes
 * through the public API token, so a non-loopback dashboard talking
 * to a hosted teacher stays single-auth-domain.
 */

type ChatMessage = PlaygroundMessage & {
  id: string;
  kind?: "normal" | "error";
  errorKind?: string;
  action?: string;
  modelId?: string;
  modelUrl?: string;
};

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful assistant. Respond concisely and accurately.";
const DEFAULT_SERVE_URL = "http://127.0.0.1:8001/v1";
const FALLBACK_SAFE_SERVE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16";

type QuickServeModel = {
  model: string;
  label: string;
  backend?: string;
  trustRemoteCode?: boolean;
};

function pickQuickServeModel(items: ModelCatalogEntry[]): QuickServeModel {
  const safePick =
    items.find((item) => item.id === FALLBACK_SAFE_SERVE_MODEL) ??
    items.find((item) => item.recommended_first_run && item.risk_level === "safe" && (item.backend_support ?? []).includes("mlx")) ??
    items.find((item) => item.recommended_first_run && item.risk_level === "safe") ??
    null;
  const model = safePick?.mlx_variant ?? safePick?.id ?? FALLBACK_SAFE_SERVE_MODEL;
  return {
    model,
    label: shortModelName(model),
    backend: model.startsWith("mlx-community/") || safePick?.mlx_variant ? "mlx" : undefined,
    trustRemoteCode: safePick?.trust_remote_code_required,
  };
}

function shortModelName(model: string): string {
  return model.replace(/^mlx-community\//, "").replace(/^Qwen\//, "").replace(/-Instruct-bf16$/, "");
}

function PlaygroundRoute() {
  const queryClient = useQueryClient();
  const serveStatus = useServeStatus();
  const serveStart = useServeStart();
  const serveStop = useServeStop();
  const serveLogs = useServeLogs(80, Boolean(serveStatus.data?.logs_available && serveStatus.data?.state !== "idle"));
  const quickModels = useModelCatalog({ backend: "mlx" });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [model, setModel] = useState("halo-forge");
  const [serveUrl, setServeUrl] = useState(DEFAULT_SERVE_URL);
  const [apiKey, setApiKey] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [showSettings, setShowSettings] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const serveState = serveStatus.data?.state ?? "idle";
  const localReady = Boolean(serveStatus.data?.running && serveState === "running");
  const managedUrl = serveStatus.data?.url ?? DEFAULT_SERVE_URL;
  const usingManagedEndpoint = serveUrl === DEFAULT_SERVE_URL || serveUrl === managedUrl;
  const externalEndpoint = Boolean(serveUrl.trim() && !usingManagedEndpoint);
  const canChat = localReady || externalEndpoint;
  const activeChatModel = canChat ? model : "No model serving";
  const activeChatUrl = canChat ? serveUrl : "Start a model to chat";
  const quickServeModel = useMemo(
    () => pickQuickServeModel(quickModels.data?.items ?? []),
    [quickModels.data?.items],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!serveStatus.data?.running || !serveStatus.data.url) return;
    if (serveUrl === DEFAULT_SERVE_URL || serveUrl === serveStatus.data.url) {
      setServeUrl(serveStatus.data.url);
      setModel(serveStatus.data.model ?? "halo-forge");
    }
  }, [serveStatus.data?.running, serveStatus.data?.url, serveStatus.data?.model, serveUrl]);

  useEffect(() => {
    if (!serveStatus.data || serveStatus.data.running || !usingManagedEndpoint) return;
    setServeUrl(DEFAULT_SERVE_URL);
    setModel("halo-forge");
  }, [serveStatus.data, usingManagedEndpoint]);

  const chatMutation = useMutation({
    mutationFn: (req: PlaygroundChatRequest) => api.playgroundChat(req),
  });

  async function send() {
    const text = input.trim();
    if (!text || chatMutation.isPending) return;
    if (!canChat) {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-e`,
          role: "assistant",
          kind: "error",
          content: "Start a local model first, or open Settings and enter an external endpoint.",
        },
      ]);
      return;
    }

    const userMsg: ChatMessage = {
      id: `${Date.now()}-u`,
      role: "user",
      content: text,
    };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput("");

    const wireMessages: PlaygroundMessage[] = [
      ...(systemPrompt.trim()
        ? [{ role: "system" as const, content: systemPrompt.trim() }]
        : []),
      ...nextMessages.map(({ role, content }) => ({ role, content })),
    ];

    try {
      const resp = await chatMutation.mutateAsync({
        messages: wireMessages,
        model,
        max_tokens: maxTokens,
        temperature,
        top_p: topP,
        serve_url: serveUrl || undefined,
        api_key: externalEndpoint ? apiKey || undefined : undefined,
      });

      if (resp.upstream_error) {
        setMessages((prev) => [
          ...prev,
          {
            id: `${Date.now()}-e`,
            role: "assistant",
            kind: "error",
            content: formatUpstreamError(resp),
            errorKind: resp.error_kind,
            action: resp.action,
            modelId: resp.model_id ?? model,
            modelUrl: resp.model_url,
          },
        ]);
        return;
      }

      const assistantContent =
        resp.choices?.[0]?.message?.content ?? "(no response)";
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-a`,
          role: "assistant",
          content: assistantContent,
        },
      ]);
    } catch (exc) {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-e`,
          role: "assistant",
          kind: "error",
          content: friendlyChatFailure(exc),
        },
      ]);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Cmd/Ctrl+Enter sends. Plain Enter inserts a newline so users can
    // compose multi-line prompts without losing the draft.
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      send();
    }
  }

function clearChat() {
    setMessages([]);
}

function formatUpstreamError(resp: { status?: number; message?: string; error_kind?: string; detail?: unknown }): string {
  if (resp.message) return resp.message;
  if (resp.error_kind === "gated_model") {
    return "This model requires Hugging Face access. Connect Hugging Face, accept the license, or choose an open model.";
  }
  if (resp.detail && typeof resp.detail === "object" && "detail" in resp.detail) {
    const detail = (resp.detail as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      return String((detail as { message?: unknown }).message);
    }
  }
  return `The model server returned an error${resp.status ? ` (${resp.status})` : ""}. Check the serve logs for details.`;
}

function friendlyChatFailure(exc: unknown): string {
  const message = exc instanceof Error ? exc.message : String(exc ?? "");
  if (/failed to fetch|network|ECONNREFUSED|timeout/i.test(message)) {
    return "The dashboard could not reach the model server. Check local serving status and logs, then try again.";
  }
  return "The chat request failed. Check local serving status and logs, then try again.";
}

  function startQuickModel() {
    if (!quickServeModel) return;
    serveStart.mutate(
      {
        model: quickServeModel.model,
        backend: quickServeModel.backend,
        trust_remote_code: Boolean(quickServeModel.trustRemoteCode),
      },
      {
        onSuccess: (status) => {
          setMessages([]);
          if (status.url) setServeUrl(status.url);
          if (status.model) setModel(status.model);
        },
        onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.serve }),
      },
    );
  }

  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Playground"
        subtitle={
          messages.length === 0
            ? "Chat with any halo-forge serve endpoint."
            : `${messages.length} message${messages.length === 1 ? "" : "s"} in this session`
        }
        actions={
          <>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSettings((v) => !v)}
              aria-pressed={showSettings}
              title="Toggle settings panel"
            >
              <Settings />
              {showSettings ? "Hide settings" : "Settings"}
            </Button>
            {messages.length ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearChat}
                title="Clear chat history"
              >
                <Trash2 />
                Clear
              </Button>
            ) : null}
            <Button variant="ghost" size="icon" asChild aria-label="Back to runs">
              <Link to="/runs">
                <ArrowLeft />
              </Link>
            </Button>
          </>
        }
      />
      <div className="px-5 py-5 space-y-4">
        {showSettings ? (
          <SettingsPanel
            systemPrompt={systemPrompt} setSystemPrompt={setSystemPrompt}
            model={model} setModel={setModel}
            serveUrl={serveUrl} setServeUrl={setServeUrl}
            apiKey={apiKey} setApiKey={setApiKey}
            managedLocal={usingManagedEndpoint}
            maxTokens={maxTokens} setMaxTokens={setMaxTokens}
            temperature={temperature} setTemperature={setTemperature}
            topP={topP} setTopP={setTopP}
          />
        ) : null}

        <ServeStatusPanel
          status={serveStatus.data ?? null}
          state={serveStatus.data?.state ?? "idle"}
          model={serveStatus.data?.model ?? null}
          url={serveStatus.data?.url ?? DEFAULT_SERVE_URL}
          message={serveStatus.data?.message ?? null}
          logsAvailable={Boolean(serveStatus.data?.logs_available)}
          logPath={serveStatus.data?.log_path ?? null}
          logLines={serveLogs.data?.lines ?? []}
          loading={serveStatus.isLoading}
          quickModelLabel={quickServeModel?.label ?? null}
          quickModelLoading={quickModels.isLoading}
          quickModelStarting={serveStart.isPending}
          quickModelError={serveStart.error?.message ?? null}
          stopping={serveStop.isPending}
          onStartQuick={startQuickModel}
          onStop={() =>
            serveStop.mutate(undefined, {
              onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.serve }),
            })
          }
        />

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardEyebrow>CHAT</CardEyebrow>
              <CardTitle>{activeChatModel}</CardTitle>
              <span className="font-mono text-[10px] text-fg-disabled truncate ml-1">
                {activeChatUrl.replace(/^https?:\/\//, "")}
              </span>
            </div>
            <Badge tone={canChat ? "neutral" : "warning"} size="sm">
              {canChat ? `t=${temperature.toFixed(2)} · top_p=${topP.toFixed(2)} · max=${maxTokens}` : "start a model first"}
            </Badge>
          </CardHeader>
          <CardContent className="p-0">
            <div className="px-4 py-4 space-y-3 min-h-[280px] max-h-[60vh] overflow-y-auto">
              {messages.length === 0 ? (
                <div className="text-center text-sm text-fg-muted py-10 max-w-[44ch] mx-auto">
                  <p>{canChat ? "Send a message to the active model." : "Start a local model to unlock chat."}</p>
                  <p className="text-[11px] text-fg-disabled mt-3">
                    {canChat
                      ? "Cmd/Ctrl+Enter sends. Settings can point this chat at an external endpoint."
                      : "Use the safe-model button above, or choose a model from Models or Results."}
                  </p>
                </div>
              ) : (
                messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)
              )}
              {chatMutation.isPending ? (
                <div className="flex items-center gap-2 text-sm text-fg-muted">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Generating…
                </div>
              ) : null}
              <div ref={messagesEndRef} />
            </div>

            <div className="border-t border-border-subtle p-3 flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={canChat ? "Send a message…" : "Start a model before chatting…"}
                disabled={!canChat}
                rows={3}
                className="flex-1 bg-bg border border-border-subtle rounded-md px-2.5 py-2 text-[13px] focus:outline-none focus:border-accent resize-y disabled:opacity-60 disabled:cursor-not-allowed"
              />
              <Button
                variant="primary"
                size="sm"
                onClick={send}
                disabled={!canChat || !input.trim() || chatMutation.isPending}
                title="Send (Cmd/Ctrl+Enter)"
              >
                {chatMutation.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Send className="h-3 w-3" />
                )}
                Send
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function ServeStatusPanel({
  status,
  state,
  model,
  url,
  message,
  logsAvailable,
  logPath,
  logLines,
  loading,
  quickModelLabel,
  quickModelLoading,
  quickModelStarting,
  quickModelError,
  stopping,
  onStartQuick,
  onStop,
}: {
  status: ServeStatus | null;
  state: string;
  model: string | null;
  url: string;
  message: string | null;
  logsAvailable: boolean;
  logPath: string | null;
  logLines: string[];
  loading: boolean;
  quickModelLabel: string | null;
  quickModelLoading: boolean;
  quickModelStarting: boolean;
  quickModelError: string | null;
  stopping: boolean;
  onStartQuick: () => void;
  onStop: () => void;
}) {
  const running = state === "running" || state === "starting" || state === "unhealthy";
  const showLogContext = state !== "idle";
  const latestLogs = showLogContext ? logLines.filter(Boolean).slice(-3) : [];
  const loadError = status?.load_error ?? null;
  const gatedLoadError = loadError?.action === "connect_huggingface" || loadError?.error_kind === "gated_model";
  const modelId = loadError?.model_id ?? loadError?.model ?? model;
  const modelUrl = loadError?.model_url ?? (modelId ? `https://huggingface.co/${modelId}` : undefined);
  const tone = state === "running" ? "success" : state === "starting" || state === "stopping" ? "warning" : state === "unhealthy" || state === "exited" ? "danger" : "neutral";
  const label =
    state === "running"
      ? status?.model_ready
        ? "ready"
        : "server ready"
      : state === "starting"
        ? "starting"
        : state === "stopping"
          ? "stopping"
          : state === "unhealthy"
            ? "unhealthy"
            : state === "exited"
              ? "exited"
              : "idle";
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 px-4 py-3 md:flex-row md:items-center md:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border-subtle bg-surface">
            <Server className="h-4 w-4 text-accent" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13px] font-medium text-fg">Local serving</span>
              {loading ? (
                <Badge tone="neutral" size="sm">checking</Badge>
              ) : (
                <Badge tone={tone} dot={state !== "idle"} size="sm">
                  {label}
                </Badge>
              )}
            </div>
            <div className="mt-0.5 truncate font-mono text-[11px] text-fg-subtle">
              {running ? `${model ?? "model"} · ${url}` : "Start serving from Models or completed Results."}
            </div>
            <div className="mt-1 text-[11px] text-fg-muted">
              {servingStateCopy(state, message)}
              {showLogContext && logsAvailable && logPath ? (
                <span className="font-mono text-fg-subtle"> · logs: {logPath}</span>
              ) : null}
            </div>
            {gatedLoadError ? (
              <div className="mt-2 flex flex-wrap gap-2">
                <Button asChild size="sm" variant="ghost">
                  <Link to="/connect" search={{ section: "huggingface", hfModel: modelId ?? undefined, from: "/playground" }}>
                    Connect Hugging Face
                  </Link>
                </Button>
                <Button asChild size="sm" variant="ghost">
                  <Link to="/models">Choose open model</Link>
                </Button>
                {modelUrl ? (
                  <Button asChild size="sm" variant="ghost">
                    <a href={modelUrl} target="_blank" rel="noreferrer">
                      <ExternalLink className="h-3.5 w-3.5" />
                      Open model page
                    </a>
                  </Button>
                ) : null}
              </div>
            ) : null}
            {quickModelError ? (
              <div className="mt-1 rounded-sm border border-danger/30 bg-danger-bg px-2 py-1 text-[11px] text-danger">
                {quickModelError}
              </div>
            ) : null}
            {latestLogs.length ? (
              <div className="mt-1 max-w-[86ch] space-y-0.5">
                {latestLogs.map((line, index) => (
                  <div key={`${index}-${line}`} className="truncate font-mono text-[11px] text-fg-disabled" title={line}>
                    {line}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {!running ? (
            <Button
              variant="primary"
              size="sm"
              onClick={onStartQuick}
              disabled={!quickModelLabel || quickModelLoading || quickModelStarting}
              title={quickModelLabel ? `Start ${quickModelLabel}` : "Loading safe model picks"}
            >
              {quickModelStarting || quickModelLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Server className="h-3.5 w-3.5" />}
              {quickModelLabel ? `Start ${quickModelLabel}` : "Start safe model"}
            </Button>
          ) : null}
          <Button asChild variant="ghost" size="sm">
            <Link to="/models">Choose model</Link>
          </Button>
          {running ? (
            <Button variant="ghost" size="sm" onClick={onStop} disabled={stopping}>
              {stopping ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5" />}
              Stop
            </Button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function servingStateCopy(state: string, message: string | null): string {
  if (state !== "idle" && message) return message;
  if (state === "running") return "Local server is ready. The model loads on the first message.";
  if (state === "starting") return "Loading the model. This can take a minute on the first run.";
  if (state === "unhealthy") return "Local serving needs attention. Check logs, stop, or choose another model.";
  if (state === "exited") return "The local model server exited. Logs are kept for review.";
  if (state === "stopping") return "Stopping the local model server.";
  return "No model is running. Start the safe local model or choose another model.";
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  const isError = msg.kind === "error" || msg.content.startsWith("[error]") || msg.content.startsWith("[upstream error");
  const gated = msg.action === "connect_huggingface" || msg.errorKind === "gated_model";
  const modelId = msg.modelId && msg.modelId !== "halo-forge" ? msg.modelId : undefined;
  const modelUrl = msg.modelUrl ?? (modelId ? `https://huggingface.co/${modelId}` : undefined);
  return (
    <div className={cn("flex gap-2.5", isUser ? "justify-end" : "justify-start")}>
      {!isUser ? (
        <div className="shrink-0 mt-0.5 text-fg-disabled">
          {isError ? (
            <AlertCircle className="h-4 w-4 text-danger" />
          ) : (
            <span className="block h-4 w-4 rounded-sm bg-accent" aria-hidden />
          )}
        </div>
      ) : null}
      <div
        className={cn(
          "rounded-md px-3 py-2 text-[13px] max-w-[78%] whitespace-pre-wrap break-words",
          isUser
            ? "bg-accent-bg text-fg border border-accent/30"
            : isError
              ? "bg-danger-bg text-danger border border-danger/30"
              : "bg-surface text-fg border border-border-subtle",
        )}
      >
        <div>{msg.content}</div>
        {gated ? (
          <div className="mt-2 flex flex-wrap gap-2 border-t border-danger/20 pt-2">
            <Button asChild size="sm" variant="ghost">
              <Link to="/connect" search={{ section: "huggingface", hfModel: modelId, from: "/playground" }}>
                Connect Hugging Face
              </Link>
            </Button>
            <Button asChild size="sm" variant="ghost">
              <Link to="/models">Choose open model</Link>
            </Button>
            {modelUrl ? (
              <Button asChild size="sm" variant="ghost">
                <a href={modelUrl} target="_blank" rel="noreferrer">
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open model page
                </a>
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>
      {isUser ? (
        <div className="shrink-0 mt-0.5 text-fg-disabled">
          <User className="h-4 w-4" />
        </div>
      ) : null}
    </div>
  );
}

function SettingsPanel(props: {
  systemPrompt: string; setSystemPrompt: (v: string) => void;
  model: string; setModel: (v: string) => void;
  serveUrl: string; setServeUrl: (v: string) => void;
  apiKey: string; setApiKey: (v: string) => void;
  managedLocal: boolean;
  maxTokens: number; setMaxTokens: (v: number) => void;
  temperature: number; setTemperature: (v: number) => void;
  topP: number; setTopP: (v: number) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>SETTINGS</CardEyebrow>
          <CardTitle>{props.managedLocal ? "Local serving settings" : "External endpoint & sampling"}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-2.5">
        <Row label="System prompt">
          <textarea
            value={props.systemPrompt}
            onChange={(e) => props.setSystemPrompt(e.target.value)}
            rows={2}
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent resize-y"
          />
        </Row>
        <Row label="Model">
          <input
            type="text"
            value={props.model}
            onChange={(e) => props.setModel(e.target.value)}
            placeholder="halo-forge / Qwen/Qwen2.5-3B-Instruct / ..."
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent font-mono"
          />
        </Row>
        <Row label="Serve URL">
          <input
            type="text"
            value={props.serveUrl}
            onChange={(e) => props.setServeUrl(e.target.value)}
            placeholder="http://127.0.0.1:8001/v1"
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent font-mono"
          />
        </Row>
        <Row label="External API key">
          <input
            type="password"
            value={props.apiKey}
            onChange={(e) => props.setApiKey(e.target.value)}
            placeholder={props.managedLocal ? "Not used for managed local serving" : "(optional)"}
            disabled={props.managedLocal}
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent font-mono disabled:opacity-60 disabled:cursor-not-allowed"
          />
          {props.managedLocal ? (
            <p className="mt-1 text-[11px] text-fg-muted">
              Managed local serving uses Hugging Face access from Connection. This field is only for external endpoints.
            </p>
          ) : null}
        </Row>
        <div className="grid grid-cols-3 gap-2">
          <Row label="Max tokens" compact>
            <input
              type="number"
              value={props.maxTokens}
              min={1}
              max={8192}
              onChange={(e) => props.setMaxTokens(Number(e.target.value) || 256)}
              className="w-full bg-bg border border-border-subtle rounded-md px-2 py-1 text-[12px] focus:outline-none focus:border-accent font-mono"
            />
          </Row>
          <Row label="Temperature" compact>
            <input
              type="number"
              value={props.temperature}
              step={0.05}
              min={0}
              max={2}
              onChange={(e) => props.setTemperature(Number(e.target.value) || 0)}
              className="w-full bg-bg border border-border-subtle rounded-md px-2 py-1 text-[12px] focus:outline-none focus:border-accent font-mono"
            />
          </Row>
          <Row label="Top-p" compact>
            <input
              type="number"
              value={props.topP}
              step={0.05}
              min={0}
              max={1}
              onChange={(e) => props.setTopP(Number(e.target.value) || 1)}
              className="w-full bg-bg border border-border-subtle rounded-md px-2 py-1 text-[12px] focus:outline-none focus:border-accent font-mono"
            />
          </Row>
        </div>
      </CardContent>
    </Card>
  );
}

function Row({
  label,
  compact = false,
  children,
}: {
  label: string;
  compact?: boolean;
  children: React.ReactNode;
}) {
  if (compact) {
    return (
      <div className="space-y-1">
        <span className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled">
          {label}
        </span>
        {children}
      </div>
    );
  }
  return (
    <div className="flex items-start gap-2">
      <span className="text-[10px] uppercase tracking-[0.12em] text-fg-disabled w-[88px] shrink-0 pt-1.5">
        {label}
      </span>
      <div className="flex-1">{children}</div>
    </div>
  );
}
