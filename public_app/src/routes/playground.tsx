import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowLeft,
  Loader2,
  Send,
  Server,
  Settings,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type PlaygroundChatRequest, type PlaygroundMessage } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardEyebrow, CardHeader, CardTitle } from "@/components/ui/card";
import { queryKeys, useServeLogs, useServeStatus, useServeStop } from "@/lib/hooks";
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

type ChatMessage = PlaygroundMessage & { id: string };

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful assistant. Respond concisely and accurately.";
const DEFAULT_SERVE_URL = "http://127.0.0.1:8001/v1";

function PlaygroundRoute() {
  const queryClient = useQueryClient();
  const serveStatus = useServeStatus();
  const serveStop = useServeStop();
  const serveLogs = useServeLogs(80, Boolean(serveStatus.data?.logs_available));
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

  const chatMutation = useMutation({
    mutationFn: (req: PlaygroundChatRequest) => api.playgroundChat(req),
  });

  async function send() {
    const text = input.trim();
    if (!text || chatMutation.isPending) return;

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
        api_key: apiKey || undefined,
      });

      if (resp.upstream_error) {
        setMessages((prev) => [
          ...prev,
          {
            id: `${Date.now()}-e`,
            role: "assistant",
            content: formatUpstreamError(resp),
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
          content: `[error] ${(exc as Error).message ?? "request failed"}`,
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
    return "This model requires Hugging Face access. Choose an open model, log in with a token, or use a local artifact.";
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
            maxTokens={maxTokens} setMaxTokens={setMaxTokens}
            temperature={temperature} setTemperature={setTemperature}
            topP={topP} setTopP={setTopP}
          />
        ) : null}

        <ServeStatusPanel
          state={serveStatus.data?.state ?? "idle"}
          model={serveStatus.data?.model ?? null}
          url={serveStatus.data?.url ?? DEFAULT_SERVE_URL}
          message={serveStatus.data?.message ?? null}
          logsAvailable={Boolean(serveStatus.data?.logs_available)}
          logPath={serveStatus.data?.log_path ?? null}
          logLines={serveLogs.data?.lines ?? []}
          loading={serveStatus.isLoading}
          stopping={serveStop.isPending}
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
              <CardTitle>{model}</CardTitle>
              <span className="font-mono text-[10px] text-fg-disabled truncate ml-1">
                {serveUrl.replace(/^https?:\/\//, "")}
              </span>
            </div>
            <Badge tone="neutral" size="sm">
              t={temperature.toFixed(2)} · top_p={topP.toFixed(2)} · max={maxTokens}
            </Badge>
          </CardHeader>
          <CardContent className="p-0">
            <div className="px-4 py-4 space-y-3 min-h-[280px] max-h-[60vh] overflow-y-auto">
              {messages.length === 0 ? (
                <div className="text-center text-sm text-fg-muted py-10 max-w-[44ch] mx-auto">
                  <p>
                    Start a local model from Models or Results, then send a message here.
                  </p>
                  <p className="text-[11px] text-fg-disabled mt-3">
                    Cmd/Ctrl+Enter sends. Settings can point this chat at an external endpoint.
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
                placeholder="Send a message…"
                rows={3}
                className="flex-1 bg-bg border border-border-subtle rounded-md px-2.5 py-2 text-[13px] focus:outline-none focus:border-accent resize-y"
              />
              <Button
                variant="primary"
                size="sm"
                onClick={send}
                disabled={!input.trim() || chatMutation.isPending}
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
  state,
  model,
  url,
  message,
  logsAvailable,
  logPath,
  logLines,
  loading,
  stopping,
  onStop,
}: {
  state: string;
  model: string | null;
  url: string;
  message: string | null;
  logsAvailable: boolean;
  logPath: string | null;
  logLines: string[];
  loading: boolean;
  stopping: boolean;
  onStop: () => void;
}) {
  const running = state === "running" || state === "starting" || state === "unhealthy";
  const latestLog = logLines.filter(Boolean).slice(-1)[0] ?? null;
  const tone = state === "running" ? "success" : state === "starting" ? "warning" : state === "unhealthy" || state === "exited" ? "danger" : "neutral";
  const label =
    state === "running"
      ? "ready"
      : state === "starting"
        ? "starting"
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
            {message ? (
              <div className="mt-1 text-[11px] text-fg-muted">
                {message}
                {logsAvailable && logPath ? (
                  <span className="font-mono text-fg-subtle"> · logs: {logPath}</span>
                ) : null}
              </div>
            ) : null}
            {latestLog ? (
              <div className="mt-1 truncate font-mono text-[11px] text-fg-disabled">
                {latestLog}
              </div>
            ) : null}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
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

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  const isError = msg.content.startsWith("[error]") || msg.content.startsWith("[upstream error");
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
        {msg.content}
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
  maxTokens: number; setMaxTokens: (v: number) => void;
  temperature: number; setTemperature: (v: number) => void;
  topP: number; setTopP: (v: number) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardEyebrow>SETTINGS</CardEyebrow>
          <CardTitle>Endpoint &amp; sampling</CardTitle>
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
        <Row label="API key">
          <input
            type="password"
            value={props.apiKey}
            onChange={(e) => props.setApiKey(e.target.value)}
            placeholder="(leave blank for local serve)"
            className="w-full bg-bg border border-border-subtle rounded-md px-2.5 py-1.5 text-[12px] focus:outline-none focus:border-accent font-mono"
          />
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
