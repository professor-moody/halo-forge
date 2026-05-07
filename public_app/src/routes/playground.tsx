import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowLeft,
  Loader2,
  Send,
  Settings,
  Trash2,
  User,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type PlaygroundChatRequest, type PlaygroundMessage } from "@/lib/api";
import { Topbar } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardEyebrow, CardHeader, CardTitle } from "@/components/ui/card";
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

function PlaygroundRoute() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [model, setModel] = useState("halo-forge");
  const [serveUrl, setServeUrl] = useState("http://127.0.0.1:8001/v1");
  const [apiKey, setApiKey] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [showSettings, setShowSettings] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
            content: `[upstream error ${resp.status ?? "?"}] ${JSON.stringify(resp.detail)}`,
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
                    Start by running{" "}
                    <span className="font-mono text-fg-subtle">halo-forge serve --model X</span>{" "}
                    in another terminal, then send a message below.
                  </p>
                  <p className="text-[11px] text-fg-disabled mt-3">
                    Cmd/Ctrl+Enter sends.
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
