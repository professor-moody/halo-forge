import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowDownToLine,
  Pause,
  Play,
  Terminal,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type RunLogs } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardEyebrow,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

/**
 * Logs panel — tails the training log for a run.
 *
 * Aesthetic: terminal feel without being a literal xterm. Mono throughout,
 * lines numbered on the left, hover highlights the row, keyword colors
 * for OK / WARNING / ERROR / iteration markers (subtle — copper accent
 * for "Iter NNN" so the operator can scan progress quickly).
 *
 * Behavior:
 *   - Polls /api/public/runs/{id}/logs every 3s while autoTail is on
 *   - "Pause" stops polling so the operator can read without the view
 *     jumping; "Resume" turns it back on
 *   - Auto-scrolls to bottom when new lines arrive — but only if the
 *     user is already at the bottom. If they've scrolled up to read,
 *     we don't yank them away. Standard terminal-tail behavior.
 *   - Renders "Jump to latest" when scrolled up so the user can rejoin
 *     the tail without manual scrolling.
 */

export interface LogsPanelProps {
  runId: string;
  /** Number of lines to fetch per poll. Default 200. */
  tail?: number;
  /** Initial polling state. Default true. */
  autoTail?: boolean;
  /** Render at this fixed height. Default 360px. */
  height?: number;
}

export function LogsPanel({
  runId,
  tail = 200,
  autoTail: initialAutoTail = true,
  height = 360,
}: LogsPanelProps) {
  const [autoTail, setAutoTail] = useState(initialAutoTail);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether the user is pinned to the bottom. We only auto-scroll when
  // they are; otherwise we leave them where they were and surface a
  // "jump to latest" button.
  const stickToBottomRef = useRef<boolean>(true);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);

  const { data, isLoading, isError, error, dataUpdatedAt } = useQuery<RunLogs>({
    queryKey: ["run-logs", runId, tail],
    queryFn: () => api.runLogs(runId, tail),
    refetchInterval: autoTail ? 3_000 : false,
    refetchIntervalInBackground: false,
  });

  // Auto-scroll handler: if the user was at the bottom when new lines
  // arrived, follow. Otherwise stay put.
  useEffect(() => {
    if (!stickToBottomRef.current) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [data?.lines.length, dataUpdatedAt]);

  // Track user scroll position. If they've scrolled up by more than
  // 24px, drop the auto-follow flag and show the "jump to latest" pill.
  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distanceFromBottom < 24;
    stickToBottomRef.current = atBottom;
    setShowJumpToLatest(!atBottom);
  }

  function jumpToLatest() {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stickToBottomRef.current = true;
    setShowJumpToLatest(false);
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2 min-w-0">
          <CardEyebrow>STREAM</CardEyebrow>
          <CardTitle>Logs</CardTitle>
          <Terminal className="h-3.5 w-3.5 text-fg-disabled" />
          {data?.log_path ? (
            <span
              className="font-mono text-[10px] text-fg-disabled truncate ml-1"
              title={data.log_path}
            >
              {basename(data.log_path)}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setAutoTail((v) => !v)}
            aria-label={autoTail ? "Pause log tail" : "Resume log tail"}
          >
            {autoTail ? (
              <>
                <Pause className="h-3 w-3" />
                <span className="text-[11px]">Tailing</span>
              </>
            ) : (
              <>
                <Play className="h-3 w-3" />
                <span className="text-[11px]">Paused</span>
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0 relative">
        {isLoading ? (
          <div className="px-4 py-12 text-center text-xs text-fg-subtle">
            Loading logs…
          </div>
        ) : isError ? (
          <div className="px-4 py-8 flex items-center gap-2 text-xs text-danger">
            <AlertCircle className="h-3.5 w-3.5" />
            {(error as Error)?.message ?? "Could not load logs."}
          </div>
        ) : data && !data.available ? (
          <div className="px-4 py-12 flex flex-col items-center gap-1.5 text-center">
            <div className="text-[12px] text-fg">No log file alongside this run.</div>
            <div className="text-[11px] text-fg-subtle max-w-[44ch]">
              {data.reason ??
                "Logs land here once the trainer's TeeWriter starts emitting."}
            </div>
          </div>
        ) : data && data.lines.length ? (
          <>
            <div
              ref={scrollRef}
              onScroll={onScroll}
              style={{ height }}
              className="overflow-y-auto bg-bg-subtle/50 font-mono text-[11.5px] leading-[1.4]"
            >
              {data.lines.map((line, i) => {
                const tone = lineTone(line);
                return (
                  <div
                    key={`${i}-${line.length}`}
                    className={cn(
                      "flex items-baseline gap-3 px-4 hover:bg-surface-hover/40",
                      tone === "error"
                        ? "text-danger"
                        : tone === "warning"
                          ? "text-warning"
                          : tone === "ok"
                            ? "text-success"
                            : tone === "iter"
                              ? "text-accent"
                              : "text-fg-muted",
                    )}
                  >
                    <span className="text-fg-disabled select-none w-10 text-right shrink-0">
                      {i + 1}
                    </span>
                    <span className="whitespace-pre-wrap break-all">
                      {line || " "}
                    </span>
                  </div>
                );
              })}
            </div>
            {showJumpToLatest ? (
              <button
                type="button"
                onClick={jumpToLatest}
                className="absolute bottom-3 right-3 inline-flex items-center gap-1.5 rounded-full border border-border bg-elevated px-3 py-1 text-[11px] text-fg shadow-md hover:bg-surface-hover transition-colors"
              >
                <ArrowDownToLine className="h-3 w-3" />
                Jump to latest
              </button>
            ) : null}
          </>
        ) : (
          <div className="px-4 py-12 text-center text-xs text-fg-subtle">
            Log file is empty.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Heuristic line classifier. Looks for canonical halo-forge markers
 * (`[OK]`, `[X]`, `[!]`, `Iter N:`, `Cycle N`). False positives don't
 * matter much — color is supplementary, not load-bearing.
 */
function lineTone(line: string): "ok" | "warning" | "error" | "iter" | "neutral" {
  const lower = line.toLowerCase();
  if (lower.includes("error") || lower.includes("traceback") || lower.includes("[x]"))
    return "error";
  if (lower.includes("warning") || lower.includes("[!]")) return "warning";
  if (line.startsWith("[OK]") || lower.includes(" complete in ") || lower.includes("saved checkpoint"))
    return "ok";
  if (/^iter\s+\d+:/i.test(line.trim())) return "iter";
  return "neutral";
}

function basename(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}
