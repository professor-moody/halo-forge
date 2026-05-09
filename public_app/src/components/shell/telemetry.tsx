import { Activity, AlertTriangle, Cpu, Gauge, Info, Thermometer, Zap } from "lucide-react";
import type { TelemetrySample } from "@/lib/api";
import { useEventSource } from "@/lib/event-source";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/**
 * Hardware telemetry strip — the visual signature of halo-forge.
 *
 * Persistent across every authenticated route: it's the operator's
 * always-visible instrumentation panel. Five readouts, each rendered
 * the same way: icon + uppercase label + mono value + unit + optional
 * supplementary text.
 *
 * Field availability varies by backend:
 *   - rocm-smi / nvidia-smi: every value populated.
 *   - Apple Silicon: GPU util / power / temp are None (no sudo APIs).
 *     The provider returns `note` explaining why; we surface it via a
 *     small info icon at the right end of the strip.
 *   - CPU-only: VRAM is None; CPU/RAM is populated.
 *
 * VRAM gets a horizontal bar overlay because it's the value most often
 * used to *act* — when the bar fills past ~80% on a long run, the
 * operator knows OOM is imminent. Color shifts from accent (copper)
 * to warning (amber) to danger (red) at thresholds.
 */
export function TelemetryStrip() {
  // Phase E: telemetry now streams over SSE. The browser's EventSource
  // handles reconnection (server emits `retry: 3000`); we just keep
  // the latest sample. `isLoading` is true until the first event lands.
  const { data, status } = useEventSource<TelemetrySample>(
    "/api/public/telemetry/stream",
  );
  const isLoading = data === null && status !== "error";

  return (
    <div className="border-b border-border bg-bg-subtle/40">
      <div className="relative flex items-stretch divide-x divide-border-subtle px-5">
        <Cell
          icon={Gauge}
          label="GPU UTIL"
          value={fmtPercent(data?.gpu_util_percent)}
          unit="%"
          loading={isLoading}
        />
        <VramCell sample={data ?? undefined} loading={isLoading} />
        <Cell
          icon={Zap}
          label="POWER"
          value={fmtNum(data?.power_watts, 0)}
          unit="W"
          loading={isLoading}
        />
        <Cell
          icon={Thermometer}
          label="TEMP"
          value={fmtNum(data?.temp_celsius, 0)}
          unit="°C"
          loading={isLoading}
          tone={tempTone(data?.temp_celsius)}
        />
        <Cell
          icon={Activity}
          label="THROUGHPUT"
          value={fmtNum(data?.throughput_tokens_per_sec, 0)}
          unit="tok/s"
          hint={data?.active_run_id ? data.active_run_id.slice(0, 12) : undefined}
          loading={isLoading}
        />
        <Cell
          icon={Cpu}
          label="CPU"
          value={fmtPercent(data?.cpu_util_percent)}
          unit="%"
          hint={
            data?.sys_mem_used_gb != null && data?.sys_mem_total_gb != null
              ? `${data.sys_mem_used_gb.toFixed(1)} / ${data.sys_mem_total_gb.toFixed(0)} GB`
              : undefined
          }
          loading={isLoading}
        />
        <MPSFallbackChip count={data?.mps_to_cpu_fallbacks_60s ?? 0} />

        {/* Note icon — only when the backend populated `note` (typical:
            "GPU util / power / temp require sudo on macOS"). Rendered
            absolutely-positioned at the right end so it never adds
            another column to the divide-x rhythm. */}
        {data?.note ? <NoteIcon note={data.note} /> : null}
      </div>
    </div>
  );
}

function MPSFallbackChip({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <div className="flex items-center px-3 py-2.5">
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="inline-flex items-center gap-1.5 rounded-sm border border-warning/40 bg-warning/10 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-warning">
            <AlertTriangle className="h-3 w-3" />
            MPS FALLBACK
            <span className="tabular-nums">{count}</span>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" align="end" className="max-w-[30ch] text-[11px]">
          PyTorch moved one or more MPS operations to CPU in the last minute. Training will still run, but throughput may drop sharply.
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

/* ------------------------------------------------------------------------
 * Cells
 * ---------------------------------------------------------------------- */

interface CellProps {
  icon: typeof Cpu;
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  tone?: "neutral" | "warning" | "danger";
  loading?: boolean;
}

function Cell({ icon: Icon, label, value, unit, hint, tone = "neutral", loading }: CellProps) {
  return (
    <div className="flex flex-1 items-center gap-2.5 px-4 py-2.5 first:pl-0 last:pr-0">
      <Icon
        className={cn(
          "h-3.5 w-3.5 shrink-0 transition-colors",
          tone === "danger"
            ? "text-danger"
            : tone === "warning"
              ? "text-warning"
              : "text-fg-subtle",
        )}
      />
      <div className="min-w-0 flex-1">
        <div className="text-[9.5px] font-medium uppercase tracking-[0.14em] text-fg-disabled">
          {label}
        </div>
        <div className="flex items-baseline gap-1.5 mt-0.5">
          {loading && value === "—" ? (
            <span
              aria-hidden
              className="block h-3.5 w-10 animate-pulse rounded-sm bg-surface-hover"
            />
          ) : (
            <span
              className={cn(
                "font-mono text-[15px] tabular-nums tracking-tight transition-colors",
                value === "—"
                  ? "text-fg-disabled"
                  : tone === "danger"
                    ? "text-danger"
                    : tone === "warning"
                      ? "text-warning"
                      : "text-fg",
              )}
            >
              {value}
            </span>
          )}
          {unit ? (
            <span className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
              {unit}
            </span>
          ) : null}
          {hint ? (
            <span className="font-mono text-[10px] text-fg-disabled ml-auto truncate">
              {hint}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/**
 * VRAM cell renders a horizontal fill bar under the value. Linear-style
 * 2px progress strip directly under the number. Color shifts past
 * thresholds. When the value is missing entirely, it falls back to the
 * generic `<Cell>` shape.
 */
function VramCell({
  sample,
  loading,
}: {
  sample: TelemetrySample | undefined;
  loading: boolean;
}) {
  const used = sample?.vram_used_gb ?? null;
  const total = sample?.vram_total_gb ?? null;
  const ratio = used != null && total != null && total > 0 ? used / total : null;

  if (used == null) {
    return (
      <Cell
        icon={Cpu}
        label="VRAM"
        value="—"
        unit="GB"
        hint={total != null ? `of ${total.toFixed(0)}` : undefined}
        loading={loading}
      />
    );
  }

  const tone: "neutral" | "warning" | "danger" =
    ratio != null && ratio > 0.9 ? "danger" : ratio != null && ratio > 0.75 ? "warning" : "neutral";

  return (
    <div className="flex flex-1 items-center gap-2.5 px-4 py-2.5 first:pl-0 last:pr-0">
      <Cpu
        className={cn(
          "h-3.5 w-3.5 shrink-0 transition-colors",
          tone === "danger"
            ? "text-danger"
            : tone === "warning"
              ? "text-warning"
              : "text-fg-subtle",
        )}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[9.5px] font-medium uppercase tracking-[0.14em] text-fg-disabled">
            VRAM
          </span>
          {ratio != null ? (
            <span className="font-mono text-[10px] text-fg-subtle">
              {(ratio * 100).toFixed(0)}%
            </span>
          ) : null}
        </div>
        <div className="flex items-baseline gap-1.5 mt-0.5">
          <span
            className={cn(
              "font-mono text-[15px] tabular-nums tracking-tight transition-colors",
              tone === "danger"
                ? "text-danger"
                : tone === "warning"
                  ? "text-warning"
                  : "text-fg",
            )}
          >
            {fmtNum(used, used < 10 ? 2 : 1)}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
            GB
          </span>
          {total != null ? (
            <span className="font-mono text-[10px] text-fg-disabled ml-auto">
              of {total.toFixed(0)}
            </span>
          ) : null}
        </div>
        {/* Fill bar */}
        {ratio != null ? (
          <div className="mt-1 h-0.5 w-full overflow-hidden rounded-sm bg-border-subtle">
            <div
              className={cn(
                "h-full transition-all duration-500",
                tone === "danger"
                  ? "bg-danger"
                  : tone === "warning"
                    ? "bg-warning"
                    : "bg-accent",
              )}
              style={{ width: `${Math.min(100, ratio * 100).toFixed(1)}%` }}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function NoteIcon({ note }: { note: string }) {
  return (
    <div className="absolute right-2 top-1/2 -translate-y-1/2">
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label="Telemetry notes"
            className="text-fg-disabled hover:text-fg-subtle transition-colors p-1 rounded-sm"
          >
            <Info className="h-3 w-3" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" align="end" className="max-w-[28ch] text-[11px]">
          {note}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

/* ------------------------------------------------------------------------
 * Formatting helpers — keep formatting stable across re-renders so the
 * strip never visually jitters when a poll completes.
 * ---------------------------------------------------------------------- */

function fmtNum(value: number | null | undefined, digits = 0): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function fmtPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(0);
}

function tempTone(temp: number | null | undefined): "neutral" | "warning" | "danger" {
  if (temp == null) return "neutral";
  if (temp >= 90) return "danger";
  if (temp >= 80) return "warning";
  return "neutral";
}
