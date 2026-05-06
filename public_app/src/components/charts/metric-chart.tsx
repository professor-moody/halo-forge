import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * Generic single-axis line chart for halo-forge metrics.
 *
 * Visual rules anchored on the workstation aesthetic:
 *   - Hairline grid (1px, border-subtle), no chartjunk
 *   - Mono numerics on the axes — these are values an operator reads
 *     across cycles, not labels they skim past
 *   - Copper for the primary series, info/danger/warning for secondaries
 *   - 0.5s ease-out animations, no spring physics
 *   - Tooltip is a small instrument readout (label + mono value), not
 *     the recharts default chrome
 */

type SeriesTone = "accent" | "info" | "success" | "warning" | "danger" | "muted";

const TONE_TO_VAR: Record<SeriesTone, string> = {
  accent: "var(--color-accent)",
  info: "var(--color-info)",
  success: "var(--color-success)",
  warning: "var(--color-warning)",
  danger: "var(--color-danger)",
  muted: "var(--color-fg-subtle)",
};

export interface MetricSeries {
  /** Field name in the data row to plot. */
  key: string;
  /** Human label for legend / tooltip. */
  label: string;
  tone?: SeriesTone;
  /** Format the value for display in the tooltip. Defaults to .toFixed(3). */
  format?: (v: number) => string;
}

export interface MetricChartProps<T extends Record<string, unknown>> {
  data: T[];
  /** The X axis key, defaults to "cycle". */
  xKey?: keyof T & string;
  series: MetricSeries[];
  /** Override Y axis tick formatter. Default: toFixed(2). */
  yFormat?: (v: number) => string;
  /** Render the chart at this CSS height. Default 180px. */
  height?: number;
  /** Show / hide the grid. Default true. */
  showGrid?: boolean;
  /** Optional empty-state node when data is missing. */
  emptyState?: ReactNode;
  className?: string;
}

export function MetricChart<T extends Record<string, unknown>>({
  data,
  xKey = "cycle" as keyof T & string,
  series,
  yFormat = (v) => v.toFixed(2),
  height = 180,
  showGrid = true,
  emptyState,
  className,
}: MetricChartProps<T>) {
  if (!data.length) {
    return (
      <div
        className={cn(
          "flex items-center justify-center text-xs text-fg-subtle",
          className,
        )}
        style={{ height }}
      >
        {emptyState ?? "No data yet."}
      </div>
    );
  }

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 4 }}>
          {showGrid ? (
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--color-border-subtle)"
              vertical={false}
            />
          ) : null}
          <XAxis
            dataKey={xKey as string}
            stroke="var(--color-fg-disabled)"
            tickLine={false}
            axisLine={{ stroke: "var(--color-border)" }}
            tick={{
              fontSize: 10,
              fill: "var(--color-fg-subtle)",
              fontFamily: "var(--font-mono)",
            }}
          />
          <YAxis
            stroke="var(--color-fg-disabled)"
            tickLine={false}
            axisLine={{ stroke: "var(--color-border)" }}
            tickFormatter={(v) => (typeof v === "number" ? yFormat(v) : String(v))}
            tick={{
              fontSize: 10,
              fill: "var(--color-fg-subtle)",
              fontFamily: "var(--font-mono)",
            }}
            width={40}
          />
          <Tooltip
            cursor={{
              stroke: "var(--color-border-strong)",
              strokeDasharray: "3 3",
            }}
            content={
              ((props: ChartTooltipProps) => (
                <ChartTooltip {...props} series={series} xKey={xKey as string} />
              )) as unknown as undefined
            }
          />
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={TONE_TO_VAR[s.tone ?? "accent"]}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
              connectNulls={false}
              isAnimationActive={true}
              animationDuration={500}
              animationEasing="ease-out"
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Custom tooltip — instrument-panel readout, not the recharts default.
 * Renders the X label up top and one row per series with the value
 * in monospace so cross-row comparison is visually aligned.
 *
 * recharts' Tooltip component prop typings are notoriously generic-heavy
 * (TooltipContentProps<ValueType, NameType>) and fight strict TS in
 * subtle ways across versions. We type the props minimally here and let
 * recharts hand us the runtime payload.
 */
type ChartTooltipPayloadEntry = {
  dataKey?: string | number;
  value?: number | string | null;
};

type ChartTooltipProps = {
  active?: boolean;
  payload?: ChartTooltipPayloadEntry[];
  label?: string | number;
};

function ChartTooltip({
  active,
  payload,
  label,
  series,
  xKey,
}: ChartTooltipProps & { series: MetricSeries[]; xKey: string }) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="rounded-md border border-border bg-elevated px-3 py-2 shadow-lg min-w-[140px]">
      <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled mb-1">
        {xKey} {label}
      </div>
      <div className="space-y-1">
        {payload.map((p, i) => {
          const seriesEntry = series.find((s) => s.key === p.dataKey);
          if (!seriesEntry || p.value == null) return null;
          const value =
            typeof p.value === "number"
              ? (seriesEntry.format ?? ((v: number) => v.toFixed(3)))(p.value)
              : String(p.value);
          return (
            <div
              key={`${String(p.dataKey)}-${i}`}
              className="flex items-center justify-between gap-3"
            >
              <span className="flex items-center gap-1.5 text-[11px] text-fg-muted">
                <span
                  className="status-dot"
                  style={{ background: TONE_TO_VAR[seriesEntry.tone ?? "accent"] }}
                />
                {seriesEntry.label}
              </span>
              <span className="font-mono tabular-nums text-[11px] text-fg">{value}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
