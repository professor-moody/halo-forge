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
 * Multi-run chart — overlays the same metric (e.g. train_loss) across
 * N runs. Each run gets a deterministic color from the palette so the
 * series order in the legend matches the chart.
 *
 * Different from MetricChart in /charts/metric-chart.tsx, which overlays
 * MULTIPLE METRICS for a SINGLE run. This component is the inverse:
 * SINGLE METRIC across MULTIPLE RUNS. Splitting them keeps each
 * component focused; sharing would force more generic prop typings
 * that obscured what each component actually does.
 */

/**
 * Per-run series for the chart.
 *   id    — short identifier shown in the legend
 *   data  — flat array of `{x, y}` records to plot
 *   color — explicit color override (otherwise we pick from the palette)
 */
export interface RunSeries {
  id: string;
  data: Array<{ cycle: number; value: number | null }>;
  color?: string;
}

export interface MultiRunChartProps {
  /** All series to overlay — each must use a stable `id`. */
  series: RunSeries[];
  /** Format the value in tooltip + Y axis. Default toFixed(3). */
  format?: (v: number) => string;
  height?: number;
  emptyState?: ReactNode;
  className?: string;
}

/**
 * Color palette chosen for distinguishability against halo-forge's
 * dark surfaces. First slot is copper (the brand accent) so the
 * primary pinned run gets the brand-mark color; the rest fan out
 * across the spectrum without repeating tones.
 */
const PALETTE = [
  "var(--color-accent)",        // copper
  "oklch(72% 0.12 200)",        // cyan
  "oklch(74% 0.13 145)",        // green
  "oklch(72% 0.16 320)",        // magenta
  "oklch(80% 0.14 75)",         // amber-yellow
  "oklch(70% 0.13 260)",        // indigo
];

export function colorForIndex(i: number): string {
  return PALETTE[i % PALETTE.length];
}

export function MultiRunChart({
  series,
  format = (v) => v.toFixed(3),
  height = 220,
  emptyState,
  className,
}: MultiRunChartProps) {
  if (!series.length) {
    return (
      <div
        className={cn(
          "flex items-center justify-center text-xs text-fg-subtle",
          className,
        )}
        style={{ height }}
      >
        {emptyState ?? "Pin runs to compare."}
      </div>
    );
  }

  // recharts wants a single data array indexed by X. Each row contains a
  // value for every series at that cycle index, with nulls where a run
  // didn't reach that cycle. Build that pivoted view here.
  const allCycles = new Set<number>();
  for (const s of series) for (const p of s.data) allCycles.add(p.cycle);
  const sortedCycles = [...allCycles].sort((a, b) => a - b);
  const pivoted = sortedCycles.map((cycle) => {
    const row: Record<string, number | null> = { cycle };
    for (const s of series) {
      const point = s.data.find((p) => p.cycle === cycle);
      row[s.id] = point?.value ?? null;
    }
    return row;
  });

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={pivoted} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-border-subtle)"
            vertical={false}
          />
          <XAxis
            dataKey="cycle"
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
            tickFormatter={(v) => (typeof v === "number" ? format(v) : String(v))}
            tick={{
              fontSize: 10,
              fill: "var(--color-fg-subtle)",
              fontFamily: "var(--font-mono)",
            }}
            width={48}
          />
          <Tooltip
            cursor={{
              stroke: "var(--color-border-strong)",
              strokeDasharray: "3 3",
            }}
            content={
              ((props: ChartTooltipProps) => (
                <RunTooltip {...props} series={series} format={format} />
              )) as unknown as undefined
            }
          />
          {series.map((s, i) => (
            <Line
              key={s.id}
              type="monotone"
              dataKey={s.id}
              stroke={s.color ?? colorForIndex(i)}
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

/* ---------------------------------------------------------------------- */

type ChartTooltipPayloadEntry = {
  dataKey?: string | number;
  value?: number | string | null;
  color?: string;
};

type ChartTooltipProps = {
  active?: boolean;
  payload?: ChartTooltipPayloadEntry[];
  label?: string | number;
};

function RunTooltip({
  active,
  payload,
  label,
  series,
  format,
}: ChartTooltipProps & {
  series: RunSeries[];
  format: (v: number) => string;
}) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="rounded-md border border-border bg-elevated px-3 py-2 shadow-lg min-w-[180px]">
      <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-fg-disabled mb-1">
        cycle {label}
      </div>
      <div className="space-y-1">
        {payload.map((p, i) => {
          if (p.value == null) return null;
          const seriesEntry = series.find((s) => s.id === p.dataKey);
          if (!seriesEntry) return null;
          const color = seriesEntry.color ?? colorForIndex(series.indexOf(seriesEntry));
          const value =
            typeof p.value === "number" ? format(p.value) : String(p.value);
          return (
            <div
              key={`${String(p.dataKey)}-${i}`}
              className="flex items-center justify-between gap-3"
            >
              <span className="flex items-center gap-1.5 text-[11px] text-fg-muted truncate max-w-[16ch]">
                <span className="status-dot" style={{ background: color }} />
                <span className="font-mono truncate">{seriesEntry.id}</span>
              </span>
              <span className="font-mono tabular-nums text-[11px] text-fg">
                {value}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
