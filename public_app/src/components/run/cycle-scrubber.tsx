import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * Cycle scrubber — a "playback head" for training history.
 *
 * Visual: a thin horizontal track with one tick per cycle, the
 * focused cycle highlighted in copper. Plus a small transport bar:
 * skip-back / play / skip-forward / reset-to-all. When the user
 * presses play, the focus advances one cycle per second; pressing
 * pause stops the animation.
 *
 * The owning component (the run detail route) holds the focus state
 * and slices its chart data accordingly. This component is purely
 * presentational + transport.
 */
export interface CycleScrubberProps {
  cycles: number[];
  /** Currently focused cycle, or null when "show all" is active. */
  focus: number | null;
  onFocusChange: (focus: number | null) => void;
  className?: string;
}

export function CycleScrubber({
  cycles,
  focus,
  onFocusChange,
  className,
}: CycleScrubberProps) {
  const playingRef = useRef(false);
  const intervalRef = useRef<number | null>(null);

  // Stop the animation timer if focus changes externally or component
  // unmounts. We don't track playing in state because the only thing
  // that wants to know is the play button label, and we re-render
  // anyway when focus advances.
  useEffect(() => {
    return () => {
      if (intervalRef.current != null) {
        window.clearInterval(intervalRef.current);
      }
    };
  }, []);

  if (!cycles.length) return null;
  const minCycle = cycles[0];
  const maxCycle = cycles[cycles.length - 1];

  function clamp(c: number): number {
    return Math.max(minCycle, Math.min(maxCycle, c));
  }

  function setFocus(next: number | null) {
    if (next === null) {
      stopPlay();
      onFocusChange(null);
      return;
    }
    onFocusChange(clamp(next));
  }

  function startPlay() {
    if (playingRef.current) return;
    playingRef.current = true;
    intervalRef.current = window.setInterval(() => {
      // Advance through the cycles. When we hit the end, loop back to
      // the start. Operators scrubbing through often want to see the
      // arc more than once.
      const cur = focus ?? minCycle;
      const next = cur >= maxCycle ? minCycle : cur + 1;
      onFocusChange(next);
    }, 900);
  }

  function stopPlay() {
    playingRef.current = false;
    if (intervalRef.current != null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }

  function togglePlay() {
    if (playingRef.current) stopPlay();
    else startPlay();
  }

  // Stop playback when we reach the last cycle while playing.
  // (The interval will re-trigger but checking here keeps the focus
  // pinned and the play button label honest.)
  const isPlaying = playingRef.current;

  return (
    <div className={cn("flex items-center gap-3", className)}>
      {/* Transport */}
      <div className="flex items-center gap-0.5">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Previous cycle"
          disabled={focus !== null && focus <= minCycle}
          onClick={() => setFocus((focus ?? maxCycle) - 1)}
          className="h-7 w-7"
        >
          <SkipBack className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label={isPlaying ? "Pause" : "Play"}
          onClick={togglePlay}
          className="h-7 w-7"
        >
          {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Next cycle"
          disabled={focus !== null && focus >= maxCycle}
          onClick={() => setFocus((focus ?? minCycle) + 1)}
          className="h-7 w-7"
        >
          <SkipForward className="h-3 w-3" />
        </Button>
      </div>

      {/* Track — one tick per cycle. Click to jump. */}
      <div
        role="slider"
        aria-valuemin={minCycle}
        aria-valuemax={maxCycle}
        aria-valuenow={focus ?? maxCycle}
        aria-label="Cycle scrubber"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setFocus((focus ?? maxCycle) - 1);
          if (e.key === "ArrowRight") setFocus((focus ?? minCycle) + 1);
        }}
        className="flex-1 flex items-center gap-1 h-7 group focus-visible:outline-none"
      >
        {cycles.map((c) => {
          const active = focus === null ? true : c <= focus;
          const isFocus = focus === c;
          return (
            <button
              key={c}
              type="button"
              onClick={() => setFocus(c)}
              aria-label={`Focus cycle ${c}`}
              className={cn(
                "flex-1 h-1.5 rounded-sm transition-colors",
                isFocus
                  ? "bg-accent h-3"
                  : active
                    ? "bg-accent/40 hover:bg-accent/70"
                    : "bg-border hover:bg-border-strong",
              )}
            />
          );
        })}
      </div>

      {/* Focus readout */}
      <div className="flex items-center gap-1.5 font-mono text-[11px]">
        {focus === null ? (
          <span className="text-fg-muted">cycles</span>
        ) : (
          <>
            <span className="text-fg-disabled">cycle</span>
            <span className="text-accent tabular-nums">{focus}</span>
          </>
        )}
        <span className="text-fg-disabled">/</span>
        <span className="text-fg-muted tabular-nums">{maxCycle}</span>
      </div>

      {/* Reset */}
      {focus !== null ? (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setFocus(null)}
          className="text-[11px] h-7"
        >
          <RotateCcw className="h-3 w-3" />
          all
        </Button>
      ) : null}
    </div>
  );
}
