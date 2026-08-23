import { Link, useRouterState } from "@tanstack/react-router";
import { Database, ListChecks } from "lucide-react";
import { cn } from "@/lib/utils";

export function DataSectionTabs() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const reviewActive = pathname.startsWith("/datasets/review");

  return (
    <nav aria-label="Data workspace" className="flex items-center gap-1 border-b border-border-subtle bg-bg-subtle/55 px-5 py-2">
      <Link
        to="/datasets"
        className={cn("inline-flex h-8 items-center gap-2 rounded-md px-3 text-[11.5px] font-medium transition-colors", !reviewActive ? "bg-surface text-fg shadow-sm" : "text-fg-muted hover:bg-surface/70 hover:text-fg")}
      >
        <Database className="h-3.5 w-3.5" />
        Datasets
      </Link>
      <Link
        to="/datasets/review"
        search={{ new: undefined, source: undefined, sourceRef: undefined, baseRef: undefined }}
        className={cn("inline-flex h-8 items-center gap-2 rounded-md px-3 text-[11.5px] font-medium transition-colors", reviewActive ? "bg-surface text-fg shadow-sm" : "text-fg-muted hover:bg-surface/70 hover:text-fg")}
      >
        <ListChecks className="h-3.5 w-3.5" />
        Review queues
      </Link>
    </nav>
  );
}
