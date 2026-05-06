import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/hooks";
import { Topbar } from "@/components/shell";
import { Card, CardContent, CardHeader, CardEyebrow, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { relativeTime } from "@/lib/utils";

export const Route = createFileRoute("/runs/$runId")({
  component: RunDetailRoute,
});

function RunDetailRoute() {
  const { runId } = Route.useParams();
  const { data, isLoading, isError } = useQuery({
    queryKey: queryKeys.runDetail(runId),
    queryFn: () => api.runDetail(runId),
  });

  return (
    <>
      <Topbar
        eyebrow={`Runs / ${data?.modality ?? ""}`}
        title={runId}
        subtitle={data?.model_name ?? undefined}
      />
      <div className="px-6 py-6 space-y-4">
        {isLoading ? (
          <Card>
            <CardContent className="space-y-2 py-12">
              <div className="h-4 w-1/2 animate-pulse rounded bg-surface-hover" />
              <div className="h-4 w-1/3 animate-pulse rounded bg-surface-hover" />
            </CardContent>
          </Card>
        ) : isError || !data ? (
          <Card>
            <CardContent className="py-12 text-center text-sm text-fg-muted">
              Run not found.
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <CardEyebrow>Summary</CardEyebrow>
                <CardTitle>{data.modality}</CardTitle>
              </div>
              <Badge
                tone={
                  data.effectiveness?.verdict === "passed"
                    ? "success"
                    : data.effectiveness?.verdict === "failed"
                      ? "danger"
                      : "neutral"
                }
                dot
              >
                {data.effectiveness?.verdict ?? "pending"}
              </Badge>
            </CardHeader>
            <CardContent className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
              <Field label="Cycles" value={String(data.cycles_executed ?? "—")} mono />
              <Field
                label="Final loss"
                value={
                  typeof data.final_train_loss === "number"
                    ? data.final_train_loss.toFixed(3)
                    : "—"
                }
                mono
              />
              <Field
                label="Weights updated"
                value={data.weights_updated ? "yes" : "no"}
              />
              <Field
                label="When"
                value={data.created_at ? relativeTime(data.created_at) : "—"}
              />
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-fg-subtle font-medium">
        {label}
      </div>
      <div className={mono ? "mt-1 font-mono text-fg" : "mt-1 text-fg"}>{value}</div>
    </div>
  );
}
