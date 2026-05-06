import { createFileRoute } from "@tanstack/react-router";
import { Construction } from "lucide-react";
import { Topbar } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";

export const Route = createFileRoute("/results")({
  component: ResultsRoute,
});

function ResultsRoute() {
  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Results"
        subtitle="Completed training results and effectiveness evidence."
      />
      <div className="px-6 py-6">
        <Card>
          <CardContent className="flex items-center gap-3 py-12 text-fg-muted">
            <Construction className="h-5 w-5 text-fg-subtle" />
            <div>
              <div className="text-sm font-medium text-fg">Results view in progress</div>
              <div className="text-xs">
                Lands after the overview design is approved.
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
