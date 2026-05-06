import { createFileRoute } from "@tanstack/react-router";
import { Construction } from "lucide-react";
import { Topbar } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";

export const Route = createFileRoute("/train")({
  component: TrainRoute,
});

function TrainRoute() {
  return (
    <>
      <Topbar
        eyebrow="Workspace"
        title="Training"
        subtitle="Launch and monitor RAFT or SFT runs."
      />
      <div className="px-6 py-6">
        <Card>
          <CardContent className="flex items-center gap-3 py-12 text-fg-muted">
            <Construction className="h-5 w-5 text-fg-subtle" />
            <div>
              <div className="text-sm font-medium text-fg">Training launch UI in progress</div>
              <div className="text-xs">
                The new training surface lands after the overview is approved.
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
