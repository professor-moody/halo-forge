import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowLeft, Sparkles } from "lucide-react";
import { OwnDataStudio } from "@/components/data/own-data-studio";
import { DataSectionTabs } from "@/components/data/data-section-tabs";
import { Topbar } from "@/components/shell";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/datasets/new")({
  component: OwnDataRoute,
  validateSearch: (search): { example?: string; inspection?: string; trainingPlanRevision?: string; source?: string; scenario?: string; repairRevision?: string } => ({
    // TanStack's default search parser may deserialize `?example=1` to the
    // number 1. Keep links and pasted URLs equivalent instead of silently
    // dropping the first-run example mode.
    example: search.example === "1" || search.example === 1 || search.example === true ? "1" : undefined,
    inspection: typeof search.inspection === "string" && search.inspection.trim() ? search.inspection : undefined,
    trainingPlanRevision: typeof search.trainingPlanRevision === "string" && search.trainingPlanRevision.trim() ? search.trainingPlanRevision : undefined,
    source: typeof search.source === "string" && search.source.trim() ? search.source : undefined,
    scenario: typeof search.scenario === "string" && search.scenario.trim() ? search.scenario : undefined,
    repairRevision: typeof search.repairRevision === "string" && search.repairRevision.trim() ? search.repairRevision : undefined,
  }),
});

function OwnDataRoute() {
  const { example, inspection, trainingPlanRevision, source, scenario, repairRevision } = Route.useSearch();
  return (
    <>
      <Topbar
        eyebrow="Data"
        title="Train on your data"
        subtitle="Inspect, map, prepare, and prove a training path without changing the source."
        actions={
          <>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/datasets">
                <ArrowLeft />
                Dataset library
              </Link>
            </Button>
            {!example ? (
              <Button variant="secondary" size="sm" asChild>
                <Link to="/datasets/new" search={{ example: "1" }}>
                  <Sparkles />
                  Try a working example
                </Link>
              </Button>
            ) : null}
          </>
        }
      />
      <DataSectionTabs />
      <OwnDataStudio
        startWithExample={example === "1" && !inspection}
        initialInspectionId={inspection}
        initialTrainingPlanRevisionId={trainingPlanRevision}
        initialSourcePath={source}
        initialScenarioRevisionId={scenario}
        initialRepairRevisionId={repairRevision}
      />
    </>
  );
}
