import { AppShell } from "../../components/ui";
import { TrainClient } from "./train-client";

export default function TrainPage() {
  return (
    <AppShell
      title="Training"
      subtitle="Configure a run, review launch risk, and start with a clear understanding of what the run is expected to produce."
      statusItems={[
        { label: "Launch review", value: "available", tone: "success" },
        { label: "Default flow", value: "quickstart", tone: "neutral" },
      ]}
    >
      <TrainClient />
    </AppShell>
  );
}
