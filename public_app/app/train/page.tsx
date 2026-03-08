import { AppShell } from "../../components/ui";
import { TrainClient } from "./train-client";

export default function TrainPage() {
  return (
    <AppShell
      title="Training"
      subtitle="Configure a run, review launch risk, and start from a compact control workspace."
      statusItems={[
        { label: "Workspace", value: "launch review", tone: "neutral" },
        { label: "Default mode", value: "quickstart", tone: "success" },
      ]}
    >
      <TrainClient />
    </AppShell>
  );
}
