import { AppShell } from "@/components/app-ui";
import { TrainClient } from "./train-client";

export default function TrainPage() {
  return (
    <AppShell
      title="Training"
      subtitle="Configure a run, review launch risk, and start training."
      statusItems={[
        { label: "Launch review", value: "available", tone: "success" },
        { label: "Flow", value: "quickstart", tone: "neutral" },
      ]}
    >
      <TrainClient />
    </AppShell>
  );
}
