import { AppShell, SectionCard } from "../../components/ui";
import { TrainClient } from "./train-client";

export default function TrainPage() {
  return (
    <AppShell>
      <SectionCard
        title="Training workspace"
        subtitle="One clear launch path for users, research details when they matter."
      >
        <p>
          This public surface keeps advanced diagnostics available but secondary. The internal
          NiceGUI console remains the place for deep ops workflows and raw trace inspection.
        </p>
      </SectionCard>
      <TrainClient />
    </AppShell>
  );
}
