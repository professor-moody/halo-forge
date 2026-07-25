import { createFileRoute, Outlet } from "@tanstack/react-router";

export const Route = createFileRoute("/datasets")({
  component: DatasetLabLayout,
});

function DatasetLabLayout() {
  return <Outlet />;
}
