import { createFileRoute } from "@tanstack/react-router";
import { ReviewStudioHome } from "@/components/review/review-studio";

export const Route = createFileRoute("/datasets/review/")({
  component: ReviewStudioHome,
});
