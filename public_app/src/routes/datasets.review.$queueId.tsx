import { createFileRoute } from "@tanstack/react-router";
import { ReviewQueueWorkspace } from "@/components/review/review-studio";

export const Route = createFileRoute("/datasets/review/$queueId")({
  component: ReviewQueueWorkspace,
});
