import { createFileRoute, Outlet } from "@tanstack/react-router";

type ReviewSearch = {
  new?: string;
  source?: string;
  sourceRef?: string;
  baseRef?: string;
  item?: string;
  status?: string;
  q?: string;
  page?: number;
  pass?: number;
};

export const Route = createFileRoute("/datasets/review")({
  validateSearch: (search: Record<string, unknown>): ReviewSearch => ({
    new: search.new === true || search.new === 1 ? "1" : typeof search.new === "string" ? search.new : undefined,
    source: typeof search.source === "string" ? search.source : undefined,
    sourceRef: typeof search.sourceRef === "string" ? search.sourceRef : undefined,
    baseRef: typeof search.baseRef === "string" ? search.baseRef : undefined,
    item: typeof search.item === "string" ? search.item : undefined,
    status: typeof search.status === "string" ? search.status : undefined,
    q: typeof search.q === "string" ? search.q : undefined,
    page: typeof search.page === "number" && Number.isFinite(search.page) ? Math.max(1, Math.floor(search.page)) : undefined,
    pass: typeof search.pass === "number" && Number.isFinite(search.pass) ? Math.max(1, Math.floor(search.pass)) : undefined,
  }),
  component: Outlet,
});
