import { createFileRoute } from "@tanstack/react-router";
import { ArrowUpRight, BookOpen } from "lucide-react";
import { Topbar } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";

export const Route = createFileRoute("/docs")({
  component: DocsRoute,
});

const LINKS = [
  { href: "https://halo-forge.io/docs", label: "halo-forge.io/docs" },
  {
    href: "https://github.com/professor-moody/halo-forge/blob/main/docs/HARDWARE_NOTES.md",
    label: "Hardware notes (ROCm / MPS / MLX)",
  },
  {
    href: "https://github.com/professor-moody/halo-forge/blob/main/docs/MLX.md",
    label: "MLX backend guide",
  },
  {
    href: "https://github.com/professor-moody/halo-forge/blob/main/docs/VERIFIERS.md",
    label: "Verifiers reference",
  },
];

function DocsRoute() {
  return (
    <>
      <Topbar eyebrow="Workspace" title="Docs" subtitle="External documentation and references." />
      <div className="px-6 py-6 space-y-3 max-w-2xl">
        {LINKS.map((link) => (
          <Card key={link.href}>
            <CardContent className="py-3.5">
              <a
                href={link.href}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-3 group"
              >
                <BookOpen className="h-4 w-4 text-fg-subtle group-hover:text-accent transition-colors" />
                <span className="flex-1 text-sm text-fg group-hover:text-accent transition-colors">
                  {link.label}
                </span>
                <ArrowUpRight className="h-4 w-4 text-fg-subtle group-hover:text-accent transition-colors" />
              </a>
            </CardContent>
          </Card>
        ))}
      </div>
    </>
  );
}
