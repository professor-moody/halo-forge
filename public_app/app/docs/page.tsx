import { ActionLink, AppShell, EmptyState, SectionCard, StatusBadge } from "@/components/app-ui";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { apiGet } from "@/lib/api";

type DocsResponse = {
  items: Array<{
    slug: string;
    title: string;
    summary: string;
    source_path: string;
    doc_url: string;
    audience: string;
  }>;
};

async function getDocs() {
  try {
    return await apiGet<DocsResponse>("/docs");
  } catch {
    return { items: [] };
  }
}

export default async function DocsPage() {
  const payload = await getDocs();
  return (
    <AppShell
      title="Docs"
      subtitle="Product guidance and capability references."
      statusItems={[
        { label: "Indexed", value: String(payload.items.length), tone: "neutral" },
      ]}
      headerActions={<ActionLink href="/readiness" label="View readiness" tone="secondary" />}
    >
      <SectionCard title="Documentation catalog" eyebrow="Docs">
        {payload.items.length ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {payload.items.map((item) => (
              <Card key={item.slug}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="text-sm font-medium text-foreground">{item.title}</h3>
                    <Badge variant={item.audience === "product" ? "success" : "secondary"} className="shrink-0">
                      {item.audience}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">{item.source_path}</div>
                  <p className="text-xs text-muted-foreground mt-2 line-clamp-2">{item.summary}</p>
                  <div className="mt-3">
                    <Button variant="outline" size="sm" asChild>
                      <a href={item.doc_url}>Open docs</a>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <EmptyState title="No docs indexed" body="Public docs could not be loaded." />
        )}
      </SectionCard>
    </AppShell>
  );
}
