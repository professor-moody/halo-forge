import { ActionLink, AppShell, EmptyState, SectionCard, StatusChip } from "../../components/ui";
import { apiGet } from "../../lib/api";

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
      subtitle="Product guidance and capability references backed by repo-truth summaries and qualification-aware language."
      statusItems={[
        { label: "Docs indexed", value: String(payload.items.length), tone: "neutral" },
        { label: "Source", value: "repo truth", tone: "success" },
      ]}
      headerActions={<ActionLink href="/readiness" label="View readiness" tone="secondary" />}
    >
      <SectionCard title="Documentation catalog" subtitle="Core references grouped like software documentation, not long-form marketing content." eyebrow="Docs">
        {payload.items.length ? (
          <div className="docs-grid">
            {payload.items.map((item) => (
              <article key={item.slug} className="doc-card">
                <div className="row-main">
                  <div>
                    <h3>{item.title}</h3>
                    <div className="doc-meta">{item.source_path}</div>
                  </div>
                  <StatusChip tone={item.audience === "product" ? "success" : "neutral"} label={item.audience} />
                </div>
                <p>{item.summary}</p>
                <div className="button-row">
                  <a className="secondary-button" href={item.doc_url}>
                    Open docs
                  </a>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No docs indexed"
            body="The public docs summaries could not be loaded from repo-hosted sources."
          />
        )}
      </SectionCard>
    </AppShell>
  );
}
