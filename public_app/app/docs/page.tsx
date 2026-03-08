import { AppShell, SectionCard, StatusChip } from "../../components/ui";
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
    <AppShell>
      <SectionCard
        title="Docs grounded in qualification truth"
        subtitle="Capability and readiness copy should follow the backend evidence, not drift from it."
      >
        <div className="list">
          {payload.items.map((item) => (
            <article key={item.slug} className="list-row">
              <header>
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.source_path}</p>
                </div>
                <StatusChip
                  tone={item.audience === "product" ? "success" : "neutral"}
                  label={item.audience}
                />
              </header>
              <p>{item.summary}</p>
              <div className="button-row">
                <a className="secondary-button" href={item.doc_url}>
                  Open doc route
                </a>
              </div>
            </article>
          ))}
        </div>
      </SectionCard>
    </AppShell>
  );
}
