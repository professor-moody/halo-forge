import Link from "next/link";
import { PropsWithChildren, type ReactNode } from "react";

type ShellStatus = {
  label: string;
  value: string;
  tone?: "success" | "warning" | "danger" | "neutral" | string;
};

type AppShellProps = PropsWithChildren<{
  title?: string;
  subtitle?: string;
  statusItems?: ShellStatus[];
}>;

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/train", label: "Train" },
  { href: "/results", label: "Results" },
  { href: "/readiness", label: "Readiness" },
  { href: "/docs", label: "Docs" },
];

export function AppShell({
  children,
  title = "Overview",
  subtitle = "Operational visibility across training, results, and readiness.",
  statusItems = [],
}: AppShellProps) {
  return (
    <div className="workstation-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-kicker">halo-forge</div>
          <div className="brand-title">Public workstation</div>
        </div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <Link key={item.href} href={item.href} className="sidebar-link">
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>
      <div className="workspace">
        <header className="workspace-header">
          <div>
            <div className="page-kicker">Workspace</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div className="status-rail">
            {statusItems.map((item) => (
              <div key={`${item.label}-${item.value}`} className="status-block">
                <label>{item.label}</label>
                <StatusChip tone={item.tone ?? "neutral"} label={item.value} />
              </div>
            ))}
          </div>
        </header>
        <main className="workspace-content">{children}</main>
      </div>
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  actions,
  children,
}: PropsWithChildren<{ title: string; subtitle?: string; actions?: ReactNode }>) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function StatusChip({
  tone,
  label,
}: {
  tone: "success" | "warning" | "danger" | "neutral" | string;
  label: string;
}) {
  return <span className={`status-chip tone-${tone}`}>{label}</span>;
}

export function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="stat-tile">
      <label>{label}</label>
      <strong>{value}</strong>
      {hint ? <span>{hint}</span> : null}
    </div>
  );
}

export function MetricPill({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="metric-pill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function ActionLink({
  href,
  label,
  tone = "secondary",
}: {
  href: string;
  label: string;
  tone?: "primary" | "secondary";
}) {
  return (
    <Link href={href} className={tone === "primary" ? "primary-button" : "secondary-button"}>
      {label}
    </Link>
  );
}

export function ResearchSection({
  title,
  summary,
  children,
  defaultOpen = false,
}: PropsWithChildren<{ title: string; summary: string; defaultOpen?: boolean }>) {
  return (
    <details className="research-section" open={defaultOpen}>
      <summary>
        <span>{title}</span>
        <small>{summary}</small>
      </summary>
      <div className="research-body">{children}</div>
    </details>
  );
}

export function EmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}
