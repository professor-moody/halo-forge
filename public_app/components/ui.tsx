import Link from "next/link";
import { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">halo-forge public</div>
          <h1>Training that stays understandable</h1>
        </div>
        <nav className="nav">
          <Link href="/train">Train</Link>
          <Link href="/results">Results</Link>
          <Link href="/readiness">Readiness</Link>
          <Link href="/docs">Docs</Link>
        </nav>
      </header>
      <main className="page-grid">{children}</main>
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  children,
}: PropsWithChildren<{ title: string; subtitle?: string }>) {
  return (
    <section className="section-card">
      <div className="section-header">
        <h2>{title}</h2>
        {subtitle ? <p>{subtitle}</p> : null}
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
