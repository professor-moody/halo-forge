"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
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
  headerActions?: ReactNode;
}>;

const NAV_ITEMS = [
  { href: "/", label: "Overview", short: "OV" },
  { href: "/train", label: "Training", short: "TR" },
  { href: "/results", label: "Results", short: "RS" },
  { href: "/readiness", label: "Readiness", short: "RD" },
  { href: "/docs", label: "Docs", short: "DC" },
];

export function AppShell({
  children,
  title = "Overview",
  subtitle = "Operational visibility across training, results, and readiness.",
  statusItems = [],
  headerActions,
}: AppShellProps) {
  const pathname = usePathname();

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="app-brand">
          <div className="app-brand-mark">HF</div>
          <div>
            <div className="app-brand-kicker">Halo-Forge</div>
            <div className="app-brand-title">Training platform</div>
          </div>
        </div>
        <nav className="app-nav">
          {NAV_ITEMS.map((item) => {
            const active =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(`${item.href}/`));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={active ? "app-nav-link is-active" : "app-nav-link"}
              >
                <span className="app-nav-short">{item.short}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="app-sidebar-note">
          <span className="app-sidebar-label">Default mode</span>
          <strong>Product workflow</strong>
          <p>Guided launch, live status, outcomes, and qualification truth.</p>
        </div>
      </aside>

      <div className="app-main">
        <header className="app-topbar">
          <div className="app-topbar-meta">
            <span className="app-topbar-dot" />
            <span>Public training application</span>
          </div>
          <div className="app-status-strip">
            {statusItems.map((item) => (
              <div key={`${item.label}-${item.value}`} className="app-status-item">
                <span className="app-status-label">{item.label}</span>
                <StatusChip tone={item.tone ?? "neutral"} label={item.value} />
              </div>
            ))}
          </div>
        </header>

        <div className="app-page-header">
          <div className="app-page-heading">
            <div className="app-page-kicker">Training platform</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          {headerActions ? <div className="app-page-actions">{headerActions}</div> : null}
        </div>

        <main className="app-page-content">{children}</main>
      </div>
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  actions,
  eyebrow,
  className,
  children,
}: PropsWithChildren<{
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  eyebrow?: string;
  className?: string;
}>) {
  return (
    <section className={className ? `surface-card ${className}` : "surface-card"}>
      <div className="surface-card-header">
        <div>
          {eyebrow ? <div className="surface-card-eyebrow">{eyebrow}</div> : null}
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className="surface-card-actions">{actions}</div> : null}
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

export function MetricTile({
  label,
  value,
  meta,
}: {
  label: string;
  value: string;
  meta?: string;
}) {
  return (
    <div className="metric-tile">
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      {meta ? <p>{meta}</p> : null}
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
    <Link href={href} className={tone === "primary" ? "button-primary" : "button-secondary"}>
      {label}
    </Link>
  );
}

export function ActionButton({
  label,
  tone = "secondary",
  onClick,
  disabled = false,
}: {
  label: string;
  tone?: "primary" | "secondary";
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      className={tone === "primary" ? "button-primary" : "button-secondary"}
      onClick={onClick}
      disabled={disabled}
      type="button"
    >
      {label}
    </button>
  );
}

export function InlineCallout({
  title,
  body,
  tone = "neutral",
  actions,
}: {
  title: string;
  body: string;
  tone?: "success" | "warning" | "danger" | "neutral";
  actions?: ReactNode;
}) {
  return (
    <div className={`inline-callout tone-${tone}`}>
      <div>
        <h3>{title}</h3>
        <p>{body}</p>
      </div>
      {actions ? <div className="inline-callout-actions">{actions}</div> : null}
    </div>
  );
}

export function ResearchSection({
  title,
  summary,
  children,
  defaultOpen = false,
}: PropsWithChildren<{ title: string; summary: string; defaultOpen?: boolean }>) {
  return (
    <details className="research-accordion" open={defaultOpen}>
      <summary>
        <div>
          <span>{title}</span>
          <small>{summary}</small>
        </div>
        <strong>Details</strong>
      </summary>
      <div className="research-accordion-body">{children}</div>
    </details>
  );
}

export function DetailDrawer({
  open,
  title,
  subtitle,
  onClose,
  children,
}: PropsWithChildren<{
  open: boolean;
  title: string;
  subtitle?: string;
  onClose: () => void;
}>) {
  if (!open) {
    return null;
  }
  return (
    <div className="drawer-root" role="dialog" aria-modal="true">
      <button className="drawer-backdrop" type="button" onClick={onClose} aria-label="Close details" />
      <aside className="drawer-panel">
        <div className="drawer-header">
          <div>
            <div className="surface-card-eyebrow">Run details</div>
            <h2>{title}</h2>
            {subtitle ? <p>{subtitle}</p> : null}
          </div>
          <button className="drawer-close" type="button" onClick={onClose} aria-label="Close details">
            Close
          </button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
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
