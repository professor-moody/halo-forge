import {
  Activity,
  BarChart3,
  BookOpen,
  Database,
  FlaskConical,
  LayoutDashboard,
  PackageSearch,
  Play,
  Plug,
  Settings2,
  Stethoscope,
  type LucideIcon,
} from "lucide-react";

export type WorkspaceNavItem = {
  id: string;
  to: string;
  label: string;
  description: string;
  icon: LucideIcon;
  shortcut?: string;
  aliases?: string[];
};

export const PRIMARY_NAV: WorkspaceNavItem[] = [
  { id: "overview", to: "/", label: "Overview", description: "Workstation summary", icon: LayoutDashboard, shortcut: "G O" },
  { id: "data", to: "/datasets", label: "Data", description: "Sources, versions, and quality", icon: Database, shortcut: "G D", aliases: ["datasets"] },
  { id: "train", to: "/train", label: "Train", description: "Guided and advanced launches", icon: Play, shortcut: "G T", aliases: ["start"] },
  { id: "experiments", to: "/sweeps", label: "Experiments", description: "Repeats and searches", icon: FlaskConical, shortcut: "G X", aliases: ["sweeps"] },
  { id: "runs", to: "/runs", label: "Runs", description: "Monitor and compare", icon: Activity, shortcut: "G R", aliases: ["results", "collections"] },
  { id: "evaluate", to: "/eval", label: "Evaluate", description: "Suites, results, and evidence", icon: BarChart3, shortcut: "G E", aliases: ["eval", "verifiers"] },
  { id: "models", to: "/models", label: "Models", description: "Catalog, artifacts, and serving", icon: PackageSearch, shortcut: "G M", aliases: ["playground", "artifact studio"] },
];

export const SYSTEM_NAV: WorkspaceNavItem[] = [
  { id: "diagnostics", to: "/diagnostics", label: "Diagnostics", description: "Runtime checks and logs", icon: Stethoscope, shortcut: "G S" },
  { id: "connection", to: "/connect", label: "Connection", description: "Workstation and access", icon: Plug },
  { id: "documentation", to: "/docs", label: "Documentation", description: "Local operator reference", icon: BookOpen },
];

export const ACTIVITY_NAV_ITEM = {
  id: "activity",
  label: "Activity",
  description: "Queue, workers, and resource owner",
  icon: Activity,
  shortcut: "G A",
};

export const SYSTEM_CONTROL = {
  id: "system",
  label: "System",
  description: "Diagnostics, connection, and help",
  icon: Settings2,
};

export function isNavigationActive(pathname: string, to: string): boolean {
  if (to === "/") return pathname === "/";
  return pathname === to || pathname.startsWith(`${to}/`);
}

