#!/usr/bin/env python3
"""
Generate internal operator burn-in markdown packet from ops reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.ops_e2e_reliability import (
    DEFAULT_OPS_E2E_REPORT_FILE,
    OPS_E2E_STATUSES,
    load_ops_e2e_report,
)
from halo_forge.ops_module_readiness import (
    DEFAULT_OPS_READINESS_REPORT_FILE,
    OPS_MODULES,
    load_ops_readiness_report,
)


def _default_packet_path() -> Path:
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return REPO_ROOT / ".internal_docs" / "research_testing" / "packets" / f"{date_tag}_ops_burnin.md"


def _safe_status(value: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in OPS_E2E_STATUSES else "fail"


def _format_packet(
    *,
    e2e_report,
    readiness_report,
    triage_owner: str,
) -> str:
    lines: list[str] = []
    lines.append("# Ops Burn-In Packet")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- E2E report: `{DEFAULT_OPS_E2E_REPORT_FILE}`")
    lines.append(f"- Readiness report: `{DEFAULT_OPS_READINESS_REPORT_FILE}`")
    lines.append(f"- Triage owner: {triage_owner}")
    lines.append("")
    lines.append("## Summary")
    for module in OPS_MODULES:
        e2e_status = _safe_status(e2e_report.modules[module].status)
        readiness_status = readiness_report.modules[module].status
        lines.append(
            f"- `{module}`: e2e={e2e_status}, readiness={readiness_status}, "
            f"errors={len(e2e_report.modules[module].errors)}, "
            f"warnings={len(e2e_report.modules[module].warnings)}"
        )
    lines.append("")

    for module in OPS_MODULES:
        e2e = e2e_report.modules[module]
        readiness = readiness_report.modules[module]
        lines.append(f"## {module.upper()}")
        lines.append(f"- E2E status: `{_safe_status(e2e.status)}`")
        lines.append(f"- Readiness status: `{readiness.status}`")
        lines.append(f"- Launch lifecycle: launch={e2e.launch_ok}, stop={e2e.stop_ok}, relaunch={e2e.relaunch_ok}, resume_latest={e2e.resume_latest_ok}")
        lines.append(f"- Artifacts ok: `{e2e.artifacts_ok}`")
        lines.append(f"- Last output dir: `{e2e.last_output_dir}`")
        lines.append("- Blockers:")
        if e2e.errors:
            for err in e2e.errors:
                lines.append(f"  - {err}")
        else:
            lines.append("  - none")
        lines.append("- Warnings:")
        if e2e.warnings:
            for warn in e2e.warnings:
                lines.append(f"  - {warn}")
        else:
            lines.append("  - none")
        lines.append("- Evidence pointers:")
        if e2e.evidence:
            for key, value in sorted(e2e.evidence.items()):
                lines.append(f"  - {key}: `{value}`")
        else:
            lines.append("  - none")
        lines.append("- Rerun command:")
        lines.append(
            "  - `python3 scripts/run_ops_e2e_reliability.py --validate-module "
            f"{module}={e2e.last_output_dir or '<output_dir>'} --strict`"
        )
        lines.append("- Triage owner:")
        lines.append(f"  - {triage_owner}")
        lines.append("- Sign-off:")
        lines.append("  - [ ] Ready for operator testing")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate internal ops burn-in packet")
    parser.add_argument(
        "--e2e-report",
        default=str(DEFAULT_OPS_E2E_REPORT_FILE),
        help="Path to ops E2E reliability report",
    )
    parser.add_argument(
        "--readiness-report",
        default=str(DEFAULT_OPS_READINESS_REPORT_FILE),
        help="Path to ops module readiness report",
    )
    parser.add_argument(
        "--output-file",
        default=str(_default_packet_path()),
        help="Internal markdown packet output path",
    )
    parser.add_argument(
        "--triage-owner",
        default="TBD",
        help="Triage owner name for packet sections",
    )
    args = parser.parse_args()

    e2e_report_path = Path(args.e2e_report)
    readiness_report_path = Path(args.readiness_report)

    if not e2e_report_path.exists():
        print(f"ERROR: missing E2E report: {e2e_report_path}")
        return 1
    if not readiness_report_path.exists():
        print(f"ERROR: missing readiness report: {readiness_report_path}")
        return 1

    try:
        e2e_report = load_ops_e2e_report(e2e_report_path)
        readiness_report = load_ops_readiness_report(readiness_report_path)
    except Exception as exc:
        print(f"ERROR: failed to parse report(s): {exc}")
        return 1

    packet = _format_packet(
        e2e_report=e2e_report,
        readiness_report=readiness_report,
        triage_owner=str(args.triage_owner).strip() or "TBD",
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet, encoding="utf-8")
    print(f"Wrote burn-in packet: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
