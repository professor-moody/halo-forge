#!/usr/bin/env python3
"""Generate internal all-module E2E walkthrough playbooks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.all_module_readiness import ALL_MODULES
from halo_forge.all_module_walkthroughs import (
    checklist_mapping_for_module,
    walkthrough_step_templates,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed

DEFAULT_WALKTHROUGH_DIR = Path(".internal_docs/research_testing/walkthroughs")
DEFAULT_DOSSIER_DIR = Path(".internal_docs/research_testing/modules")

MODULE_DOSSIER_MAP: Dict[str, str] = {
    "config": "CONFIG_RESEARCH_TESTING.md",
    "data": "DATA_RESEARCH_TESTING.md",
    "info": "PLOT_INFO_RESEARCH_TESTING.md",
    "plot": "PLOT_INFO_RESEARCH_TESTING.md",
    "sft": "SFT_RESEARCH_TESTING.md",
    "raft": "RAFT_RESEARCH_TESTING.md",
    "benchmark_code": "CODE_BENCHMARK_RESEARCH_TESTING.md",
    "benchmark_non_code": "BENCHMARK_OPS_RESEARCH_TESTING.md",
    "inference": "INFERENCE_RESEARCH_TESTING.md",
    "vlm": "VLM_RESEARCH_TESTING.md",
    "audio": "AUDIO_RESEARCH_TESTING.md",
    "reasoning": "REASONING_RESEARCH_TESTING.md",
    "agentic": "AGENTIC_RESEARCH_TESTING.md",
    "ui_ops": "UI_OPS_RESEARCH_TESTING.md",
}


def _select_modules(module_values: Iterable[str]) -> List[str]:
    selected: List[str] = []
    for module in module_values:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported module selection: {key}")
        if key not in selected:
            selected.append(key)
    if selected:
        return selected
    return list(ALL_MODULES)


def _module_title(module: str) -> str:
    return module.replace("_", " ").upper()


def _playbook_file_name(module: str) -> str:
    return f"{module.upper()}_E2E_WALKTHROUGH.md"


def _render_playbook(module: str, seed: int) -> str:
    steps = walkthrough_step_templates(seed=seed)[module]
    checklist_rows = checklist_mapping_for_module(module, seed=seed)

    lines: List[str] = []
    lines.append(f"# {_module_title(module)} Full E2E Walkthrough")
    lines.append("")
    lines.append("## Preconditions")
    lines.append("- Local repository checkout is up to date.")
    lines.append(f"- Deterministic seed: `{seed}`.")
    lines.append("- Required runtime dependencies for selected module are installed.")
    lines.append("")

    lines.append("## CLI Step-by-Step Commands")
    for step in steps:
        if step.kind != "cli":
            continue
        cmd = " ".join(step.command)
        lines.append(f"### `{step.step_id}` {step.title}")
        lines.append(f"- Instruction: {step.instruction}")
        lines.append(f"- Command: `{cmd}`")
        lines.append(f"- Expected: {step.expected_outcome}")
        lines.append("")

    lines.append("## UI Click Path and Expected States")
    for step in steps:
        if step.kind != "ui":
            continue
        lines.append(f"### `{step.step_id}` {step.title}")
        lines.append(f"- Route: `{step.ui_route}`")
        lines.append(f"- Action: {step.instruction}")
        lines.append(f"- Expected: {step.expected_outcome}")
        lines.append("")

    lines.append("## Evidence Capture Checklist")
    for step in steps:
        if step.kind != "evidence":
            continue
        lines.append(f"### `{step.step_id}` {step.title}")
        lines.append(f"- Validation: {step.instruction}")
        lines.append(f"- Expected: {step.expected_outcome}")
        if step.evidence_paths:
            for path in step.evidence_paths:
                lines.append(f"- Required path: `{path}`")
        else:
            lines.append("- Required path: *(none declared)*")
        lines.append("")

    lines.append("## Failure Branches and Triage")
    lines.append("- CLI command parse failure: capture stderr and verify command flags against `halo-forge --help`.")
    lines.append("- UI route/state mismatch: capture screenshot + browser console and cross-check route wiring.")
    lines.append("- Missing evidence artifact: capture `ls -la` of output directory and inspect launch context/training summary.")
    lines.append("")

    lines.append("## Rerun and Resume Instructions")
    lines.append(f"- Contract rerun: `python3 scripts/run_all_module_walkthroughs.py --module {module} --profile contract-v1`")
    lines.append(f"- Live local rerun: `python3 scripts/run_all_module_walkthroughs.py --module {module} --profile live-local --execute`")
    lines.append("- For cycle-based trainers, prefer UI `Resume Latest` when checkpoint metadata exists.")
    lines.append("")

    lines.append("## Checklist-to-Step Mapping")
    lines.append("| Checklist Item | Step IDs | Coverage |")
    lines.append("|---|---|---|")
    for row in checklist_rows:
        step_ids = ", ".join(row["step_ids"])
        lines.append(f"| {row['item']} | `{step_ids}` | `{row['coverage']}` |")
    lines.append("")

    return "\n".join(lines)


def _render_index(modules: List[str], seed: int) -> str:
    lines: List[str] = []
    lines.append("# All Module E2E Walkthrough Index")
    lines.append("")
    lines.append(f"Generated for modules: `{', '.join(modules)}`")
    lines.append(f"Deterministic seed: `{seed}`")
    lines.append("")
    lines.append("## Playbooks")
    for module in modules:
        filename = _playbook_file_name(module)
        lines.append(f"- `{filename}`: {_module_title(module)}")
    lines.append("")

    lines.append("## Gap Summary Appendix")
    lines.append("| Module | Partial Items | Gap Items |")
    lines.append("|---|---|---|")
    for module in modules:
        rows = checklist_mapping_for_module(module, seed=seed)
        partial = [row["item"] for row in rows if row["coverage"] == "partial"]
        gaps = [row["item"] for row in rows if row["coverage"] == "gap"]
        partial_text = "; ".join(partial) if partial else "-"
        gap_text = "; ".join(gaps) if gaps else "-"
        lines.append(f"| {module} | {partial_text} | {gap_text} |")
    lines.append("")
    return "\n".join(lines)


def _upsert_dossier_sections(modules: List[str], walkthrough_dir: Path, dossier_dir: Path) -> None:
    start_marker = "<!-- WALKTHROUGH_MAPPING_START -->"
    end_marker = "<!-- WALKTHROUGH_MAPPING_END -->"

    touched_files: Dict[Path, List[str]] = {}
    for module in modules:
        dossier_name = MODULE_DOSSIER_MAP.get(module)
        if not dossier_name:
            continue
        dossier_path = dossier_dir / dossier_name
        if not dossier_path.exists():
            continue
        touched_files.setdefault(dossier_path, []).append(module)

    for dossier_path, mapped_modules in touched_files.items():
        existing = dossier_path.read_text(encoding="utf-8")
        block_lines: List[str] = []
        block_lines.append(start_marker)
        block_lines.append("## Full E2E Walkthrough Link")
        for module in mapped_modules:
            filename = _playbook_file_name(module)
            block_lines.append(
                f"- `{filename}` ({module})"
            )
        block_lines.append("")
        block_lines.append("## Walkthrough Coverage Mapping")
        block_lines.append("| Module | Checklist Item | Step IDs | Coverage |")
        block_lines.append("|---|---|---|---|")
        for module in mapped_modules:
            for row in checklist_mapping_for_module(module):
                step_ids = ", ".join(row["step_ids"])
                block_lines.append(
                    f"| {module} | {row['item']} | `{step_ids}` | `{row['coverage']}` |"
                )
        block_lines.append(end_marker)
        block = "\n".join(block_lines)

        if start_marker in existing and end_marker in existing:
            prefix, remainder = existing.split(start_marker, 1)
            _, suffix = remainder.split(end_marker, 1)
            updated = prefix.rstrip() + "\n\n" + block + "\n" + suffix.lstrip("\n")
        else:
            updated = existing.rstrip() + "\n\n" + block + "\n"

        dossier_path.write_text(updated, encoding="utf-8")


def _update_modules_index(modules: List[str], dossier_dir: Path) -> None:
    index_path = dossier_dir / "INDEX.md"
    if not index_path.exists():
        return

    start_marker = "<!-- WALKTHROUGH_INDEX_START -->"
    end_marker = "<!-- WALKTHROUGH_INDEX_END -->"
    block_lines: List[str] = []
    block_lines.append(start_marker)
    block_lines.append("## Full E2E Walkthrough Coverage")
    block_lines.append("- Playbook root: `.internal_docs/research_testing/walkthroughs/`")
    block_lines.append("")
    block_lines.append("| Module | Playbook | Gaps |")
    block_lines.append("|---|---|---|")
    for module in modules:
        filename = _playbook_file_name(module)
        rows = checklist_mapping_for_module(module)
        gaps = [row["item"] for row in rows if row["coverage"] == "gap"]
        gap_text = "; ".join(gaps) if gaps else "-"
        block_lines.append(
            f"| {module} | `{filename}` | {gap_text} |"
        )
    block_lines.append(end_marker)
    block = "\n".join(block_lines)

    existing = index_path.read_text(encoding="utf-8")
    if start_marker in existing and end_marker in existing:
        prefix, remainder = existing.split(start_marker, 1)
        _, suffix = remainder.split(end_marker, 1)
        updated = prefix.rstrip() + "\n\n" + block + "\n" + suffix.lstrip("\n")
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    index_path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate all-module walkthrough playbooks")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Deterministic seed used in generated command examples",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Filter module(s) for playbook generation (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_WALKTHROUGH_DIR),
        help="Target directory for generated walkthrough markdown files",
    )
    parser.add_argument(
        "--dossier-dir",
        default=str(DEFAULT_DOSSIER_DIR),
        help="Module dossier directory for walkthrough mapping updates",
    )
    parser.add_argument(
        "--skip-dossier-update",
        action="store_true",
        help="Skip updating existing module dossier mapping blocks",
    )
    args = parser.parse_args()

    seed = normalize_seed(args.seed)
    modules = _select_modules(args.module)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    for module in modules:
        content = _render_playbook(module, seed)
        path = output_dir / _playbook_file_name(module)
        path.write_text(content + "\n", encoding="utf-8")
        print(f"PLAYBOOK module={module} file={path}")

    index_path = output_dir / "INDEX.md"
    index_path.write_text(_render_index(modules, seed) + "\n", encoding="utf-8")
    print(f"PLAYBOOK index={index_path}")

    if not args.skip_dossier_update:
        dossier_dir = Path(args.dossier_dir)
        _upsert_dossier_sections(modules, output_dir, dossier_dir)
        _update_modules_index(modules, dossier_dir)
        print(f"PLAYBOOK dossier_update_dir={dossier_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
