#!/usr/bin/env python3
"""Fail CI when public release surfaces drift from their checked contracts.

This intentionally uses only the Python standard library so it can run before
the optional training stacks are installed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import tomllib
import types
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
HUGO_CONTENT = ROOT / "website" / "hugo-docs" / "content"
ERRORS: list[str] = []


def fail(message: str) -> None:
    ERRORS.append(message)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {path.relative_to(ROOT)}: {exc}")
        return ""


def load_json(path: Path) -> dict:
    try:
        return json.loads(read(path))
    except (TypeError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")
        return {}


def load_toml(path: Path) -> dict:
    try:
        return tomllib.loads(read(path))
    except tomllib.TOMLDecodeError as exc:
        fail(f"invalid TOML in {path.relative_to(ROOT)}: {exc}")
        return {}


def literal_assignments(path: Path, names: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        tree = ast.parse(read(path), filename=str(path))
    except SyntaxError as exc:
        fail(f"invalid Python in {path.relative_to(ROOT)}: {exc}")
        return values
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, TypeError):
                    fail(f"{target.id} must remain a literal in {path.relative_to(ROOT)}")
                    continue
                values[target.id] = str(parsed)
    return values


def check_versions() -> None:
    pyproject = load_toml(ROOT / "pyproject.toml")
    version_values = literal_assignments(
        ROOT / "halo_forge" / "version.py", {"PACKAGE_VERSION", "DISPLAY_VERSION"}
    )
    package_version = version_values.get("PACKAGE_VERSION")
    display_version = version_values.get("DISPLAY_VERSION")
    if not package_version or not display_version:
        fail("halo_forge/version.py must define literal PACKAGE_VERSION and DISPLAY_VERSION")
        return

    expected_package = (
        display_version.replace("-alpha-", "a")
        .replace("-beta-", "b")
        .replace("-rc-", "rc")
    )
    if package_version != expected_package:
        fail(
            f"package/display version mismatch: {package_version!r} != {expected_package!r} "
            f"derived from {display_version!r}"
        )

    package_surfaces = {
        "pyproject.toml": pyproject.get("project", {}).get("version"),
    }
    display_surfaces = {
        "public_app/package.json": load_json(ROOT / "public_app" / "package.json").get("version"),
        "apps/desktop-tauri/package.json": load_json(
            ROOT / "apps" / "desktop-tauri" / "package.json"
        ).get("version"),
        "apps/desktop-tauri/src-tauri/Cargo.toml": load_toml(
            ROOT / "apps" / "desktop-tauri" / "src-tauri" / "Cargo.toml"
        ).get("package", {}).get("version"),
        "apps/desktop-tauri/src-tauri/tauri.conf.json": load_json(
            ROOT / "apps" / "desktop-tauri" / "src-tauri" / "tauri.conf.json"
        ).get("version"),
        "website/hugo-docs/hugo.toml": load_toml(
            ROOT / "website" / "hugo-docs" / "hugo.toml"
        ).get("params", {}).get("version"),
    }
    for source, value in package_surfaces.items():
        if value != package_version:
            fail(f"release-version drift in {source}: {value!r} != {package_version!r}")
    for source, value in display_surfaces.items():
        if value != display_version:
            fail(f"release-version drift in {source}: {value!r} != {display_version!r}")

    startup = read(ROOT / "apps" / "desktop-tauri" / "startup" / "index.html")
    startup_versions = set(re.findall(r"\b\d+\.\d+\.\d+-(?:alpha|beta|rc)-\d+\b", startup))
    if startup_versions != {display_version}:
        fail(
            "desktop startup version drift: "
            f"found {sorted(startup_versions)!r}, expected only {display_version!r}"
        )


def public_text_files() -> list[Path]:
    files = [
        ROOT / "README.md",
        ROOT / "apps" / "desktop-tauri" / "README.md",
        ROOT / "public_app" / "src" / "routes" / "docs.tsx",
        ROOT / "public_app" / "src" / "components" / "shell" / "navigation.ts",
    ]
    files.extend(
        path
        for path in (ROOT / "docs").glob("*.md")
        if not path.name.startswith("RELEASE_NOTES_")
    )
    files.extend(
        path
        for path in (HUGO_CONTENT / "docs").rglob("*.md")
        if path.name != "changelog.md"
    )
    files.extend(
        path
        for path in (ROOT / "public_app" / "src").rglob("*.tsx")
        if path.name not in {"start.tsx", "results.tsx", "registry.tsx"}
        and path.name != "routeTree.gen.ts"
    )
    return sorted(set(files))


def check_stale_routes() -> None:
    forbidden = (
        (re.compile(r"(?<![\w-])/start\b"), "legacy /start route"),
        (re.compile(r"(?<![\w-])/results\b"), "legacy /results route"),
        (re.compile(r"(?<![\w-])/bundles\b"), "legacy /bundles route"),
        (re.compile(r"\bOpen Start\b", re.IGNORECASE), "legacy Open Start label"),
        (re.compile(r"\bUse in Start\b", re.IGNORECASE), "legacy Use in Start label"),
        (re.compile(r"\bRun Bundles\b", re.IGNORECASE), "legacy Run Bundles label"),
    )
    for path in public_text_files():
        text = read(path)
        for pattern, label in forbidden:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                fail(f"{path.relative_to(ROOT)}:{line}: {label}: {match.group(0)!r}")


LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def markdown_files_for_links() -> list[Path]:
    files = [
        ROOT / "README.md",
        ROOT / "apps" / "desktop-tauri" / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "OWN_DATA_WORKFLOW.md",
        ROOT / "docs" / "WORKSTATION_SURFACES.md",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "TRAINERS.md",
        ROOT / "docs" / "RELEASE_CHECKLIST.md",
    ]
    files.extend((HUGO_CONTENT / "docs").rglob("*.md"))
    files.extend(path for path in HUGO_CONTENT.glob("*.md"))
    return sorted(set(files))


def strip_link_target(value: str) -> str:
    value = value.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    # Markdown permits an optional quoted title after a whitespace separator.
    return value.split(maxsplit=1)[0]


def resolve_hugo_link(target: str) -> Path | None:
    clean = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not clean or clean == "/":
        return HUGO_CONTENT / "_index.md"
    relative = clean.lstrip("/").rstrip("/")
    candidates = [
        HUGO_CONTENT / f"{relative}.md",
        HUGO_CONTENT / relative / "_index.md",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def check_links() -> None:
    for path in markdown_files_for_links():
        text = read(path)
        is_hugo = HUGO_CONTENT in path.parents
        for match in LINK_RE.finditer(text):
            raw_target = strip_link_target(match.group(1))
            if not raw_target or raw_target.startswith(("#", "http://", "https://", "mailto:", "data:")):
                continue
            target_no_fragment = raw_target.split("#", 1)[0].split("?", 1)[0]
            if not target_no_fragment:
                continue
            if raw_target.startswith("/"):
                target = resolve_hugo_link(raw_target)
                # Dashboard routes are application links, not documentation files.
                if not raw_target.startswith(("/docs", "/download")):
                    continue
            else:
                target = (path.parent / unquote(target_no_fragment)).resolve()
                if is_hugo and not target.exists() and not Path(target_no_fragment).suffix:
                    target = Path(f"{target}.md")
            if target is None or not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                shown = target.relative_to(ROOT) if target and ROOT in target.parents else target
                fail(
                    f"{path.relative_to(ROOT)}:{line}: broken local link "
                    f"{raw_target!r} (resolved to {shown})"
                )


def load_scenario_registry():
    """Load the stdlib-only registry without importing halo_forge.__init__."""

    own_data = ROOT / "halo_forge" / "own_data"
    package = types.ModuleType("halo_forge")
    package.__path__ = [str(ROOT / "halo_forge")]
    subpackage = types.ModuleType("halo_forge.own_data")
    subpackage.__path__ = [str(own_data)]
    previous = {name: sys.modules.get(name) for name in ("halo_forge", "halo_forge.own_data")}
    sys.modules["halo_forge"] = package
    sys.modules["halo_forge.own_data"] = subpackage
    try:
        for name in ("models", "registry"):
            module_name = f"halo_forge.own_data.{name}"
            spec = importlib.util.spec_from_file_location(module_name, own_data / f"{name}.py")
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        return sys.modules["halo_forge.own_data.registry"].TRAINING_SCENARIOS
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


JSONL_FENCE_RE = re.compile(r"```jsonl\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
EXAMPLE_DOCS = (
    ROOT / "docs" / "OWN_DATA_WORKFLOW.md",
    HUGO_CONTENT / "docs" / "data" / "own-data.md",
    HUGO_CONTENT / "docs" / "getting-started" / "scenarios.md",
)


def check_v10_storage_and_replay_contracts() -> None:
    replay_values = literal_assignments(
        ROOT / "halo_forge" / "replay" / "manifest.py", {"MANIFEST_VERSION"}
    )
    schema_values = literal_assignments(
        ROOT / "halo_forge" / "run_db" / "schema.py", {"SCHEMA_VERSION"}
    )
    try:
        replay_version = int(replay_values["MANIFEST_VERSION"])
        schema_version = int(schema_values["SCHEMA_VERSION"])
    except (KeyError, ValueError):
        fail("replay MANIFEST_VERSION and run-db SCHEMA_VERSION must remain integer literals")
        return
    if replay_version < 5:
        fail(f"V10 requires replay manifest v5 or newer, found v{replay_version}")
    if schema_version < 13:
        fail(f"V10 requires additive SQLite schema v13 or newer, found v{schema_version}")

    replay_docs = (
        ROOT / "docs" / "REPLAY.md",
        HUGO_CONTENT / "docs" / "replay" / "_index.md",
    )
    for path in replay_docs:
        text = read(path)
        if f"format v{replay_version}" not in text:
            fail(
                f"{path.relative_to(ROOT)} must identify current replay format "
                f"v{replay_version}"
            )
        if f"MANIFEST_VERSION` is {replay_version}" not in text and (
            f"MANIFEST_VERSION = {replay_version}" not in text
        ):
            fail(
                f"{path.relative_to(ROOT)} must identify MANIFEST_VERSION "
                f"{replay_version}"
            )

    corpus_docs = (
        ROOT / "docs" / "CORPUS_ADAPTATION.md",
        HUGO_CONTENT / "docs" / "data" / "corpus-adaptation.md",
    )
    required_fragments = (
        "~/.halo-forge/corpus/bundles/<hash-prefix>/<content-hash>/",
        "halo-forge data extract --path",
        "halo-forge data inspect --path",
        "halo-forge data render <version-id> --trainer cpt",
        "halo-forge cpt train",
    )
    for path in corpus_docs:
        text = read(path)
        for fragment in required_fragments:
            if fragment not in text:
                fail(
                    f"{path.relative_to(ROOT)} is missing the V10 corpus contract "
                    f"fragment {fragment!r}"
                )
        for stale in (
            "data corpus-profile --path",
            "~/.halo-forge/datasets/extractions/",
            "--packing greedy",
        ):
            if stale in text:
                fail(
                    f"{path.relative_to(ROOT)} contains stale corpus guidance "
                    f"{stale!r}"
                )
    readme = read(ROOT / "README.md")
    if "Replay manifests v4" in readme:
        fail("README.md still advertises replay manifest v4 after V10 moved to v5")
    for required in ("**CPT**", "halo-forge cpt train", "halo-forge data extract"):
        if required not in readme:
            fail(f"README.md is missing the V10 public capability {required!r}")


def check_v17_product_completion_contracts() -> None:
    replay_values = literal_assignments(
        ROOT / "halo_forge" / "replay" / "manifest.py", {"MANIFEST_VERSION"}
    )
    schema_values = literal_assignments(
        ROOT / "halo_forge" / "run_db" / "schema.py", {"SCHEMA_VERSION"}
    )
    if replay_values.get("MANIFEST_VERSION") != "14":
        fail("V21 requires replay MANIFEST_VERSION 14")
    if schema_values.get("SCHEMA_VERSION") != "23":
        fail("V21 requires additive SQLite SCHEMA_VERSION 23")

    required = {
        ROOT / "docs" / "DATASET_REPAIR.md": (
            "repair_overlay",
            "halo-forge data repair inspect",
            "halo-forge data repair rebase",
            "Replay format v11",
        ),
        HUGO_CONTENT / "docs" / "data" / "repair.md": (
            "Fix data",
            "immutable overlay",
            "halo-forge data repair inspect",
        ),
        HUGO_CONTENT / "docs" / "getting-started" / "install-desktop.md": (
            "macOS",
            "Linux",
            "Windows",
            "release manifest",
            "SmartScreen",
        ),
    }
    for path, fragments in required.items():
        text = read(path)
        for fragment in fragments:
            if fragment not in text:
                fail(f"{path.relative_to(ROOT)} is missing V17 contract {fragment!r}")

    current_surfaces = (
        ROOT / "README.md",
        ROOT / "docs" / "WORKSTATION_SURFACES.md",
        HUGO_CONTENT / "docs" / "reference" / "public-frontend.md",
        HUGO_CONTENT / "docs" / "getting-started" / "hardware.md",
    )
    stale_claims = (
        "Windows desktop packaging is unavailable",
        "No Tauri package",
        "Windows desktop packaging is not available",
    )
    for path in current_surfaces:
        text = read(path)
        for claim in stale_claims:
            if claim in text:
                fail(f"{path.relative_to(ROOT)} contains stale V17 claim {claim!r}")

    release_workflow = read(ROOT / ".github" / "workflows" / "release.yml")
    for fragment in (
        "windows-latest",
        "--bundles deb,appimage",
        "--bundles nsis",
        "Smoke Windows bundled runtime",
        "Verify upgrade and uninstall preservation contract",
        "distribution-capability-",
    ):
        if fragment not in release_workflow:
            fail(f"release workflow is missing cross-platform contract {fragment!r}")

    v18_required = {
        ROOT / "docs" / "GUIDED_TRAINING_PLAN.md": (
            "Recommended plan → Prepare and check → Ready for proof run",
            "halo-forge train-plan recommend",
            "Schema v21",
            "Replay format v12",
        ),
        ROOT / "halo_forge" / "public_api" / "app.py": (
            '"/training-plans/recommend"',
            '"/training-plan-revisions/{revision_id}/prepare"',
            '"/training-plan-revisions/{revision_id}/capacity-check"',
            '"/training-plan-revisions/{revision_id}/proof"',
        ),
        ROOT / "public_app" / "src" / "components" / "data" / "own-data-studio.tsx": (
            "Recommended plan",
            "Prepare and check",
            "Start proof run",
        ),
    }
    for path, fragments in v18_required.items():
        text = read(path)
        for fragment in fragments:
            if fragment not in text:
                fail(f"{path.relative_to(ROOT)} is missing V18 contract {fragment!r}")

    v21_required = {
        ROOT / "docs" / "VERIFIED_TRAINING_PATHS.md": (
            "Runtime ready",
            "Path verified",
            "halo-forge runtime certify",
            "Replay v14",
        ),
        ROOT / "halo_forge" / "public_api" / "app.py": (
            '"/runtime/paths"',
            '"/training-path-revisions/{revision_id}/certify"',
            '"/training-path-certifications/{certification_id}/verify"',
            '"/release/workstation-certify"',
        ),
        ROOT / "public_app" / "src" / "routes" / "setup.tsx": (
            "Verify text training",
            "generic tensor update",
            "api.certifyTrainingPath",
        ),
        ROOT / "public_app" / "src" / "routes" / "train.tsx": (
            "Verify this training path",
            "Hardware detection alone is not treated as training readiness",
        ),
    }
    for path, fragments in v21_required.items():
        text = read(path)
        for fragment in fragments:
            if fragment not in text:
                fail(f"{path.relative_to(ROOT)} is missing V21 contract {fragment!r}")


def record_matches_scenario(record: dict, scenario) -> bool:
    for field in scenario.required_fields:
        if field in scenario.safe_constants:
            continue
        aliases = scenario.field_aliases.get(field, (field,))
        if not any(alias in record and record[alias] not in (None, "") for alias in aliases):
            return False
    return True


def check_examples_and_registry() -> None:
    try:
        registry = load_scenario_registry()
    except Exception as exc:  # pragma: no cover - exercised as a CI diagnostic
        fail(f"cannot load own-data scenario registry: {exc}")
        return

    expected_available = {
        "instruction-sft",
        "chat-sft",
        "preference-pairs",
        "prompt-reward",
        "reasoning-sft",
        "tool-agentic",
        "vlm-captioning",
        "vlm-qa",
        "audio-asr",
        "corpus-adaptation",
        "text-classification",
        "text-multilabel",
        "embedding-pairs",
        "reranking",
        "image-classification",
        "audio-classification",
    }
    expected_unavailable = {"audio-tts"}
    scenarios = {item.id: item for item in registry.list(include_unavailable=True)}
    if {key for key, value in scenarios.items() if value.available} != expected_available:
        fail("own-data available scenario IDs drifted; update docs and this checked contract together")
    if {key for key, value in scenarios.items() if not value.available} != expected_unavailable:
        fail("own-data unavailable scenario IDs drifted; update docs and this checked contract together")

    for identifier, scenario in scenarios.items():
        expected_anchor = f"own-data/{identifier}"
        if scenario.documentation_anchor != expected_anchor:
            fail(
                f"scenario {identifier!r} anchor drift: "
                f"{scenario.documentation_anchor!r} != {expected_anchor!r}"
            )
        if scenario.available and not scenario.examples:
            fail(f"available scenario {identifier!r} has no checked format example")
        for example in scenario.examples:
            filename, files = registry.template_files(identifier, example.id)
            payload = files[filename]
            if filename != example.filename:
                fail(f"scenario {identifier!r} template filename drift for {example.id!r}")
            if example.format in {"markdown", "md", "txt", "text"}:
                if identifier != "corpus-adaptation":
                    fail(
                        f"scenario {identifier!r} uses a document fixture without "
                        "the corpus-adaptation contract"
                    )
                if not payload.decode("utf-8").strip():
                    fail(
                        f"scenario {identifier!r} example {example.id!r} is empty"
                    )
                if not all(
                    isinstance(record, dict)
                    and isinstance(record.get("text"), str)
                    and record["text"].strip()
                    for record in example.records
                ):
                    fail(
                        f"scenario {identifier!r} example {example.id!r} "
                        "does not provide extractable document text"
                    )
                continue
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"scenario {identifier!r} example {example.id!r}:{line_number}: {exc}")
                    continue
                if not isinstance(record, dict) or not record_matches_scenario(record, scenario):
                    fail(
                        f"scenario {identifier!r} example {example.id!r}:{line_number} "
                        "does not cover its required shape"
                    )
                    continue
                for media_field in ("image", "image_path", "audio"):
                    media_path = record.get(media_field)
                    if isinstance(media_path, str) and media_path not in files:
                        fail(
                            f"scenario {identifier!r} example {example.id!r}:{line_number} "
                            f"does not bundle referenced media asset {media_path!r}"
                        )

    available = [value for value in scenarios.values() if value.available]
    for path in EXAMPLE_DOCS:
        text = read(path)
        for identifier, scenario in scenarios.items():
            if identifier not in text or scenario.documentation_anchor not in text:
                fail(
                    f"{path.relative_to(ROOT)} must name scenario {identifier!r} "
                    f"and anchor {scenario.documentation_anchor!r}"
                )
        fences = JSONL_FENCE_RE.findall(text)
        if not fences:
            fail(f"{path.relative_to(ROOT)} contains no fenced JSONL examples")
            continue
        contains_media = False
        for fence_index, fence in enumerate(fences, 1):
            for line_index, line in enumerate(fence.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(
                        f"{path.relative_to(ROOT)} JSONL fence {fence_index}, "
                        f"line {line_index}: {exc}"
                    )
                    continue
                if not isinstance(record, dict):
                    fail(
                        f"{path.relative_to(ROOT)} JSONL fence {fence_index}, "
                        f"line {line_index}: record must be an object"
                    )
                    continue
                if not any(record_matches_scenario(record, scenario) for scenario in available):
                    fail(
                        f"{path.relative_to(ROOT)} JSONL fence {fence_index}, "
                        f"line {line_index}: no available scenario accepts this shape"
                    )
                contains_media = contains_media or bool({"image", "image_path", "audio"} & record.keys())
        normalized = text.lower()
        if contains_media and not (
            "referenced" in normalized
            and "checksummed" in normalized
            and ("image/audio" in normalized or "png/wav" in normalized)
        ):
            fail(
                f"{path.relative_to(ROOT)} must distinguish standalone media snippets "
                "from complete checksummed image/audio fixture bundles"
            )


def main() -> int:
    check_versions()
    check_stale_routes()
    check_links()
    check_v10_storage_and_replay_contracts()
    check_v17_product_completion_contracts()
    check_examples_and_registry()
    if ERRORS:
        print("Release interface checks failed:", file=sys.stderr)
        for error in ERRORS:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        "Release interface checks passed: versions, routes, links, replay/storage, "
        "V17 product completion, V18 training plans, V21 real training paths, "
        "examples, and scenarios are aligned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
