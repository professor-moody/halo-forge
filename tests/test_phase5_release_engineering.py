#!/usr/bin/env python3
"""Phase 5 release-engineering regression guards."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

CI_WORKFLOW = Path(".github/workflows/ci.yml")
PYPROJECT = Path("pyproject.toml")

# Readiness/report scripts the main CI job runs. Each one must either gate the
# build (--strict) or carry a documented TODO(ci-strict) exception.
READINESS_SCRIPTS = (
    "scripts/run_ops_module_matrix.py",
    "scripts/run_ops_e2e_reliability.py",
    "scripts/run_ops_dataset_burnin.py",
    "scripts/run_all_module_matrix.py",
    "scripts/run_all_module_qualification.py",
    "scripts/run_all_module_bootstrap.py",
    "scripts/run_all_module_live_matrix.py",
)


# Import name -> distribution name, for the guarded modules whose PyPI name
# differs from the module you import.
_DISTRIBUTION_ALIASES = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "docx": "python-docx",
    "sklearn": "scikit-learn",
}


def _core_dependency_names() -> set[str]:
    """Return the distribution names in pyproject's [project] dependencies.

    Parsed with a regex rather than tomllib so this test still runs in the
    stdlib-only CI job and on Python 3.10, where tomllib does not exist.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert block, "pyproject.toml has no [project] dependencies array"
    return {
        re.split(r"[<>=!~\[; ]", raw.strip().strip('",'))[0].strip().lower()
        for raw in block.group(1).splitlines()
        if raw.strip().startswith('"')
    }


def _job_installs(job: str, module: str) -> bool:
    """Return True when `job` installs the distribution providing `module`.

    A job satisfies the dependency either by naming it explicitly in a pip
    command, or by installing the package itself (`pip install -e .`), which
    pulls everything in pyproject's core dependency array. Checking only for
    the literal module name would force the job back to a hand-written package
    list -- exactly the drift that let a declared core dependency go missing.
    """
    distribution = _DISTRIBUTION_ALIASES.get(module, module).lower()
    if re.search(rf"pip install[^\n]*\b{re.escape(distribution)}\b", job):
        return True
    if re.search(r"pip install[^\n]*[\"']?\.\[dev\][\"']?", job):
        text = PYPROJECT.read_text(encoding="utf-8")
        block = re.search(r"^dev\s*=\s*\[(.*?)^\]", text, re.S | re.M)
        assert block, "pyproject.toml has no dev optional-dependency array"
        declared = {
            re.split(r"[<>=!~\[; ]", raw.strip().strip('\",'))[0].strip().lower()
            for raw in block.group(1).splitlines()
            if raw.strip().startswith('"')
        }
        if distribution in declared:
            return True
    installs_project = re.search(
        r"pip install[^\n]*(?:-e\s+)?[\"']?\.(?:\[[^\]]+\])?[\"']?(?:\s|$)",
        job,
    )
    if installs_project:
        return distribution in _core_dependency_names()
    return False


def _ci_workflow_text() -> str:
    assert CI_WORKFLOW.exists()
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _ci_step_blocks(content: str) -> list[str]:
    """Split the workflow into per-step blocks, keeping each step's comments.

    A comment block sitting directly above a step documents that step, so it
    is carried forward instead of being left on the previous one.
    """
    blocks: list[str] = []
    carry = ""
    for chunk in re.split(r"\n(?=\s*- name: )", content):
        lines = chunk.split("\n")
        trailing: list[str] = []
        while lines and (not lines[-1].strip() or lines[-1].strip().startswith("#")):
            trailing.insert(0, lines.pop())
        blocks.append(carry + "\n".join(lines))
        carry = "\n".join(trailing) + "\n" if trailing else ""
    if carry:
        blocks.append(carry)
    return blocks


def _ci_job_block(content: str, job_name: str) -> str:
    """Return the text of a single top-level job definition."""
    blocks = re.split(r"\n(?=  [A-Za-z0-9_-]+:\n)", content)
    for block in blocks:
        if re.match(rf"\s*{re.escape(job_name)}:\s*\n", block):
            return block
    raise AssertionError(f"CI workflow defines no `{job_name}` job")


def test_ci_workflow_exists_with_compile_and_core_regression_steps():
    """Repository should include a CI workflow covering compile + core regression tests.

    The legacy NiceGUI-era per-phase test files were retired alongside
    the GUI itself; what survives is the modality + contract test set
    plus the nightly readiness workflows. CI must still reference
    those, the compileall guard, and the readiness baselines.
    """
    workflow = Path(".github/workflows/ci.yml")
    assert workflow.exists()

    content = workflow.read_text(encoding="utf-8")
    assert "python -m compileall -q halo_forge ui tests" in content
    # Surviving phase + contract tests CI must still run.
    assert "tests/test_phase3_truth_in_advertising.py" in content
    assert "tests/test_phase5_release_engineering.py" in content
    assert "tests/test_benchmark_results_trust_matrix.py" in content
    assert "tests/test_training_pipeline_contracts.py" in content
    assert "tests/test_dependency_packaging_contracts.py" in content
    assert "tests/test_phase6_modality_graduation.py" in content
    assert "tests/test_phase7c_modality_baseline_drift.py" in content
    assert "tests/test_phase9_non_code_modality_research_matrix.py" in content
    # Modality tests still wired.
    for modality_test in (
        "tests/test_pipeline.py",
        "tests/test_verifiers.py",
        "tests/test_inference.py",
        "tests/test_agentic.py",
        "tests/test_audio.py",
        "tests/test_reasoning.py",
        "tests/test_vlm.py",
    ):
        assert modality_test in content
    assert "scripts/generate_modality_baseline.py" in content
    assert "scripts/run_ops_module_matrix.py" in content
    assert "scripts/run_ops_e2e_reliability.py" in content
    assert "scripts/run_ops_dataset_burnin.py" in content
    assert "scripts/run_all_module_matrix.py" in content
    assert "scripts/run_all_module_qualification.py" in content
    assert "scripts/run_all_module_bootstrap.py" in content
    assert "scripts/run_all_module_live_matrix.py" in content
    assert "--fixture-pack v1" in content
    assert "tests/baselines/modality_runtime_baseline.v1.json" in content
    assert "ops-readiness-reports" in content
    assert "ops_e2e_launch_reliability.v1.json" in content
    assert "ops_dataset_burnin.v1.json" in content
    assert "all_modules_readiness.v1.json" in content
    assert "all_module_qualification.v1.json" in content
    assert "all_module_bootstrap.v1.json" in content
    assert "all_module_live_execution.v1.json" in content

    nightly_workflow = Path(".github/workflows/nightly_ops_readiness.yml")
    assert nightly_workflow.exists()
    nightly_content = nightly_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_content
    assert "workflow_dispatch:" in nightly_content
    assert "scripts/run_ops_module_matrix.py" in nightly_content
    assert "--fixture-pack v1" in nightly_content
    assert "--strict" in nightly_content

    nightly_e2e_workflow = Path(".github/workflows/nightly_ops_e2e_reliability.yml")
    assert nightly_e2e_workflow.exists()
    nightly_e2e_content = nightly_e2e_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_e2e_content
    assert "workflow_dispatch:" in nightly_e2e_content
    assert "scripts/run_ops_e2e_reliability.py" in nightly_e2e_content
    assert "--fixture-pack v1" in nightly_e2e_content
    assert "--strict" in nightly_e2e_content

    nightly_burnin_workflow = Path(".github/workflows/nightly_ops_dataset_burnin.yml")
    assert nightly_burnin_workflow.exists()
    nightly_burnin_content = nightly_burnin_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_burnin_content
    assert "workflow_dispatch:" in nightly_burnin_content
    assert "scripts/run_ops_dataset_burnin.py" in nightly_burnin_content
    assert "--strict" in nightly_burnin_content
    assert "--compare-baseline" in nightly_burnin_content
    assert "tests/baselines/ops_dataset_burnin_baseline.v1.json" in nightly_burnin_content

    nightly_all_modules_workflow = Path(".github/workflows/nightly_all_module_readiness.yml")
    assert nightly_all_modules_workflow.exists()
    nightly_all_modules_content = nightly_all_modules_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_all_modules_content
    assert "workflow_dispatch:" in nightly_all_modules_content
    assert "scripts/run_all_module_matrix.py" in nightly_all_modules_content
    assert "--fixture-pack v1" in nightly_all_modules_content
    assert "--strict" in nightly_all_modules_content

    nightly_qualification_workflow = Path(".github/workflows/nightly_all_module_qualification.yml")
    assert nightly_qualification_workflow.exists()
    nightly_qualification_content = nightly_qualification_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_qualification_content
    assert "workflow_dispatch:" in nightly_qualification_content
    assert "scripts/run_all_module_qualification.py" in nightly_qualification_content
    assert "--qualification-profile fixture-v1" in nightly_qualification_content
    assert "--strict" in nightly_qualification_content
    assert "--compare-baseline" in nightly_qualification_content
    assert "tests/baselines/all_module_qualification_baseline.v1.json" in nightly_qualification_content

    nightly_live_workflow = Path(".github/workflows/nightly_all_module_live_execution.yml")
    assert nightly_live_workflow.exists()
    nightly_live_content = nightly_live_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_live_content
    assert "workflow_dispatch:" in nightly_live_content
    assert "scripts/run_all_module_live_matrix.py" in nightly_live_content
    assert "--live-profile live-smoke-v1" in nightly_live_content
    assert "--strict" in nightly_live_content

    # Phase 7K walkthroughs are local/operator flows only; no CI gating changes.
    assert "run_all_module_walkthroughs.py" not in content
    assert not Path(".github/workflows/nightly_all_module_walkthroughs.yml").exists()
    assert not Path(".github/workflows/nightly_walkthroughs.yml").exists()


def test_ci_readiness_steps_gate_the_build_or_document_the_exception():
    """Every readiness script in CI must gate the build or say why it does not.

    Without --strict these steps print `status=fail` and exit 0, so CI stays
    green over a broken tree. Where the tree genuinely is not ready, the
    exception has to be visible in the diff as a TODO(ci-strict) comment
    naming the failing modules rather than being silently unguarded.
    """
    content = _ci_workflow_text()
    blocks = _ci_step_blocks(content)

    for script in READINESS_SCRIPTS:
        owning = [block for block in blocks if script in block]
        assert owning, f"{script} is referenced by no CI step"
        assert any(
            "--strict" in block or "TODO(ci-strict)" in block for block in owning
        ), (
            f"{script} runs in CI without --strict and without a documented "
            "TODO(ci-strict) exception, so it cannot turn the build red"
        )

    assert content.count("--strict") >= 4, (
        "expected the readiness steps that pass today to gate the build"
    )


def test_ci_trainer_regression_job_installs_torch_and_runs_the_whole_suite():
    """A job must install the training stack and run all of tests/.

    The compile-and-core-regression job installs pytest only, which turns
    every `pytest.importorskip("torch")` into a silent skip and leaves the
    trainers untested. Selection must be whole-directory (with an explicit
    deny-list at most), never a hand-maintained allow-list of files.
    """
    content = _ci_workflow_text()
    job = _ci_job_block(content, "trainer-regression")

    assert "https://download.pytorch.org/whl/cpu" in job
    for dependency in ("torch", "transformers", "peft", "datasets", "trl"):
        assert dependency in job, f"trainer-regression does not install {dependency}"
    assert re.search(r"^\s*timeout-minutes:\s*\d+", job, re.M), (
        "trainer-regression needs a timeout"
    )
    assert re.search(r"pytest\s+-q\s+tests/(\s|\\|$)", job, re.M), (
        "trainer-regression must select the whole tests/ directory"
    )


def test_ci_lint_job_gates_ruff_and_documents_the_formatter_ratchet():
    """Lint must gate on ruff; formatters may lag but must stay visible."""
    content = _ci_workflow_text()
    job = _ci_job_block(content, "lint")

    # `continue-on-error` sits on its own line, so a per-line filter would call
    # every step gating. Judge each step block as a whole instead.
    gating = [
        block
        for block in _ci_step_blocks(job)
        if "ruff check" in block and "continue-on-error" not in block
    ]
    assert gating, "lint job has no gating ruff check"
    assert "black --check" in job
    assert "isort --check-only" in job
    # Formatter drift is huge on this tree; it must be non-gating for now.
    assert "continue-on-error: true" in job


def test_ci_workflow_parses_as_yaml_with_named_gating_jobs():
    """The workflow must be valid YAML and define the gating jobs by name."""
    yaml = pytest.importorskip("yaml")

    document = yaml.safe_load(_ci_workflow_text())
    jobs = document["jobs"]
    for job_name in ("compile-and-core-regression", "trainer-regression", "lint"):
        assert job_name in jobs, f"CI workflow defines no `{job_name}` job"
        assert isinstance(jobs[job_name]["timeout-minutes"], int)
        assert jobs[job_name]["steps"]


def test_heavy_dependency_skip_guards_are_backed_by_a_job_that_installs_them():
    """importorskip guards are acceptable only if some CI job installs the deps.

    This replaces an older guard that asserted the modality tests keep their
    `pytest.importorskip` markers and nothing else. On a CI that never
    installs torch those markers silently disabled every trainer test, so the
    marker is not the contract worth enforcing; the contract is that a job
    exists which installs the heavy dependency and executes the file.
    """
    guarded = {
        "tests/test_inference.py": "torch",
        "tests/test_agentic.py": "torch",
        "tests/test_audio.py": "numpy",
        "tests/test_reasoning.py": "numpy",
        "tests/test_vlm.py": "numpy",
    }
    content = _ci_workflow_text()
    job = _ci_job_block(content, "trainer-regression")
    light_job = _ci_job_block(content, "compile-and-core-regression")

    for rel_path, module in guarded.items():
        path = Path(rel_path)
        assert path.exists(), rel_path
        # The marker itself is still load-bearing: compile-and-core-regression
        # runs these same files with pytest only, so dropping the guard turns
        # that job red at collection time. Guard + a job that installs the
        # dependency are both required; neither alone is sufficient.
        if rel_path in light_job:
            assert f'pytest.importorskip("{module}")' in path.read_text(encoding="utf-8"), (
                f"{rel_path} runs in compile-and-core-regression, which does not "
                f"install {module}, so it must keep its importorskip guard"
            )
        assert _job_installs(job, module), (
            f"{rel_path} skips itself without {module}, but no CI job installs it"
        )
        assert f"--ignore={rel_path}" not in job, (
            f"{rel_path} is guarded by importorskip and excluded from CI"
        )


def test_release_surface_excludes_legacy_tui_and_textual_dependency():
    """Legacy TUI should not be part of tracked source or active dependency contracts."""
    if not shutil.which("git"):
        return

    tracked_tui = subprocess.run(
        ["git", "ls-files", "halo_forge/tui"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_tui.returncode == 0
    assert tracked_tui.stdout.strip() == ""

    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("textual" in line for line in active_lines)

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    assert "textual" not in pyproject
    assert "halo_forge.tui" not in pyproject
