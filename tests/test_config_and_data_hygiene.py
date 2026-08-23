#!/usr/bin/env python3
"""Regression tests for config strictness, dependency hygiene, and run-db indexes.

These cover four independent defects:

  1. ``SFTConfig.from_yaml`` silently dropped unrecognized YAML keys, so a typo
     in ``learning_rate`` trained a full run at the default instead.
  2. ``SFTConfig.target_modules`` was annotated ``List[str]`` with a ``None``
     default — a type error any checker flags.
  3. ``matplotlib`` was a required core dependency while nothing in the
     importable packages uses it.
  4. ``runs.output_dir`` was unindexed despite being probed once per summary
     file on the filesystem-to-DB sync path.
"""

from __future__ import annotations

import gc
import re
import sqlite3
import sys
import tomllib
import typing
import weakref
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, List

import pytest

from halo_forge.data.dedup import fuzzy_dedup
from halo_forge.sft.config import SFTConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- 1. strict from_yaml ---------------------------------------------


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_from_yaml_loads_a_valid_nested_config(tmp_path: Path) -> None:
    """The happy path must keep working, including the key remapping."""
    cfg_path = tmp_path / "sft.yaml"
    _write(
        cfg_path,
        """
model:
  name: Qwen/Qwen2.5-Coder-1.5B
data:
  train_file: data/train.jsonl
  max_seq_length: 4096
lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: [q_proj, k_proj]
training:
  output_dir: models/sft
  per_device_train_batch_size: 4
  num_train_epochs: 2
  learning_rate: 0.0002
""",
    )

    cfg = SFTConfig.from_yaml(str(cfg_path))
    assert cfg.model_name == "Qwen/Qwen2.5-Coder-1.5B"
    assert cfg.train_file == "data/train.jsonl"
    assert cfg.max_seq_length == 4096
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
    assert cfg.target_modules == ["q_proj", "k_proj"]
    assert cfg.batch_size == 4
    assert cfg.num_epochs == 2
    assert cfg.learning_rate == 0.0002


def test_from_yaml_rejects_a_typoed_key(tmp_path: Path) -> None:
    """A typo must fail loudly instead of silently training at the default."""
    cfg_path = tmp_path / "sft.yaml"
    _write(
        cfg_path,
        """
model:
  name: Qwen/Qwen2.5-Coder-1.5B
data:
  train_file: data/train.jsonl
training:
  output_dir: models/sft
  learnign_rate: 0.00002
""",
    )

    with pytest.raises(ValueError) as excinfo:
        SFTConfig.from_yaml(str(cfg_path))

    message = str(excinfo.value)
    assert "learnign_rate" in message
    # The closest valid field must be suggested — that is the whole point.
    assert "learning_rate" in message


def test_from_yaml_reports_every_unknown_key_at_once(tmp_path: Path) -> None:
    """Fixing one typo should not require a second run to find the next."""
    cfg_path = tmp_path / "sft.yaml"
    _write(
        cfg_path,
        """
model:
  name: Qwen/Qwen2.5-Coder-1.5B
lora:
  rnak: 16
training:
  totally_bogus_key: 1
""",
    )

    with pytest.raises(ValueError) as excinfo:
        SFTConfig.from_yaml(str(cfg_path))

    message = str(excinfo.value)
    assert "rnak" in message
    assert "totally_bogus_key" in message


def test_from_yaml_accepts_the_shipped_example_config() -> None:
    """configs/sft_example.yaml ships HF TrainingArguments keys; it must load."""
    example = REPO_ROOT / "configs" / "sft_example.yaml"
    assert example.exists()

    cfg = SFTConfig.from_yaml(str(example))
    assert cfg.model_name == "Qwen/Qwen2.5-Coder-7B"
    assert cfg.batch_size == 2
    assert cfg.num_epochs == 3
    assert cfg.output_dir == "models/sft"


def test_from_yaml_rejects_a_typo_of_a_passthrough_key(tmp_path: Path) -> None:
    """The passthrough allowlist must not become a wildcard."""
    cfg_path = tmp_path / "sft.yaml"
    _write(
        cfg_path,
        """
training:
  loging_steps: 10
""",
    )

    with pytest.raises(ValueError) as excinfo:
        SFTConfig.from_yaml(str(cfg_path))
    assert "loging_steps" in str(excinfo.value)


def test_from_yaml_rejects_a_stray_top_level_key(tmp_path: Path) -> None:
    """A hyperparameter written outside every section is dropped, not applied."""
    cfg_path = tmp_path / "sft.yaml"
    _write(
        cfg_path,
        """
model:
  name: Qwen/Qwen2.5-Coder-1.5B
learning_rate: 0.00002
""",
    )

    with pytest.raises(ValueError) as excinfo:
        SFTConfig.from_yaml(str(cfg_path))

    message = str(excinfo.value)
    assert "learning_rate" in message
    # The message must say where the key belongs, not just that it is unknown.
    assert "training" in message


def test_from_yaml_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    """A YAML file that is not a mapping silently produced an all-default run."""
    cfg_path = tmp_path / "sft.yaml"
    _write(cfg_path, "aeyb werbysea es\n")

    with pytest.raises(ValueError):
        SFTConfig.from_yaml(str(cfg_path))


# ---------- 2. target_modules annotation ------------------------------------


def test_target_modules_is_annotated_optional_with_a_none_default() -> None:
    """``List[str] = None`` is a type error; the default must stay non-mutable."""
    hints = typing.get_type_hints(SFTConfig)
    annotation = hints["target_modules"]
    assert type(None) in typing.get_args(annotation), (
        f"target_modules is annotated {annotation!r}; None is not permitted by it "
        "even though None is the declared default"
    )
    assert List[str] in typing.get_args(annotation)

    field = {f.name: f for f in dataclass_fields(SFTConfig)}["target_modules"]
    assert field.default is None
    assert field.default_factory is __import__("dataclasses").MISSING


def test_target_modules_default_is_not_shared_between_instances() -> None:
    """Two configs must not alias one list (the mutable-default trap)."""
    a = SFTConfig()
    b = SFTConfig()
    assert a.target_modules == b.target_modules
    assert a.target_modules is not b.target_modules
    a.target_modules.append("sentinel")
    assert "sentinel" not in b.target_modules


# ---------- 3. dependency hygiene -------------------------------------------


def _core_dependency_names() -> set[str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        re.match(r"\s*([A-Za-z0-9_.-]+)", spec).group(1).lower().replace("_", "-")
        for spec in pyproject["project"]["dependencies"]
    }


def test_matplotlib_is_not_a_required_core_dependency() -> None:
    """Nothing in halo_forge/ or ui/ imports matplotlib; only scripts/ does, lazily."""
    assert "matplotlib" not in _core_dependency_names()


def test_matplotlib_remains_installable_as_a_declared_extra() -> None:
    """`halo-forge plot training` needs it, so it must stay reachable."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert "matplotlib" in extras.get("plot", [])
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    active = [
        line.split("#", 1)[0].strip()
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    ]
    assert "matplotlib" in active


def test_pandas_and_tensorboard_remain_core_dependencies() -> None:
    """Both are genuinely imported (lazily) by halo_forge — do not drop them."""
    core = _core_dependency_names()
    assert "pandas" in core
    assert "tensorboard" in core


# ---------- 4. fuzzy_dedup signature retention ------------------------------


class _StubMinHash:
    """Minimal MinHash stand-in; `datasketch` is not a test dependency."""

    def __init__(self, num_perm: int = 128) -> None:
        self.num_perm = num_perm
        self.tokens: set[bytes] = set()
        _LIVE_SIGNATURES.append(weakref.ref(self))

    def update(self, token: bytes) -> None:
        self.tokens.add(token)


class _StubMinHashLSH:
    """Stores derived band data, never the signature object — like the real LSH."""

    def __init__(self, threshold: float = 0.9, num_perm: int = 128) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self._entries: dict[str, frozenset[bytes]] = {}

    def query(self, minhash: _StubMinHash) -> List[str]:
        _PEAK_LIVE.append(_live_signature_count())
        hits = []
        for key, tokens in self._entries.items():
            union = tokens | minhash.tokens
            if not union:
                continue
            if len(tokens & minhash.tokens) / len(union) >= self.threshold:
                hits.append(key)
        return hits

    def insert(self, key: str, minhash: _StubMinHash) -> None:
        self._entries[key] = frozenset(minhash.tokens)


_LIVE_SIGNATURES: List[weakref.ref] = []
_PEAK_LIVE: List[int] = []


def _live_signature_count() -> int:
    gc.collect()
    return sum(1 for ref in _LIVE_SIGNATURES if ref() is not None)


@pytest.fixture()
def stub_datasketch(monkeypatch: pytest.MonkeyPatch) -> Any:
    import types

    module = types.ModuleType("datasketch")
    module.MinHash = _StubMinHash  # type: ignore[attr-defined]
    module.MinHashLSH = _StubMinHashLSH  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasketch", module)
    _LIVE_SIGNATURES.clear()
    _PEAK_LIVE.clear()
    yield module
    _LIVE_SIGNATURES.clear()
    _PEAK_LIVE.clear()


def test_fuzzy_dedup_still_removes_near_duplicates(stub_datasketch: Any) -> None:
    records = [
        {"text": "the quick brown fox jumps over the lazy dog"},
        {"text": "the quick brown fox jumps over the lazy dog"},
        {"text": "a completely different sentence about boats and water today"},
    ]
    result = fuzzy_dedup(records, threshold=0.85, shingle_n=3)
    assert result.kept_indices == [0, 2]
    assert result.removed_indices == [1]
    assert result.n_input == 3
    assert result.n_output == 2


def test_fuzzy_dedup_does_not_retain_every_signature(stub_datasketch: Any) -> None:
    """The signature list was write-only — ~1 GB of live MinHash at 1M records."""
    records = [
        {"text": f"shared preamble text number {i % 2} for the dedup corpus"}
        for i in range(200)
    ]

    result = fuzzy_dedup(records, threshold=0.85, shingle_n=3)
    assert result.n_input == 200
    assert result.n_output == 2

    peak = max(_PEAK_LIVE)
    assert peak <= 8, (
        f"fuzzy_dedup held {peak} MinHash signatures live at once for 200 records; "
        "the per-record signature list is never read again and must not be retained"
    )


# ---------- 5. runs.output_dir index + migration ----------------------------


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
            (table,),
        ).fetchall()
    }


def test_fresh_database_indexes_runs_output_dir(tmp_path: Path) -> None:
    from halo_forge.run_db.db import RunDatabase

    db = RunDatabase(str(tmp_path / "fresh.db"))
    try:
        names = _index_names(db._conn, "runs")  # noqa: SLF001 — schema inspection
    finally:
        db.close()
    assert "idx_runs_output_dir" in names


def test_sync_lookup_uses_the_output_dir_index(tmp_path: Path) -> None:
    """The sync path probes output_dir once per summary file; it must not scan."""
    from halo_forge.run_db.db import RunDatabase

    db = RunDatabase(str(tmp_path / "plan.db"))
    plan = db._conn.execute(  # noqa: SLF001 — planner inspection
        "EXPLAIN QUERY PLAN SELECT source_mtime FROM runs WHERE output_dir = ?",
        ("models/whatever",),
    ).fetchall()
    detail = " ".join(str(row["detail"]) for row in plan)
    assert "idx_runs_output_dir" in detail, detail


def test_existing_database_gains_the_index_without_losing_rows(tmp_path: Path) -> None:
    """An installed v23 database must migrate in place, keeping its rows."""
    from halo_forge.run_db.db import RunDatabase, RunRecord

    path = tmp_path / "legacy.db"
    db = RunDatabase(str(path))
    db.upsert_run(
        RunRecord(
            run_id="run-legacy",
            fs_id="run-legacy",
            modality="sft",
            model_name="Qwen/Qwen2.5-Coder-1.5B",
            status="completed",
            timestamp="2026-01-01T00:00:00+00:00",
            output_dir="models/legacy",
            indexed_at="2026-01-01T00:00:00+00:00",
        )
    )
    db._conn.commit()  # noqa: SLF001

    # Roll the on-disk database back to the pre-fix shape.
    raw = sqlite3.connect(str(path))
    raw.execute("DROP INDEX IF EXISTS idx_runs_output_dir")
    raw.execute("UPDATE schema_meta SET value = '22' WHERE key = 'schema_version'")
    raw.commit()
    assert "idx_runs_output_dir" not in {
        str(row[0])
        for row in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='runs'"
        ).fetchall()
    }
    raw.close()

    reopened = RunDatabase(str(path))
    assert "idx_runs_output_dir" in _index_names(reopened._conn, "runs")  # noqa: SLF001
    row = reopened._conn.execute(  # noqa: SLF001
        "SELECT run_id, output_dir FROM runs WHERE output_dir = ?", ("models/legacy",)
    ).fetchone()
    assert row is not None
    assert row["run_id"] == "run-legacy"
