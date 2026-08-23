"""Managed runtime catalog, preparation, qualification, and bindings."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from halo_forge.run_db import RunDatabase

from .adapters import RUNTIME_EXECUTION_ADAPTERS, RuntimeMount
from .models import (
    AcceleratorPreflightDecision,
    ManagedRuntimeCapability,
    ManagedRuntimeProfile,
    ManagedRuntimeRevision,
    RuntimeBinding,
    RuntimePreparation,
    RuntimeQualification,
    RuntimeQualificationStep,
)
from .occupancy import probe_accelerator, wait_for_stable_idle


ADAPTER_VERSION = "1"
GIB = 1024**3
DEPENDENCY_LOCK = {
    "transformers": "5.13.1",
    "peft": "0.19.1",
    "datasets": "3.6.0",
    "trl": "0.29.1",
    "accelerate": "1.14.0",
}
TRAINER_CONTRACTS = (
    "sft", "chat", "tool", "cpt", "dpo", "orpo", "rm", "raft", "grpo",
    "reasoning", "agentic", "vlm", "audio", "classify", "embed", "rerank",
    "image_classify", "audio_classify",
)
BUILTIN_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "strix-halo-rocm-7.2.1",
        "name": "AMD Strix Halo training",
        "accelerator_family": "rocm",
        "description": "Pinned ROCm 7.2.1 / PyTorch 2.9.1 runtime for gfx1151 workstations.",
        "adapter_id": "podman_rocm",
        "engine": "podman",
        "base_image": "docker.io/rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1",
        "base_image_digest": "sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191",
        "download_bytes": int(10.7 * GIB),
        "installed_bytes": 31 * GIB,
        "configuration": {
            "host_support": {"vendor": ["ubuntu:24.04"], "local_verification_allowed": True},
            "device_mounts": ["/dev/kfd", "/dev/dri"],
            "ipc": "host",
            "privileged": False,
        },
    },
    {
        "id": "nvidia-cuda-12.8",
        "name": "NVIDIA CUDA training",
        "accelerator_family": "cuda",
        "description": "Pinned CUDA 12.8 / PyTorch 2.9.1 runtime with CDI device access.",
        "adapter_id": "podman_cuda",
        "engine": "podman",
        "base_image": "docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime",
        "base_image_digest": "sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385",
        "download_bytes": None,
        "installed_bytes": None,
        "configuration": {
            "engine_preference": ["podman", "docker"],
            "device_interface": "nvidia-cdi",
            "privileged": False,
            "hardware_release_gate": True,
        },
    },
)


class ManagedRuntimeError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value not in (None, "") else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hash(root: Path) -> str:
    """Fingerprint the application code that is baked into a derived image.

    A base-image digest and dependency lock are not enough to identify the
    product runtime: two Halo Forge checkouts could otherwise publish the same
    runtime revision while containing different trainer code.
    """

    digest = hashlib.sha256()
    candidates: list[Path] = []
    for name in ("pyproject.toml", "requirements.txt"):
        candidate = root / name
        if candidate.is_file():
            candidates.append(candidate)
    package = root / "halo_forge"
    if package.is_dir():
        candidates.extend(
            path
            for path in package.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        )
    for path in sorted(candidates, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _identifier(prefix: str, value: Any) -> str:
    return f"{prefix}-{_hash(value)[:24]}"


def _sanitized_environment() -> dict[str, str]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "XDG_RUNTIME_DIR", "TMPDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _default_runner(
    argv: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[Mapping[str, str]] = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd else None,
        env={**_sanitized_environment(), **dict(env or {})},
        check=False,
        capture_output=True,
        text=True,
        timeout=None,
        shell=False,
    )


class ManagedRuntimeService:
    def __init__(
        self,
        database: RunDatabase,
        *,
        root: str | Path | None = None,
        scheduler: Any = None,
        source_root: str | Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = _default_runner,
        occupancy_probe: Callable[[str], Any] = probe_accelerator,
    ) -> None:
        self.database = database
        self.conn = database._conn
        configured = os.environ.get("HALOFORGE_RUNTIME_ROOT")
        self.root = Path(root or configured or (Path.home() / ".halo-forge" / "runtimes")).expanduser().resolve()
        self.source_root = Path(source_root or Path(__file__).parents[2]).resolve()
        self.source_hash = _source_tree_hash(self.source_root)
        self.scheduler = scheduler
        self.runner = runner
        self.occupancy_probe = occupancy_probe
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_builtin_profiles()

    # ---- immutable profile registry ---------------------------------

    def _ensure_builtin_profiles(self) -> None:
        now = _now()
        for value in BUILTIN_PROFILES:
            identity = {
                "adapter_id": value["adapter_id"],
                "adapter_version": ADAPTER_VERSION,
                "engine": value["engine"],
                "base_image": value["base_image"],
                "base_image_digest": value["base_image_digest"],
                "dependency_lock": DEPENDENCY_LOCK,
                "halo_forge_source_hash": self.source_hash,
                "configuration": value["configuration"],
                "trainer_contracts": TRAINER_CONTRACTS,
            }
            content_hash = _hash(identity)
            revision_id = f"runtime-{value['id']}-{content_hash[:12]}"
            derived_ref = f"localhost/halo-forge-{value['accelerator_family']}:{content_hash[:12]}"
            with self.database._lock:
                self.conn.execute(
                    """INSERT OR IGNORE INTO managed_runtime_profiles
                       (id,name,accelerator_family,description,latest_revision_id,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (value["id"], value["name"], value["accelerator_family"], value["description"], revision_id, now, now),
                )
                self.conn.execute(
                    """INSERT OR IGNORE INTO managed_runtime_revisions
                       (id,profile_id,revision_number,content_hash,adapter_id,adapter_version,
                        engine,base_image,base_image_digest,derived_image_ref,
                        dependency_lock_json,configuration_json,trainer_contracts_json,
                        download_bytes,installed_bytes,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (revision_id, value["id"], 1, content_hash, value["adapter_id"], ADAPTER_VERSION,
                     value["engine"], value["base_image"], value["base_image_digest"], derived_ref,
                     _json(DEPENDENCY_LOCK), _json(value["configuration"]), _json(TRAINER_CONTRACTS),
                     value["download_bytes"], value["installed_bytes"], now),
                )
                self.conn.execute(
                    "UPDATE managed_runtime_profiles SET latest_revision_id=?,updated_at=? WHERE id=?",
                    (revision_id, now, value["id"]),
                )
                self.conn.commit()

    def list_profiles(self) -> tuple[ManagedRuntimeProfile, ...]:
        rows = self.conn.execute("SELECT * FROM managed_runtime_profiles ORDER BY name").fetchall()
        return tuple(self._profile(row) for row in rows)

    def get_profile(self, profile_id: str) -> Optional[ManagedRuntimeProfile]:
        row = self.conn.execute("SELECT * FROM managed_runtime_profiles WHERE id=?", (profile_id,)).fetchone()
        return self._profile(row) if row else None

    def list_revisions(self, profile_id: Optional[str] = None) -> tuple[ManagedRuntimeRevision, ...]:
        if profile_id:
            rows = self.conn.execute("SELECT * FROM managed_runtime_revisions WHERE profile_id=? ORDER BY revision_number DESC", (profile_id,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM managed_runtime_revisions ORDER BY created_at DESC").fetchall()
        return tuple(self._revision(row) for row in rows)

    def get_revision(self, revision_id: str) -> Optional[ManagedRuntimeRevision]:
        row = self.conn.execute("SELECT * FROM managed_runtime_revisions WHERE id=?", (revision_id,)).fetchone()
        return self._revision(row) if row else None

    # ---- engine, storage, and preparation ----------------------------

    def _engine(self, revision: ManagedRuntimeRevision) -> tuple[str, Any]:
        if revision.configuration.get("engine_preference"):
            for engine in revision.configuration["engine_preference"]:
                adapter_id = f"{engine}_{'cuda' if revision.profile_id == 'nvidia-cuda-12.8' else 'rocm'}"
                try:
                    adapter = RUNTIME_EXECUTION_ADAPTERS.get(adapter_id)
                except KeyError:
                    continue
                available, _ = adapter.available()
                if available:
                    return str(engine), adapter
            raise ManagedRuntimeError("No rootless CUDA container engine can access NVIDIA CDI")
        adapter = RUNTIME_EXECUTION_ADAPTERS.get(revision.adapter_id)
        available, reason = adapter.available()
        if not available:
            raise ManagedRuntimeError(reason or f"{revision.engine} is unavailable")
        return revision.engine, adapter

    def _podman_prefix(self, engine: str) -> list[str]:
        if engine != "podman":
            return [engine]
        image_root = self.root / "images"
        run_root = self.root / "run"
        image_root.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)
        return [engine, "--root", str(image_root), "--runroot", str(run_root)]

    def _engine_argv(self, engine: str, argv: Sequence[str]) -> list[str]:
        return [*self._podman_prefix(engine), *list(argv)]

    def prepare(self, revision_id: str, *, confirmed: bool, enqueue: bool = True) -> RuntimePreparation:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        if not confirmed:
            raise ManagedRuntimeError("Runtime preparation requires confirmation of the download and installed-size forecast")
        existing = self.conn.execute(
            "SELECT * FROM runtime_preparations WHERE runtime_revision_id=? AND status='completed' ORDER BY created_at DESC LIMIT 1",
            (revision_id,),
        ).fetchone()
        if existing:
            return self._preparation(existing)
        active = self.conn.execute(
            "SELECT * FROM runtime_preparations WHERE runtime_revision_id=? "
            "AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1",
            (revision_id,),
        ).fetchone()
        if active:
            return self._preparation(active)
        engine, _ = self._engine(revision)
        now = _now()
        preparation_id = _identifier("runtime-preparation", {"revision": revision_id, "created": now})
        self.conn.execute(
            """INSERT INTO runtime_preparations
               (id,runtime_revision_id,status,stage,engine,storage_path,progress_json,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (preparation_id, revision_id, "queued" if enqueue else "running", "queued", engine, str(self.root), _json({"download_bytes": revision.download_bytes, "installed_bytes": revision.installed_bytes}), now),
        )
        self.conn.commit()
        if enqueue:
            if self.scheduler is None:
                from halo_forge.workstation_jobs import WorkstationScheduler
                self.scheduler = WorkstationScheduler(self.database)
            item = self.scheduler.enqueue(
                kind="runtime_prepare",
                launch_spec={"handler": "managed_runtime.execute_work_item", "operation": "prepare", "runtime_root": str(self.root), "source_root": str(self.source_root)},
                resource_class="cpu",
                resource_requirements={"projected_disk_bytes": int(revision.installed_bytes or 0), "output_path": str(self.root)},
                domain_kind="runtime_preparation", domain_id=preparation_id, max_retries=2,
            )
            self.conn.execute("UPDATE runtime_preparations SET work_item_id=? WHERE id=?", (item.id, preparation_id))
            self.conn.commit()
        return self.get_preparation(preparation_id)  # type: ignore[return-value]

    def run_preparation(self, preparation_id: str) -> RuntimePreparation:
        record = self.get_preparation(preparation_id)
        if record is None:
            raise KeyError(preparation_id)
        revision = self.get_revision(record.runtime_revision_id)
        assert revision is not None
        engine, _ = self._engine(revision)
        self._update_preparation(preparation_id, status="running", stage="pulling")
        pinned_base = f"{revision.base_image}@{revision.base_image_digest}"
        pull = self.runner(self._engine_argv(engine, ("pull", pinned_base)))
        if pull.returncode:
            return self._fail_preparation(preparation_id, "pull", pull.stderr or pull.stdout)
        build_dir = self.root / "build" / preparation_id
        build_dir.mkdir(parents=True, exist_ok=True)
        wheel = self.runner(
            (sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(build_dir)),
            cwd=self.source_root,
        )
        if wheel.returncode:
            return self._fail_preparation(preparation_id, "building_wheel", wheel.stderr or wheel.stdout)
        wheels = sorted(build_dir.glob("halo_forge*.whl"))
        if not wheels:
            return self._fail_preparation(preparation_id, "building_wheel", "Halo Forge wheel was not produced")
        containerfile = build_dir / "Containerfile"
        lock_args = " ".join(f"{name}=={version}" for name, version in sorted(revision.dependency_lock.items()))
        containerfile.write_text(
            f"FROM {pinned_base}\n"
            "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
            f"RUN python -m pip install --no-cache-dir {lock_args}\n"
            f"COPY {wheels[0].name} /tmp/halo_forge.whl\n"
            "RUN python -m pip install --no-cache-dir --no-deps /tmp/halo_forge.whl && rm /tmp/halo_forge.whl\n",
            encoding="utf-8",
        )
        self._update_preparation(preparation_id, stage="building", progress={"base_pulled": True, "wheel": wheels[0].name})
        build = self.runner(self._engine_argv(engine, ("build", "--pull=never", "--file", str(containerfile), "--tag", str(revision.derived_image_ref), str(build_dir))))
        if build.returncode:
            return self._fail_preparation(preparation_id, "building", build.stderr or build.stdout)
        inspect = self.runner(self._engine_argv(engine, ("image", "inspect", str(revision.derived_image_ref), "--format", "{{.Id}}")))
        if inspect.returncode or not inspect.stdout.strip():
            return self._fail_preparation(preparation_id, "verifying", inspect.stderr or "built image could not be inspected")
        image_id = inspect.stdout.strip().splitlines()[-1]
        manifest = {
            "runtime_revision_id": revision.id,
            "runtime_content_hash": revision.content_hash,
            "base_image": revision.base_image,
            "base_image_digest": revision.base_image_digest,
            "derived_image_ref": revision.derived_image_ref,
            "image_id": image_id,
            "dependency_lock": revision.dependency_lock,
            "halo_forge_source_hash": self.source_hash,
            "engine": engine,
            "storage_root": str(self.root),
        }
        manifest_dir = self.root / "manifests" / revision.content_hash
        manifest_dir.mkdir(parents=True, exist_ok=True)
        target = manifest_dir / "preparation.json"
        with tempfile.NamedTemporaryFile("w", dir=manifest_dir, delete=False, encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True))
            temporary = Path(handle.name)
        os.replace(temporary, target)
        now = _now()
        self.conn.execute(
            """UPDATE runtime_preparations SET status='completed',stage='completed',image_id=?,
               image_digest=?,manifest_path=?,manifest_hash=?,progress_json=?,completed_at=? WHERE id=?""",
            (image_id, revision.base_image_digest, str(target), _file_hash(target), _json({"base_pulled": True, "derived_built": True, "verified": True}), now, preparation_id),
        )
        self.conn.commit()
        return self.get_preparation(preparation_id)  # type: ignore[return-value]

    # ---- qualification -----------------------------------------------

    @staticmethod
    def _read_identity_file(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _host_identity(
        self,
        family: str,
        *,
        revision: Optional[ManagedRuntimeRevision] = None,
        image_id: Optional[str] = None,
    ) -> tuple[str, str, str, dict[str, Any]]:
        release: dict[str, str] = {}
        try:
            for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    release[key.lower()] = value.strip().strip('"')
        except OSError:
            pass
        host = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "os": release}
        drm_devices: list[dict[str, Optional[str]]] = []
        for entry in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            drm_devices.append(
                {
                    "path": entry.parent.name,
                    "vendor": self._read_identity_file(entry / "vendor"),
                    "device": self._read_identity_file(entry / "device"),
                    "revision": self._read_identity_file(entry / "revision"),
                }
            )
        driver = {
            "amdgpu": self._read_identity_file(Path("/sys/module/amdgpu/version")),
            "nvidia": self._read_identity_file(Path("/sys/module/nvidia/version")),
        }
        device = {
            "family": family,
            "nodes": {
                path: Path(path).exists()
                for path in ("/dev/kfd", "/dev/dri", "/dev/nvidia0", "/dev/nvidiactl")
            },
            "drm_devices": drm_devices,
            "driver": driver,
        }
        runtime = {
            "kernel": platform.release(),
            "python": platform.python_version(),
            "family": family,
            "host": host,
            "device": device,
            "runtime_revision_hash": revision.content_hash if revision else None,
            "derived_image_id": image_id,
        }
        return _hash(host), _hash(device), _hash(runtime), runtime

    def qualify(self, revision_id: str, *, enqueue: bool = True) -> RuntimeQualification:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        preparation = self.conn.execute("SELECT * FROM runtime_preparations WHERE runtime_revision_id=? AND status='completed' ORDER BY created_at DESC LIMIT 1", (revision_id,)).fetchone()
        if preparation is None:
            raise ManagedRuntimeError("Prepare and verify this runtime before qualification")
        current = self.conn.execute(
            "SELECT * FROM runtime_qualifications WHERE runtime_revision_id=? "
            "AND status IN ('queued','running','vendor_supported','local_verified') "
            "ORDER BY created_at DESC LIMIT 1",
            (revision_id,),
        ).fetchone()
        if current is not None:
            value = self._qualification(current)
            if value.status in {"queued", "running"} or self.verify(value.id)["valid"]:
                return value
        family = self.get_profile(revision.profile_id).accelerator_family  # type: ignore[union-attr]
        host_hash, device_hash, runtime_hash, _ = self._host_identity(
            family, revision=revision, image_id=preparation["image_id"]
        )
        now = _now()
        qualification_id = _identifier("runtime-qualification", {"revision": revision_id, "runtime": runtime_hash, "created": now})
        self.conn.execute(
            """INSERT INTO runtime_qualifications
               (id,runtime_revision_id,preparation_id,status,stage,host_identity_hash,
                device_identity_hash,runtime_identity_hash,progress_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (qualification_id, revision_id, preparation["id"], "queued" if enqueue else "running", "queued", host_hash, device_hash, runtime_hash, _json({}), now),
        )
        for ordinal, (step_id, label, _) in enumerate(self._qualification_ladder(revision), 1):
            self.conn.execute("INSERT INTO runtime_qualification_steps (qualification_id,ordinal,step_id,label,status) VALUES (?,?,?,?, 'pending')", (qualification_id, ordinal, step_id, label))
        self.conn.commit()
        if enqueue:
            if self.scheduler is None:
                from halo_forge.workstation_jobs import WorkstationScheduler
                self.scheduler = WorkstationScheduler(self.database)
            item = self.scheduler.enqueue(
                kind="runtime_qualify",
                launch_spec={"handler": "managed_runtime.execute_work_item", "operation": "qualify", "runtime_root": str(self.root), "source_root": str(self.source_root), "runtime_profile_revision_id": revision.id},
                resource_class="accelerator",
                resource_requirements={"accelerator_family": family, "runtime_profile_revision_id": revision.id, "output_path": str(self.root)},
                domain_kind="runtime_qualification", domain_id=qualification_id, max_retries=1,
            )
            self.conn.execute("UPDATE runtime_qualifications SET work_item_id=? WHERE id=?", (item.id, qualification_id))
            self.conn.commit()
        return self.get_qualification(qualification_id)  # type: ignore[return-value]

    def _qualification_ladder(self, revision: ManagedRuntimeRevision) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        # V21 core qualification stops at the generic optimizer operation.
        # Hand-written Qwen and per-trainer tensor probes remain callable as
        # diagnostics, but only TrainingPathCertificationService can unlock a
        # guided trainer after running Dataset Lab + the shipped entrypoint.
        values: list[tuple[str, str, tuple[str, ...]]] = [
            ("engine_device", "Engine and device access", ("python", "-m", "halo_forge.managed_runtime.qualify", "enumerate")),
            ("dependency_integrity", "Dependencies and image integrity", ("python", "-m", "halo_forge.managed_runtime.qualify", "dependencies")),
            ("gpu_enumeration", "GPU enumeration", ("python", "-m", "halo_forge.managed_runtime.qualify", "enumerate")),
            ("fp32_bf16", "FP32 allocation and BF16 matmul", ("python", "-m", "halo_forge.managed_runtime.qualify", "kernels")),
            ("optimizer_step", "Backward and AdamW optimizer step", ("python", "-m", "halo_forge.managed_runtime.qualify", "optimizer")),
        ]
        return tuple(values)

    def _container_argv(self, revision: ManagedRuntimeRevision, engine: str, adapter: Any, command: Sequence[str], evidence_dir: Path, name: str) -> list[str]:
        cache = self.root / "model-cache"
        cache.mkdir(parents=True, exist_ok=True)
        wrapped = adapter.wrap(
            command,
            image=revision.derived_image_ref,
            mounts=(RuntimeMount(evidence_dir, Path("/evidence")), RuntimeMount(cache, Path("/cache"))),
            env={"HF_HOME": "/cache", "TRANSFORMERS_CACHE": "/cache"},
            name=name,
        )
        values = list(wrapped.argv)
        if engine == "podman":
            values[1:1] = self._podman_prefix(engine)[1:]
        return values

    def run_qualification(self, qualification_id: str) -> RuntimeQualification:
        qualification = self.get_qualification(qualification_id)
        if qualification is None:
            raise KeyError(qualification_id)
        revision = self.get_revision(qualification.runtime_revision_id)
        assert revision is not None
        profile = self.get_profile(revision.profile_id)
        assert profile is not None
        stable, samples = wait_for_stable_idle(profile.accelerator_family, probe=self.occupancy_probe)
        self.record_preflight_decision(
            accelerator_family=profile.accelerator_family,
            decision="idle" if stable else ("unknown" if samples[-1].state == "unknown" else "waiting"),
            evidence={"samples": [sample.to_dict() for sample in samples]},
            work_item_id=qualification.work_item_id,
            runtime_revision_id=revision.id,
        )
        if not stable:
            reason = samples[-1].reason or "accelerator is not independently verified idle"
            self.conn.execute("UPDATE runtime_qualifications SET status='blocked',stage='waiting_for_accelerator',error=?,progress_json=? WHERE id=?", (reason, _json({"availability": samples[-1].to_dict()}), qualification_id))
            self.conn.commit()
            return self.get_qualification(qualification_id)  # type: ignore[return-value]
        engine, adapter = self._engine(revision)
        evidence_dir = self.root / "qualifications" / qualification_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.conn.execute("UPDATE runtime_qualifications SET status='running',stage='qualifying',error=NULL WHERE id=?", (qualification_id,))
        self.conn.commit()
        ladder = self._qualification_ladder(revision)
        evidence: list[dict[str, Any]] = []
        for ordinal, (step_id, label, command) in enumerate(ladder, 1):
            current = self.conn.execute("SELECT cancel_requested FROM runtime_qualifications WHERE id=?", (qualification_id,)).fetchone()
            if current and bool(current["cancel_requested"]):
                self.conn.execute("UPDATE runtime_qualifications SET status='cancelled',stage='cancelled',completed_at=? WHERE id=?", (_now(), qualification_id))
                self.conn.commit()
                return self.get_qualification(qualification_id)  # type: ignore[return-value]
            command_hash = _hash({"runtime": revision.content_hash, "command": list(command)})
            started = _now()
            self.conn.execute("UPDATE runtime_qualification_steps SET status='running',command_hash=?,started_at=? WHERE qualification_id=? AND ordinal=?", (command_hash, started, qualification_id, ordinal))
            self.conn.execute("UPDATE runtime_qualifications SET stage=?,progress_json=? WHERE id=?", (step_id, _json({"current": ordinal - 1, "total": len(ladder), "label": label}), qualification_id))
            self.conn.commit()
            # The scheduler sampled three times; close the final race directly
            # before the first child process.
            if ordinal == 1:
                immediate = self.occupancy_probe(profile.accelerator_family)
                if not immediate.idle:
                    self.record_preflight_decision(accelerator_family=profile.accelerator_family, decision="unknown" if immediate.state == "unknown" else "contention", evidence={"pre_spawn": immediate.to_dict()}, work_item_id=qualification.work_item_id, runtime_revision_id=revision.id)
                    self.conn.execute("UPDATE runtime_qualification_steps SET status='failed',result_json=?,completed_at=? WHERE qualification_id=? AND ordinal=?", (_json({"reason": immediate.reason, "state": immediate.state}), _now(), qualification_id, ordinal))
                    self.conn.execute("UPDATE runtime_qualifications SET status='blocked',stage='waiting_for_accelerator',error=? WHERE id=?", (immediate.reason or "accelerator became busy before launch", qualification_id))
                    self.conn.commit()
                    return self.get_qualification(qualification_id)  # type: ignore[return-value]
            argv = self._container_argv(revision, engine, adapter, command, evidence_dir, f"hf-qualify-{qualification_id[-10:]}-{ordinal}")
            result = self.runner(argv)
            log_path = evidence_dir / f"{ordinal:02d}-{step_id}.log"
            log_path.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
            parsed: dict[str, Any] = {"returncode": result.returncode}
            if result.returncode == 0:
                try:
                    parsed.update(json.loads((result.stdout or "{}").splitlines()[-1]))
                except (json.JSONDecodeError, IndexError):
                    parsed["output_verified"] = bool(result.stdout.strip())
            evidence.append({"step_id": step_id, "command_hash": command_hash, "result": parsed, "log_hash": _file_hash(log_path)})
            status = "passed" if result.returncode == 0 else "failed"
            self.conn.execute("UPDATE runtime_qualification_steps SET status=?,result_json=?,log_path=?,completed_at=? WHERE qualification_id=? AND ordinal=?", (status, _json(parsed), str(log_path), _now(), qualification_id, ordinal))
            self.conn.commit()
            if result.returncode:
                error = f"{label} failed; inspect the step log"
                self.conn.execute("UPDATE runtime_qualifications SET status='failed',stage=?,error=?,progress_json=?,completed_at=? WHERE id=?", (step_id, error, _json({"current": ordinal, "total": len(ladder), "failed_step": step_id}), _now(), qualification_id))
                self.conn.commit()
                return self.get_qualification(qualification_id)  # type: ignore[return-value]
        preparation = self.get_preparation(str(qualification.preparation_id))
        host_details = self._host_identity(
            profile.accelerator_family,
            revision=revision,
            image_id=preparation.image_id if preparation else None,
        )[3]
        os_info = dict(host_details.get("host", {}).get("os") or {})
        vendor = profile.accelerator_family == "rocm" and os_info.get("id") == "ubuntu" and os_info.get("version_id") == "24.04"
        # CUDA publication remains local-verified until real hardware evidence
        # exists for this exact revision; static tests can never create this row.
        status = "vendor_supported" if vendor else "local_verified"
        bundle = {"runtime_revision": revision.to_dict(), "host": host_details, "steps": evidence, "status": status}
        bundle_path = evidence_dir / "qualification.json"
        with tempfile.NamedTemporaryFile("w", dir=evidence_dir, delete=False, encoding="utf-8") as handle:
            handle.write(json.dumps(bundle, indent=2, sort_keys=True))
            temporary = Path(handle.name)
        os.replace(temporary, bundle_path)
        qualification_hash = _file_hash(bundle_path)
        self.conn.execute("UPDATE runtime_qualifications SET status=?,stage='completed',qualification_hash=?,evidence_path=?,progress_json=?,completed_at=? WHERE id=?", (status, qualification_hash, str(bundle_path), _json({"current": len(ladder), "total": len(ladder)}), _now(), qualification_id))
        self.conn.commit()
        return self.get_qualification(qualification_id)  # type: ignore[return-value]

    def verify(self, qualification_id: str) -> dict[str, Any]:
        value = self.get_qualification(qualification_id)
        if value is None:
            raise KeyError(qualification_id)
        issues: list[str] = []
        if not value.evidence_path or not Path(value.evidence_path).is_file():
            issues.append("qualification bundle is missing")
        elif value.qualification_hash != _file_hash(Path(value.evidence_path)):
            issues.append("qualification bundle checksum changed")
        revision = self.get_revision(value.runtime_revision_id)
        if revision is None:
            issues.append("runtime revision is missing")
        else:
            preparation = self.get_preparation(str(value.preparation_id)) if value.preparation_id else None
            host_hash, device_hash, runtime_hash, _ = self._host_identity(
                self.get_profile(revision.profile_id).accelerator_family,  # type: ignore[union-attr]
                revision=revision,
                image_id=preparation.image_id if preparation else None,
            )
            if (host_hash, device_hash, runtime_hash) != (value.host_identity_hash, value.device_identity_hash, value.runtime_identity_hash):
                issues.append("host, device, kernel, or runtime identity changed")
        return {"valid": not issues, "stale": any("changed" in issue for issue in issues), "issues": issues, "qualification": value.to_dict()}

    def capabilities(self) -> tuple[ManagedRuntimeCapability, ...]:
        values: list[ManagedRuntimeCapability] = []
        for profile in self.list_profiles():
            revision = self.get_revision(str(profile.latest_revision_id)) if profile.latest_revision_id else None
            qualification = None
            if revision:
                row = self.conn.execute("SELECT * FROM runtime_qualifications WHERE runtime_revision_id=? AND status IN ('vendor_supported','local_verified') ORDER BY completed_at DESC LIMIT 1", (revision.id,)).fetchone()
                qualification = self._qualification(row, include_steps=False) if row else None
                if qualification and not self.verify(qualification.id)["valid"]:
                    qualification = None
            values.append(ManagedRuntimeCapability(
                profile.accelerator_family,
                qualification is not None,
                qualification.status if qualification else "not_qualified",
                "Core accelerator runtime is ready; real training paths are verified separately" if qualification else "Runtime preparation and qualification are required",
                revision.id if revision else None,
                qualification.id if qualification else None,
                (),
                None if qualification else "Hardware detection alone does not prove a safe optimizer update and artifact reload",
            ))
        return tuple(values)

    # ---- occupancy decisions and immutable bindings ------------------

    def availability(self, family: str) -> Any:
        return self.occupancy_probe(family)

    def record_preflight_decision(self, *, accelerator_family: str, decision: str, evidence: Mapping[str, Any], work_item_id: Optional[str] = None, runtime_revision_id: Optional[str] = None, override_reason: Optional[str] = None) -> AcceleratorPreflightDecision:
        created = _now()
        evidence_hash = _hash(evidence)
        identifier = _identifier("accelerator-preflight", {"work": work_item_id, "created": created, "evidence": evidence_hash})
        sample_count = len(evidence.get("samples") or []) or 1
        self.conn.execute("""INSERT INTO accelerator_preflight_decisions
            (id,work_item_id,runtime_revision_id,accelerator_family,decision,sample_count,evidence_hash,evidence_json,override_reason,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (identifier, work_item_id, runtime_revision_id, accelerator_family, decision, sample_count, evidence_hash, _json(dict(evidence)), override_reason, created))
        self.conn.commit()
        return AcceleratorPreflightDecision(identifier, work_item_id, runtime_revision_id, accelerator_family, decision, sample_count, evidence_hash, dict(evidence), override_reason, created)

    def bind(self, *, revision_id: str, domain_kind: str, domain_id: str, qualification_id: Optional[str] = None, role: str = "execution", details: Optional[Mapping[str, Any]] = None) -> RuntimeBinding:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        qualification = self.get_qualification(qualification_id) if qualification_id else self.latest_qualification(revision_id)
        if qualification is None or qualification.status not in {"vendor_supported", "local_verified"} or not self.verify(qualification.id)["valid"]:
            raise ManagedRuntimeError("Only a current qualified runtime can be bound to managed execution")
        created = _now()
        identifier = _identifier("runtime-binding", {"domain": [domain_kind, domain_id, role], "revision": revision_id})
        existing = self.conn.execute(
            "SELECT * FROM runtime_bindings WHERE domain_kind=? AND domain_id=? AND role=?",
            (domain_kind, domain_id, role),
        ).fetchone()
        if existing is not None:
            if str(existing["runtime_revision_id"]) != revision_id:
                raise ManagedRuntimeError("The domain is already bound to a different immutable runtime")
            return RuntimeBinding(
                str(existing["id"]), str(existing["runtime_revision_id"]),
                existing["qualification_id"], str(existing["domain_kind"]),
                str(existing["domain_id"]), str(existing["role"]),
                str(existing["runtime_identity_hash"]),
                _loads(existing["details_json"], {}), str(existing["created_at"]),
            )
        self.conn.execute("INSERT INTO runtime_bindings (id,runtime_revision_id,qualification_id,domain_kind,domain_id,role,runtime_identity_hash,details_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)", (identifier, revision_id, qualification.id, domain_kind, domain_id, role, qualification.runtime_identity_hash, _json(dict(details or {})), created))
        self.conn.commit()
        return RuntimeBinding(identifier, revision_id, qualification.id, domain_kind, domain_id, role, qualification.runtime_identity_hash, dict(details or {}), created)

    def wrap_execution(
        self,
        revision_id: str,
        command: Sequence[str],
        *,
        cwd: Optional[str],
        launch_spec: Mapping[str, Any],
    ) -> tuple[list[str], Optional[str], dict[str, str], RuntimeQualification]:
        """Render a qualified path-preserving managed execution command."""

        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        qualification = self.latest_qualification(revision_id)
        if qualification is None or qualification.status not in {"vendor_supported", "local_verified"}:
            raise ManagedRuntimeError("The selected runtime has not completed hardware qualification")
        verification = self.verify(qualification.id)
        if not verification["valid"]:
            raise ManagedRuntimeError("The selected runtime qualification is stale or corrupt")
        engine, adapter = self._engine(revision)
        candidates: list[Path] = []
        if cwd:
            candidates.append(Path(cwd).expanduser())
        for key in ("output_dir", "output_root", "dataset", "prompts", "validation_file", "asset_root", "model", "reference_model"):
            raw = launch_spec.get(key)
            if raw:
                path = Path(str(raw)).expanduser()
                if key in {"model", "reference_model"} and not (
                    path.is_absolute() or path.exists()
                ):
                    continue
                candidates.append(path if path.is_dir() else path.parent)
        for raw in launch_spec.get("runtime_mounts") or []:
            candidates.append(Path(str(raw)).expanduser())
        cache = self.root / "model-cache"
        cache.mkdir(parents=True, exist_ok=True)
        candidates.append(cache)
        mounts: list[RuntimeMount] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            value = str(resolved)
            if value in seen:
                continue
            seen.add(value)
            resolved.mkdir(parents=True, exist_ok=True)
            mounts.append(RuntimeMount(resolved, resolved))
        wrapped = adapter.wrap(
            command,
            image=revision.derived_image_ref,
            mounts=tuple(mounts),
            env={"HF_HOME": str(cache), "TRANSFORMERS_CACHE": str(cache)},
            cwd=Path(cwd).resolve() if cwd else None,
            name=f"halo-forge-{uuid.uuid4().hex[:12]}",
        )
        values = list(wrapped.argv)
        if engine == "podman":
            values[1:1] = self._podman_prefix(engine)[1:]
        return values, None, dict(wrapped.env), qualification

    # ---- reads / cancellation ---------------------------------------

    def get_preparation(self, identifier: str) -> Optional[RuntimePreparation]:
        row = self.conn.execute("SELECT * FROM runtime_preparations WHERE id=?", (identifier,)).fetchone()
        return self._preparation(row) if row else None

    def list_preparations(self, revision_id: Optional[str] = None) -> tuple[RuntimePreparation, ...]:
        if revision_id:
            rows = self.conn.execute("SELECT * FROM runtime_preparations WHERE runtime_revision_id=? ORDER BY created_at DESC", (revision_id,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM runtime_preparations ORDER BY created_at DESC").fetchall()
        return tuple(self._preparation(row) for row in rows)

    def get_qualification(self, identifier: Optional[str]) -> Optional[RuntimeQualification]:
        if not identifier:
            return None
        row = self.conn.execute("SELECT * FROM runtime_qualifications WHERE id=?", (identifier,)).fetchone()
        return self._qualification(row) if row else None

    def latest_qualification(self, revision_id: str) -> Optional[RuntimeQualification]:
        row = self.conn.execute("SELECT * FROM runtime_qualifications WHERE runtime_revision_id=? ORDER BY created_at DESC LIMIT 1", (revision_id,)).fetchone()
        return self._qualification(row) if row else None

    def list_qualifications(self, revision_id: Optional[str] = None) -> tuple[RuntimeQualification, ...]:
        if revision_id:
            rows = self.conn.execute("SELECT * FROM runtime_qualifications WHERE runtime_revision_id=? ORDER BY created_at DESC", (revision_id,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM runtime_qualifications ORDER BY created_at DESC").fetchall()
        return tuple(self._qualification(row, include_steps=False) for row in rows)

    def cancel(self, kind: str, identifier: str) -> Any:
        table = "runtime_preparations" if kind == "preparation" else "runtime_qualifications"
        row = self.conn.execute(f"SELECT work_item_id FROM {table} WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        self.conn.execute(f"UPDATE {table} SET cancel_requested=1 WHERE id=?", (identifier,))
        self.conn.commit()
        if row["work_item_id"] and self.scheduler:
            self.scheduler.cancel(str(row["work_item_id"]))
        return self.get_preparation(identifier) if kind == "preparation" else self.get_qualification(identifier)

    # ---- row conversion and updates ---------------------------------

    @staticmethod
    def _profile(row: Any) -> ManagedRuntimeProfile:
        return ManagedRuntimeProfile(str(row["id"]), str(row["name"]), str(row["accelerator_family"]), row["description"], row["latest_revision_id"], str(row["created_at"]), str(row["updated_at"]))

    @staticmethod
    def _revision(row: Any) -> ManagedRuntimeRevision:
        return ManagedRuntimeRevision(str(row["id"]), str(row["profile_id"]), int(row["revision_number"]), str(row["content_hash"]), str(row["adapter_id"]), str(row["adapter_version"]), str(row["engine"]), row["base_image"], row["base_image_digest"], row["derived_image_ref"], _loads(row["dependency_lock_json"], {}), _loads(row["configuration_json"], {}), tuple(_loads(row["trainer_contracts_json"], [])), row["download_bytes"], row["installed_bytes"], str(row["created_at"]))

    @staticmethod
    def _preparation(row: Any) -> RuntimePreparation:
        return RuntimePreparation(str(row["id"]), str(row["runtime_revision_id"]), str(row["status"]), str(row["stage"]), str(row["engine"]), row["image_id"], row["image_digest"], row["storage_path"], row["manifest_path"], row["manifest_hash"], _loads(row["progress_json"], {}), row["work_item_id"], row["error"], str(row["created_at"]), row["completed_at"])

    def _qualification(self, row: Any, *, include_steps: bool = True) -> RuntimeQualification:
        steps: tuple[RuntimeQualificationStep, ...] = ()
        if include_steps:
            values = self.conn.execute("SELECT * FROM runtime_qualification_steps WHERE qualification_id=? ORDER BY ordinal", (row["id"],)).fetchall()
            steps = tuple(RuntimeQualificationStep(str(value["qualification_id"]), int(value["ordinal"]), str(value["step_id"]), str(value["label"]), str(value["status"]), value["command_hash"], _loads(value["result_json"], {}), value["log_path"], value["started_at"], value["completed_at"]) for value in values)
        return RuntimeQualification(str(row["id"]), str(row["runtime_revision_id"]), row["preparation_id"], str(row["status"]), str(row["stage"]), str(row["host_identity_hash"]), str(row["device_identity_hash"]), str(row["runtime_identity_hash"]), row["qualification_hash"], row["evidence_path"], _loads(row["progress_json"], {}), row["work_item_id"], row["error"], str(row["created_at"]), row["completed_at"], steps)

    def _update_preparation(self, identifier: str, *, status: Optional[str] = None, stage: Optional[str] = None, progress: Optional[Mapping[str, Any]] = None) -> None:
        values: list[Any] = []
        assignments: list[str] = []
        for key, value in (("status", status), ("stage", stage), ("progress_json", _json(dict(progress)) if progress is not None else None)):
            if value is not None:
                assignments.append(f"{key}=?")
                values.append(value)
        values.append(identifier)
        self.conn.execute(f"UPDATE runtime_preparations SET {','.join(assignments)} WHERE id=?", values)
        self.conn.commit()

    def _fail_preparation(self, identifier: str, stage: str, error: str) -> RuntimePreparation:
        self.conn.execute("UPDATE runtime_preparations SET status='failed',stage=?,error=?,completed_at=? WHERE id=?", (stage, str(error)[-4000:], _now(), identifier))
        self.conn.commit()
        return self.get_preparation(identifier)  # type: ignore[return-value]

    def execute_work_item(self, item: Any) -> Mapping[str, Any]:
        operation = str(item.launch_spec.get("operation") or "")
        if operation == "prepare":
            return self.run_preparation(str(item.domain_id)).to_dict()
        if operation == "qualify":
            return self.run_qualification(str(item.domain_id)).to_dict()
        raise ManagedRuntimeError(f"unknown managed runtime operation: {operation}")
