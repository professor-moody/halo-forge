"""Safe argv-only execution adapters for native and managed runtimes."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class RuntimeMount:
    host_path: Path
    container_path: Path
    read_only: bool = False

    def validate(self) -> None:
        host = self.host_path.expanduser().resolve()
        target = self.container_path
        if not host.is_absolute() or not target.is_absolute():
            raise ValueError("runtime mounts require absolute paths")
        if ".." in target.parts:
            raise ValueError("runtime mount targets cannot traverse parents")


@dataclass(frozen=True)
class RuntimeCommand:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Optional[Path] = None


class RuntimeExecutionAdapter:
    id = "base"
    version = "1"
    engines: tuple[str, ...] = ()

    def available(self) -> tuple[bool, Optional[str]]:
        raise NotImplementedError

    def wrap(
        self,
        argv: Sequence[str],
        *,
        image: Optional[str] = None,
        mounts: Sequence[RuntimeMount] = (),
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> RuntimeCommand:
        raise NotImplementedError

    @staticmethod
    def _argv(argv: Sequence[str]) -> tuple[str, ...]:
        values = tuple(str(value) for value in argv)
        if not values or any("\x00" in value for value in values):
            raise ValueError("runtime command must be non-empty argv without NUL bytes")
        return values


class NativeRuntimeAdapter(RuntimeExecutionAdapter):
    id = "native"
    engines = ("native",)

    def available(self) -> tuple[bool, Optional[str]]:
        return True, None

    def wrap(self, argv: Sequence[str], **kwargs: object) -> RuntimeCommand:
        cwd = kwargs.get("cwd")
        env = kwargs.get("env")
        return RuntimeCommand(
            self._argv(argv),
            {str(key): str(value) for key, value in dict(env or {}).items()},
            Path(cwd) if cwd else None,
        )


class ContainerRuntimeAdapter(RuntimeExecutionAdapter):
    executable = ""
    accelerator_family = ""

    def available(self) -> tuple[bool, Optional[str]]:
        if not shutil.which(self.executable):
            return False, f"{self.executable} is not installed"
        if self.executable == "podman" and getattr(os, "geteuid", lambda: 1)() == 0:
            return False, "rootless Podman is required; the current process is root"
        try:
            check = subprocess.run(
                [self.executable, "info", "--format", "json"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{self.executable} could not be inspected without elevation: {exc}"
        if check.returncode != 0:
            return False, f"{self.executable} is not accessible to the current user"
        if self.accelerator_family == "rocm":
            missing = [path for path in ("/dev/kfd", "/dev/dri") if not Path(path).exists()]
            if missing:
                return False, f"ROCm device access is unavailable: {', '.join(missing)}"
            if not os.access("/dev/kfd", os.R_OK | os.W_OK):
                return False, "/dev/kfd is not accessible to the current user"
        elif self.accelerator_family == "cuda":
            cdi_specs = (
                Path("/etc/cdi/nvidia.yaml"),
                Path("/var/run/cdi/nvidia.yaml"),
                Path("/run/cdi/nvidia.yaml"),
            )
            if not any(path.is_file() for path in cdi_specs):
                tool = shutil.which("nvidia-ctk")
                if not tool:
                    return False, "NVIDIA CDI is not configured for this container engine"
                try:
                    cdi = subprocess.run(
                        [tool, "cdi", "list"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        shell=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    return False, "NVIDIA CDI could not be inspected"
                if cdi.returncode != 0 or "nvidia.com/gpu" not in cdi.stdout:
                    return False, "NVIDIA CDI does not expose a GPU to the current user"
        return True, None

    def wrap(
        self,
        argv: Sequence[str],
        *,
        image: Optional[str] = None,
        mounts: Sequence[RuntimeMount] = (),
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> RuntimeCommand:
        if not image:
            raise ValueError("container runtime commands require a pinned image")
        values = [self.executable, "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"]
        if name:
            values.extend(("--name", str(name)))
        if self.accelerator_family == "rocm":
            values.extend(("--device", "/dev/kfd", "--device", "/dev/dri", "--ipc", "host"))
        elif self.accelerator_family == "cuda":
            # CDI works for both rootless Podman and a correctly configured
            # Docker daemon. It avoids privileged mode and daemon-socket mounts.
            values.extend(("--device", "nvidia.com/gpu=all"))
        for mount in mounts:
            mount.validate()
            source = str(mount.host_path.expanduser().resolve())
            target = str(mount.container_path)
            spec = f"type=bind,src={source},dst={target}"
            if mount.read_only:
                spec += ",readonly"
            values.extend(("--mount", spec))
        for key, value in sorted(dict(env or {}).items()):
            if any(token in key.upper() for token in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
                raise ValueError(f"credentials cannot be embedded in runtime command env: {key}")
            values.extend(("--env", f"{key}={value}"))
        if cwd:
            values.extend(("--workdir", str(cwd)))
        values.append(str(image))
        values.extend(self._argv(argv))
        return RuntimeCommand(tuple(values), {}, None)


class PodmanRocmRuntimeAdapter(ContainerRuntimeAdapter):
    id = "podman_rocm"
    executable = "podman"
    accelerator_family = "rocm"
    engines = ("podman",)


class PodmanCudaRuntimeAdapter(ContainerRuntimeAdapter):
    id = "podman_cuda"
    executable = "podman"
    accelerator_family = "cuda"
    engines = ("podman",)


class DockerCudaRuntimeAdapter(ContainerRuntimeAdapter):
    id = "docker_cuda"
    executable = "docker"
    accelerator_family = "cuda"
    engines = ("docker",)


class RuntimeExecutionAdapterRegistry:
    def __init__(self) -> None:
        self._values: dict[str, RuntimeExecutionAdapter] = {}

    def register(self, adapter: RuntimeExecutionAdapter) -> None:
        if adapter.id in self._values:
            raise ValueError(f"runtime adapter already registered: {adapter.id}")
        self._values[adapter.id] = adapter

    def get(self, adapter_id: str) -> RuntimeExecutionAdapter:
        try:
            return self._values[adapter_id]
        except KeyError as exc:
            raise KeyError(f"unknown runtime adapter: {adapter_id}") from exc

    def list(self) -> tuple[RuntimeExecutionAdapter, ...]:
        return tuple(self._values[key] for key in sorted(self._values))


RUNTIME_EXECUTION_ADAPTERS = RuntimeExecutionAdapterRegistry()
for _adapter in (
    NativeRuntimeAdapter(),
    PodmanRocmRuntimeAdapter(),
    PodmanCudaRuntimeAdapter(),
    DockerCudaRuntimeAdapter(),
):
    RUNTIME_EXECUTION_ADAPTERS.register(_adapter)
