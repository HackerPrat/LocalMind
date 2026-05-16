#!/usr/bin/env python3
"""
LocalMind.py

A single-file local AI trainer/workbench launcher.

This file deliberately starts with only Python standard-library imports.  The
bootstrap layer can create managed environments, then relaunch into the richer
PySide6 desktop UI when dependencies are available.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import http.client
import json
import os
import platform
import queue
import signal
import shutil
import socket
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import venv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


APP_NAME = "LocalMind"
APP_VERSION = "0.1.0"
PYTHON_TARGET = "3.13"

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".localmind"
LOG_DIR = STATE_DIR / "logs"
PROJECTS_DIR = STATE_DIR / "projects"
TOOLS_DIR = STATE_DIR / "tools"
ENVS_DIR = STATE_DIR / "envs"
BOOTSTRAP_DIR = STATE_DIR / "bootstrap"
MANIFEST_PATH = STATE_DIR / "manifest.json"
SERVER_PID_PATH = STATE_DIR / "llama-server.pid"
DEFAULT_PROJECT = "default"

PRESERVED_ASSETS = {
    "LocalMind.png",
    "prompt.txt",
    "imatrix.dat",
    "Darwin-2B-Opus-heretic.Q8_0.gguf",
    "Darwin-2B-Opus-heretic.imatrix.gguf",
    "polaris-heretic-q4_k_m-imat.gguf",
}

UI_PACKAGES = [
    "PySide6>=6.7",
    "requests>=2.32",
    "psutil>=5.9",
    "pillow>=10.0",
    "packaging>=24.0",
]

INFER_PACKAGES = [
    "requests>=2.32",
]

TRAIN_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "sentencepiece",
    "safetensors",
    "gguf",
    "pymupdf",
    "docling",
]

OPTIONAL_TRAIN_PACKAGES = [
    "bitsandbytes",
    "unsloth",
]

ENV_IMPORT_CHECKS = {
    "ui": [["PySide6"], ["requests"], ["psutil"], ["PIL"], ["packaging"]],
    "infer": [["requests"]],
    "train": [
        ["torch"],
        ["transformers"],
        ["datasets"],
        ["accelerate"],
        ["peft"],
        ["trl"],
        ["sentencepiece"],
        ["safetensors"],
        ["gguf"],
        ["pymupdf", "fitz"],
        ["docling"],
    ],
}

QUANT_PRESETS = {
    "Fast Q4_K_M": {
        "type": "Q4_K_M",
        "description": "Balanced size, speed, and quality for common local chat.",
        "imatrix": False,
    },
    "Quality Q5_K_M": {
        "type": "Q5_K_M",
        "description": "Larger output with better retention than 4-bit.",
        "imatrix": False,
    },
    "Compact IQ4_XS + iMatrix": {
        "type": "IQ4_XS",
        "description": "Small i-quant. Best used with an importance matrix.",
        "imatrix": True,
    },
    "Archive Q8_0": {
        "type": "Q8_0",
        "description": "Large, high-quality quantized export.",
        "imatrix": False,
    },
    "F16 Copy": {
        "type": "COPY",
        "description": "Metadata/pruning/copy operation without quantizing tensors.",
        "imatrix": False,
    },
}

SUPPORTED_AI_EXTS = {
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".tflite",
    ".mlmodel",
    ".pb",
    ".h5",
}

DOCUMENT_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".jsonl",
}

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (STATE_DIR, LOG_DIR, PROJECTS_DIR, TOOLS_DIR, ENVS_DIR, BOOTSTRAP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def log_path(name: str = "localmind.log") -> Path:
    ensure_dirs()
    return LOG_DIR / name


def append_log(message: str, name: str = "localmind.log") -> None:
    ensure_dirs()
    line = f"[{utc_now()}] {message.rstrip()}\n"
    with log_path(name).open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line)


def format_bytes(size: int | float | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    remaining = limit
    with path.open("rb") as fh:
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def safe_json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def command_display(cmd: list[str] | tuple[str, ...] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    return " ".join(str(part) for part in cmd)


def run_checked(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    line_cb: Callable[[str], None] | None = None,
) -> int:
    append_log(f"run: {command_display(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line_cb:
            line_cb(line.rstrip("\n"))
        append_log(line.rstrip("\n"), "commands.log")
    code = proc.wait()
    append_log(f"exit {code}: {command_display(cmd)}")
    if code:
        raise RuntimeError(f"Command failed with exit code {code}: {command_display(cmd)}")
    return code


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return str(pid) in proc.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def record_server_process(pid: int, cmd: list[str]) -> None:
    safe_json_dump(
        SERVER_PID_PATH,
        {
            "pid": pid,
            "cmd": cmd,
            "created_at": utc_now(),
        },
    )


def clear_server_process(pid: int | None = None) -> None:
    data = safe_json_load(SERVER_PID_PATH, {})
    if pid is not None and isinstance(data, dict) and int(data.get("pid") or -1) != pid:
        return
    try:
        SERVER_PID_PATH.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        append_log(f"could not clear server pid file: {exc}")


def clear_dead_server_record() -> None:
    data = safe_json_load(SERVER_PID_PATH, {})
    if not isinstance(data, dict):
        return
    try:
        pid = int(data.get("pid") or 0)
    except Exception:
        clear_server_process()
        return
    if not process_exists(pid):
        clear_server_process(pid)


def stop_recorded_server(timeout: float = 5.0) -> None:
    data = safe_json_load(SERVER_PID_PATH, {})
    if not isinstance(data, dict):
        return
    try:
        pid = int(data.get("pid") or 0)
    except Exception:
        return
    if not process_exists(pid):
        clear_server_process(pid)
        return
    append_log(f"stopping recorded llama-server pid {pid}")
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clear_server_process(pid)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        clear_server_process(pid)
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_exists(pid):
            clear_server_process(pid)
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    clear_server_process(pid)


def python_in_venv(path: Path) -> Path:
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def script_in_venv(path: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    if os.name == "nt":
        return path / "Scripts" / f"{name}{suffix}"
    return path / "bin" / name


def executable_names(base: str) -> list[str]:
    if os.name == "nt":
        return [base + ".exe", base]
    return [base]


def is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def find_free_port(start: int = 8765, stop: int = 8865) -> int:
    for port in range(start, stop):
        if not is_port_open("127.0.0.1", port):
            return port
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def system_summary() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def nvidia_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def rocm_available() -> bool:
    return shutil.which("rocm-smi") is not None or Path("/opt/rocm").exists()


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def cuda_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for key in ("LOCALMIND_CUDA_ROOT", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.extend(
        [
            Path("/usr/local/cuda"),
            Path("/opt/cuda"),
        ]
    )
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))
    return unique_paths(candidates)


def cuda_toolkit_info() -> dict[str, str] | None:
    lib_relatives = (
        "targets/x86_64-linux/lib/libcudart_static.a",
        "lib64/libcudart_static.a",
        "lib/libcudart_static.a",
    )
    for root in cuda_root_candidates():
        nvcc = root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if not nvcc.exists():
            continue
        for relative in lib_relatives:
            static_runtime = root / relative
            if static_runtime.exists():
                return {
                    "root": str(root),
                    "nvcc": str(nvcc),
                    "lib_dir": str(static_runtime.parent),
                    "static_runtime": str(static_runtime),
                }
    return None


def cuda_build_env() -> dict[str, str] | None:
    info = cuda_toolkit_info()
    if not info:
        return None
    env = os.environ.copy()
    root = info["root"]
    bin_dir = str(Path(root) / "bin")
    lib_dir = info["lib_dir"]
    env["CUDA_HOME"] = root
    env["CUDA_PATH"] = root
    env["CUDAToolkit_ROOT"] = root
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def compiler_major(executable: str) -> int | None:
    path = shutil.which(executable)
    if not path:
        return None
    try:
        out = subprocess.check_output([path, "-dumpfullversion"], text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            out = subprocess.check_output([path, "-dumpversion"], text=True, stderr=subprocess.DEVNULL).strip()
        return int(out.split(".")[0])
    except Exception:
        return None


def cuda_host_compiler_pair() -> tuple[str, str] | None:
    for major in (14, 13, 12, 11, 10):
        cc = shutil.which(f"gcc-{major}")
        cxx = shutil.which(f"g++-{major}")
        if cc and cxx:
            return cc, cxx
    return None


def cuda_default_host_supported() -> bool:
    major = compiler_major("gcc")
    return major is not None and major <= 14


def requested_llama_backend() -> str | None:
    value = os.environ.get("LOCALMIND_LLAMA_BACKEND", "").strip().lower()
    if value in {"cpu", "cuda", "vulkan", "metal", "hip"}:
        return value
    return None


def likely_llama_backend() -> str:
    requested = requested_llama_backend()
    if requested:
        return requested
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and ("arm" in machine or "aarch64" in machine):
        return "metal"
    if nvidia_available() and cuda_toolkit_info() and (cuda_host_compiler_pair() or cuda_default_host_supported()):
        return "cuda"
    if rocm_available():
        return "hip"
    if shutil.which("vulkaninfo"):
        return "vulkan"
    return "cpu"


def detect_python_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    names = ["python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"]
    for name in names:
        exe = shutil.which(name)
        if not exe or exe in seen:
            continue
        seen.add(exe)
        try:
            out = subprocess.check_output([exe, "--version"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            out = f"unavailable: {exc}"
        candidates.append({"name": name, "path": exe, "version": out})
    return candidates


def load_manifest() -> dict[str, Any]:
    data = safe_json_load(MANIFEST_PATH, {})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", APP_VERSION)
    data.setdefault("created_at", utc_now())
    data.setdefault("system", system_summary())
    data.setdefault("envs", {})
    data.setdefault("tools", {})
    return data


def save_manifest(data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    safe_json_dump(MANIFEST_PATH, data)


class Bootstrapper:
    def __init__(
        self,
        no_network: bool = False,
        line_cb: Callable[[str], None] | None = None,
        python_target: str = PYTHON_TARGET,
    ):
        ensure_dirs()
        self.no_network = no_network
        self.line_cb = line_cb or (lambda line: None)
        self.python_target = python_target or PYTHON_TARGET
        self.manifest = load_manifest()

    def emit(self, message: str) -> None:
        self.line_cb(message)
        append_log(message, "bootstrap.log")

    def find_uv(self) -> Path | None:
        env_uv = os.environ.get("LOCALMIND_UV")
        candidates: list[Path] = []
        if env_uv:
            candidates.append(Path(env_uv))
        for name in executable_names("uv"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
        candidates.extend(
            [
                script_in_venv(BOOTSTRAP_DIR / "uv-venv", "uv"),
                BOOTSTRAP_DIR / "uv-venv" / "bin" / "uv",
                BOOTSTRAP_DIR / "uv-venv" / "Scripts" / "uv.exe",
            ]
        )
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def ensure_uv(self) -> Path | None:
        uv = self.find_uv()
        if uv:
            self.emit(f"uv available: {uv}")
            self.manifest.setdefault("tools", {})["uv"] = str(uv)
            save_manifest(self.manifest)
            return uv
        if self.no_network:
            self.emit("uv is missing and --no-network is active; managed env setup is deferred.")
            return None

        uv_venv = BOOTSTRAP_DIR / "uv-venv"
        py = python_in_venv(uv_venv)
        if not py.exists():
            self.emit(f"creating bootstrap venv for uv: {uv_venv}")
            venv.EnvBuilder(with_pip=True, clear=False).create(str(uv_venv))
        self.emit("installing uv into local bootstrap venv")
        run_checked([str(py), "-m", "pip", "install", "--upgrade", "pip", "uv"], line_cb=self.emit)
        uv = self.find_uv()
        if not uv:
            raise RuntimeError("uv installation completed but uv executable was not found")
        self.manifest.setdefault("tools", {})["uv"] = str(uv)
        save_manifest(self.manifest)
        return uv

    def python_minor(self, py: Path) -> str:
        try:
            out = subprocess.check_output(
                [
                    str(py),
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            return out
        except Exception as exc:
            return f"unknown: {exc}"

    def env_import_failures(self, py: Path, module_groups: list[list[str]]) -> list[str]:
        if not module_groups:
            return []
        script = (
            "import importlib, json, sys\n"
            f"checks = {json.dumps(module_groups)}\n"
            "failures = []\n"
            "for group in checks:\n"
            "    ok = False\n"
            "    errors = []\n"
            "    for name in group:\n"
            "        try:\n"
            "            importlib.import_module(name)\n"
            "            ok = True\n"
            "            break\n"
            "        except Exception as exc:\n"
            "            errors.append(f'{name}: {exc}')\n"
            "    if not ok:\n"
            "        failures.append(' or '.join(group))\n"
            "print(json.dumps(failures))\n"
            "sys.exit(1 if failures else 0)\n"
        )
        proc = subprocess.run(
            [str(py), "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = proc.stdout.strip().splitlines()
        if not output:
            return ["import-check"]
        try:
            data = json.loads(output[-1])
        except Exception:
            return [f"import-check: {proc.stdout.strip()[-500:]}"]
        if isinstance(data, list):
            return [str(item) for item in data]
        return ["import-check"]

    def ensure_env(
        self,
        name: str,
        packages: list[str],
        python_version: str | None = None,
        optional_packages: list[str] | None = None,
    ) -> Path | None:
        python_version = python_version or self.python_target
        python_minor_target = ".".join(python_version.split(".")[:2])
        env_dir = ENVS_DIR / name
        py = python_in_venv(env_dir)
        if py.exists():
            actual_python = self.python_minor(py)
            if actual_python != python_minor_target:
                message = f"{name} env uses Python {actual_python}; target is {python_version}"
                if self.no_network:
                    self.emit(f"{message}; --no-network prevents rebuilding.")
                    self.manifest.setdefault("envs", {})[name] = {
                        "path": str(env_dir),
                        "python": str(py),
                        "status": "wrong-python-no-network",
                        "python_actual": actual_python,
                        "python_target": python_version,
                    }
                    save_manifest(self.manifest)
                    return None
                self.emit(f"{message}; rebuilding managed env.")
                shutil.rmtree(env_dir)
                py = python_in_venv(env_dir)
            else:
                failures = self.env_import_failures(py, ENV_IMPORT_CHECKS.get(name, []))
                if not failures:
                    self.emit(f"{name} env available: {py}")
                    self.manifest.setdefault("envs", {})[name] = {
                        "path": str(env_dir),
                        "python": str(py),
                        "status": "available",
                        "python_target": python_version,
                    }
                    save_manifest(self.manifest)
                    return py
                if self.no_network:
                    self.emit(f"{name} env is incomplete and --no-network is active: {', '.join(failures)}")
                    self.manifest.setdefault("envs", {})[name] = {
                        "path": str(env_dir),
                        "python": str(py),
                        "status": "incomplete-no-network",
                        "missing": failures,
                        "python_target": python_version,
                    }
                    save_manifest(self.manifest)
                    return None
                uv = self.ensure_uv()
                if not uv:
                    self.emit(f"{name} env cannot be repaired because uv is unavailable.")
                    return None
                self.emit(f"repairing {name} env; missing imports: {', '.join(failures)}")
                run_checked([str(uv), "pip", "install", "--python", str(py), *packages], line_cb=self.emit)
                failures = self.env_import_failures(py, ENV_IMPORT_CHECKS.get(name, []))
                if failures:
                    raise RuntimeError(f"{name} env repair finished but imports still fail: {', '.join(failures)}")
            self.emit(f"{name} env available: {py}")
            self.manifest.setdefault("envs", {})[name] = {
                "path": str(env_dir),
                "python": str(py),
                "status": "available",
                "python_target": python_version,
            }
            save_manifest(self.manifest)
            return py

        uv = self.ensure_uv()
        if self.no_network:
            self.emit(f"{name} env missing and --no-network is active; setup is deferred.")
            self.manifest.setdefault("envs", {})[name] = {
                "path": str(env_dir),
                "python": str(py),
                "status": "missing-no-network",
                "python_target": python_version,
            }
            save_manifest(self.manifest)
            return None
        if not uv:
            self.emit(f"{name} env cannot be created because uv is unavailable.")
            return None

        self.emit(f"ensuring managed Python {python_version} is installed for {name}")
        try:
            run_checked([str(uv), "python", "install", python_version], line_cb=self.emit)
        except Exception as exc:
            raise RuntimeError(
                f"LocalMind needs Python {python_version} for the {name} environment, but uv could not "
                f"install it. Check network access or uv's python-downloads setting, then rerun. "
                f"Original error: {exc}"
            ) from exc

        self.emit(f"creating {name} env with Python {python_version}: {env_dir}")
        run_checked(
            [str(uv), "venv", "--managed-python", "--python", python_version, str(env_dir)],
            line_cb=self.emit,
        )
        py = python_in_venv(env_dir)
        if not py.exists():
            raise RuntimeError(f"uv created {env_dir}, but python was not found at {py}")
        install_cmd = [str(uv), "pip", "install", "--python", str(py), *packages]
        self.emit(f"installing {name} packages")
        run_checked(install_cmd, line_cb=self.emit)
        failures = self.env_import_failures(py, ENV_IMPORT_CHECKS.get(name, []))
        if failures:
            raise RuntimeError(f"{name} env installed but imports still fail: {', '.join(failures)}")
        if optional_packages:
            self.emit(f"installing optional {name} packages if supported")
            try:
                run_checked([str(uv), "pip", "install", "--python", str(py), *optional_packages], line_cb=self.emit)
                optional_status = "installed"
            except Exception as exc:
                optional_status = f"skipped: {exc}"
                self.emit(f"optional package install skipped: {exc}")
        else:
            optional_status = "none"
        self.manifest.setdefault("envs", {})[name] = {
            "path": str(env_dir),
            "python": str(py),
            "status": "available",
            "python_target": python_version,
            "optional": optional_status,
        }
        save_manifest(self.manifest)
        return py

    def bootstrap_ui(self) -> Path | None:
        return self.ensure_env("ui", UI_PACKAGES)

    def bootstrap_train(self) -> Path | None:
        return self.ensure_env("train", TRAIN_PACKAGES, optional_packages=OPTIONAL_TRAIN_PACKAGES)

    def bootstrap_infer(self) -> Path | None:
        return self.ensure_env("infer", INFER_PACKAGES)

    def bootstrap_all_light(self) -> None:
        self.emit("LocalMind bootstrap started")
        self.emit(f"system: {json.dumps(system_summary(), sort_keys=True)}")
        self.emit(f"python candidates: {json.dumps(detect_python_candidates(), sort_keys=True)}")
        self.ensure_uv()
        self.bootstrap_ui()
        self.bootstrap_infer()
        self.emit("LocalMind core bootstrap finished")

    def bootstrap_all_full(self) -> None:
        self.bootstrap_all_light()
        self.bootstrap_train()
        self.emit("LocalMind bootstrap finished")


@dataclasses.dataclass
class FileCapability:
    path: Path
    kind: str
    status: str
    workflow: str
    details: dict[str, Any]


class GGUFReader:
    TYPE_UINT8 = 0
    TYPE_INT8 = 1
    TYPE_UINT16 = 2
    TYPE_INT16 = 3
    TYPE_UINT32 = 4
    TYPE_INT32 = 5
    TYPE_FLOAT32 = 6
    TYPE_BOOL = 7
    TYPE_STRING = 8
    TYPE_ARRAY = 9
    TYPE_UINT64 = 10
    TYPE_INT64 = 11
    TYPE_FLOAT64 = 12

    def __init__(self, path: Path, max_kv: int = 512):
        self.path = path
        self.max_kv = max_kv

    def read(self) -> dict[str, Any]:
        import struct

        meta: dict[str, Any] = {
            "format": "GGUF",
            "path": str(self.path),
            "size": self.path.stat().st_size,
        }
        with self.path.open("rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                raise ValueError("not a GGUF file")
            version = struct.unpack("<I", fh.read(4))[0]
            tensor_count = struct.unpack("<Q", fh.read(8))[0]
            kv_count = struct.unpack("<Q", fh.read(8))[0]
            meta.update({"version": version, "tensor_count": tensor_count, "kv_count": kv_count})
            kvs: dict[str, Any] = {}
            for _ in range(min(kv_count, self.max_kv)):
                key = self._read_string(fh, struct)
                value_type = struct.unpack("<I", fh.read(4))[0]
                try:
                    value = self._read_value(fh, struct, value_type)
                except Exception:
                    value = "<unparsed>"
                    break
                kvs[key] = value
            meta["metadata"] = kvs
            for preferred in (
                "general.name",
                "general.architecture",
                "llama.context_length",
                "qwen2.context_length",
                "gemma.context_length",
                "general.file_type",
                "tokenizer.chat_template",
            ):
                if preferred in kvs:
                    meta[preferred] = kvs[preferred]
        return meta

    def _read_string(self, fh: Any, struct: Any) -> str:
        length = struct.unpack("<Q", fh.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("unreasonable GGUF string length")
        return fh.read(length).decode("utf-8", errors="replace")

    def _read_scalar(self, fh: Any, struct: Any, value_type: int) -> Any:
        formats = {
            self.TYPE_UINT8: "<B",
            self.TYPE_INT8: "<b",
            self.TYPE_UINT16: "<H",
            self.TYPE_INT16: "<h",
            self.TYPE_UINT32: "<I",
            self.TYPE_INT32: "<i",
            self.TYPE_FLOAT32: "<f",
            self.TYPE_BOOL: "<?",
            self.TYPE_UINT64: "<Q",
            self.TYPE_INT64: "<q",
            self.TYPE_FLOAT64: "<d",
        }
        fmt = formats.get(value_type)
        if not fmt:
            raise ValueError(f"unsupported scalar type {value_type}")
        return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]

    def _read_value(self, fh: Any, struct: Any, value_type: int) -> Any:
        if value_type == self.TYPE_STRING:
            value = self._read_string(fh, struct)
            if len(value) > 4096:
                return value[:4096] + "...<truncated>"
            return value
        if value_type == self.TYPE_ARRAY:
            item_type = struct.unpack("<I", fh.read(4))[0]
            length = struct.unpack("<Q", fh.read(8))[0]
            values = []
            max_items = 128
            for idx in range(length):
                item = self._read_value(fh, struct, item_type)
                if idx < max_items:
                    values.append(item)
            if length > max_items:
                values.append(f"...<{length - max_items} more>")
            return values
        return self._read_scalar(fh, struct, value_type)


def read_model_config_dir(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    data = safe_json_load(config_path, {}) if config_path.exists() else {}
    if not isinstance(data, dict):
        data = {}
    files = [p.name for p in path.iterdir()] if path.is_dir() else []
    return {
        "format": "hf-directory",
        "config": data,
        "architectures": data.get("architectures"),
        "model_type": data.get("model_type"),
        "quantization_config": data.get("quantization_config"),
        "files": sorted(files)[:200],
    }


def detect_file_capability(path: Path) -> FileCapability:
    suffix = path.suffix.lower()
    details: dict[str, Any] = {
        "size": path.stat().st_size if path.exists() and path.is_file() else None,
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        if path.exists()
        else None,
    }
    if path.is_dir():
        config = path / "config.json"
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            details.update(safe_json_load(adapter_config, {}))
            return FileCapability(
                path,
                "PEFT adapter",
                "supported",
                "Adapter can be loaded with its original base model, merged when method/settings allow.",
                details,
            )
        if config.exists():
            details.update(read_model_config_dir(path))
            arch_blob = json.dumps(details.get("architectures", "")).lower()
            if any(token in arch_blob for token in ("vision", "vl", "llava", "image")):
                kind = "Hugging Face VLM"
                workflow = "VLM LoRA/QLoRA training path when train env and hardware support it."
            else:
                kind = "Hugging Face LLM"
                workflow = "Full LoRA/QLoRA, adapter merge, and GGUF conversion path."
            return FileCapability(path, kind, "supported", workflow, details)
        return FileCapability(path, "folder", "inspect-only", "Folder is not a recognized model/project yet.", details)

    if suffix == ".gguf":
        try:
            details.update(GGUFReader(path).read())
            status = "supported"
            workflow = (
                "Direct chat/metadata/quantization supported. Fine-tuning requires supported "
                "Transformers GGUF load or original/base weights."
            )
        except Exception as exc:
            details["error"] = str(exc)
            status = "damaged-or-unknown"
            workflow = "GGUF magic or metadata could not be parsed."
        return FileCapability(path, "GGUF model", status, workflow, details)

    if suffix == ".safetensors":
        return FileCapability(
            path,
            "Safetensors weights",
            "supported-with-config",
            "Train/export when paired with tokenizer/config model folder.",
            details,
        )
    if suffix in {".bin", ".pt", ".pth"}:
        return FileCapability(
            path,
            "PyTorch weights",
            "supported-with-config",
            "Train/export when paired with tokenizer/config model folder. Prefer safetensors when possible.",
            details,
        )
    if suffix in {".onnx", ".tflite", ".mlmodel", ".pb", ".h5"}:
        return FileCapability(
            path,
            "non-LLM runtime artifact",
            "recognized-not-trainable-v1",
            "Metadata/import only in this LocalMind build; no LLM LoRA/GGUF export path.",
            details,
        )
    if suffix in DOCUMENT_EXTS:
        return FileCapability(
            path,
            "document source",
            "ingestible",
            "Can be extracted into a versioned knowledge source and training dataset.",
            details,
        )
    if suffix in IMAGE_EXTS:
        return FileCapability(
            path,
            "image source",
            "ingestible",
            "Can be stored as VLM source material; OCR/captioning depends on local tools/models.",
            details,
        )
    return FileCapability(path, "unknown", "inspect-only", "Recognized only as a generic file.", details)


def scan_workspace_for_models(root: Path = ROOT) -> list[FileCapability]:
    found: list[FileCapability] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith(".localmind"):
            continue
        if child.is_file() and child.suffix.lower() in SUPPORTED_AI_EXTS:
            found.append(detect_file_capability(child))
        elif child.is_dir() and ((child / "config.json").exists() or (child / "adapter_config.json").exists()):
            found.append(detect_file_capability(child))
    return found


def normalize_project_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name.strip())
    cleaned = cleaned.strip("-_") or DEFAULT_PROJECT
    return cleaned[:80]


class ProjectStore:
    def __init__(self, project_name: str = DEFAULT_PROJECT):
        ensure_dirs()
        self.name = normalize_project_name(project_name)
        self.root = PROJECTS_DIR / self.name
        self.sources_dir = self.root / "sources"
        self.datasets_dir = self.root / "datasets"
        self.exports_dir = self.root / "exports"
        self.jobs_dir = self.root / "jobs"
        self.project_path = self.root / "project.json"
        self.db_path = self.root / "sources.sqlite3"
        for path in (self.root, self.sources_dir, self.datasets_dir, self.exports_dir, self.jobs_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._ensure_project()
        self._ensure_db()

    def _ensure_project(self) -> None:
        data = safe_json_load(self.project_path, {})
        if not isinstance(data, dict) or not data:
            data = {
                "version": APP_VERSION,
                "name": self.name,
                "created_at": utc_now(),
                "settings": default_settings(),
                "current_model": None,
                "conversation_summary": "",
                "chat_history": [],
            }
            safe_json_dump(self.project_path, data)
        else:
            data.setdefault("settings", default_settings())
            data.setdefault("chat_history", [])
            data.setdefault("conversation_summary", "")
            safe_json_dump(self.project_path, data)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    stored_path TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT,
                    sha256 TEXT,
                    chars INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id)
                )
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS source_fts
                    USING fts5(text, source_id UNINDEXED, chunk_id UNINDEXED)
                    """
                )
            except sqlite3.OperationalError:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS source_fts (
                        text TEXT,
                        source_id TEXT,
                        chunk_id TEXT
                    )
                    """
                )

    def load_project(self) -> dict[str, Any]:
        data = safe_json_load(self.project_path, {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("settings", default_settings())
        data.setdefault("chat_history", [])
        data.setdefault("conversation_summary", "")
        return data

    def save_project(self, data: dict[str, Any]) -> None:
        data["updated_at"] = utc_now()
        safe_json_dump(self.project_path, data)

    def settings(self) -> dict[str, Any]:
        return self.load_project().setdefault("settings", default_settings())

    def update_settings(self, settings: dict[str, Any]) -> None:
        data = self.load_project()
        merged = default_settings()
        merged.update(settings)
        data["settings"] = merged
        self.save_project(data)

    def set_current_model(self, path: str | None) -> None:
        data = self.load_project()
        data["current_model"] = path
        self.save_project(data)

    def current_model(self) -> str | None:
        return self.load_project().get("current_model")

    def add_chat(self, role: str, content: str) -> None:
        data = self.load_project()
        history = data.setdefault("chat_history", [])
        history.append({"role": role, "content": content, "created_at": utc_now()})
        if len(history) > 200:
            older = history[:-120]
            summary = data.get("conversation_summary", "")
            summary = compact_summary(summary, older)
            data["conversation_summary"] = summary
            data["chat_history"] = history[-120:]
        self.save_project(data)

    def chat_history(self) -> list[dict[str, Any]]:
        return list(self.load_project().get("chat_history", []))

    def conversation_summary(self) -> str:
        return str(self.load_project().get("conversation_summary") or "")

    def list_sources(self, include_deleted: bool = False) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if include_deleted:
                rows = conn.execute("SELECT * FROM sources ORDER BY updated_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sources WHERE status != 'deleted' ORDER BY updated_at DESC"
                ).fetchall()
        return rows

    def get_source(self, source_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()

    def active_chunks(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT chunks.*
                FROM chunks
                JOIN sources ON chunks.source_id = sources.id
                WHERE sources.status != 'deleted'
                ORDER BY sources.updated_at DESC, chunks.ordinal ASC
                """
            ).fetchall()

    def search(self, query_text: str, limit: int = 6) -> list[dict[str, Any]]:
        query = " ".join(token for token in query_text.replace('"', " ").split() if len(token) > 2)
        if not query:
            return []
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT source_fts.source_id, source_fts.chunk_id, chunks.text, sources.title, sources.path
                    FROM source_fts
                    JOIN chunks ON source_fts.chunk_id = chunks.id
                    JOIN sources ON source_fts.source_id = sources.id
                    WHERE source_fts MATCH ? AND sources.status != 'deleted'
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{query.split()[0]}%"
                rows = conn.execute(
                    """
                    SELECT chunks.source_id, chunks.id AS chunk_id, chunks.text, sources.title, sources.path
                    FROM chunks
                    JOIN sources ON chunks.source_id = sources.id
                    WHERE chunks.text LIKE ? AND sources.status != 'deleted'
                    LIMIT ?
                    """,
                    (like, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def add_or_update_source(self, path: Path, manual_text: str | None = None) -> str:
        path = path.resolve()
        source_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
        existing = self.get_source(source_id)
        version = int(existing["version"] + 1) if existing else 1
        extracted = (
            ExtractedSource.from_manual(path, manual_text)
            if manual_text is not None
            else extract_source(path)
        )
        chunks = chunk_text(extracted.text)
        stored_path = ""
        if path.exists() and path.is_file():
            dest = self.sources_dir / f"{source_id}_v{version}_{path.name}"
            try:
                shutil.copy2(path, dest)
                stored_path = str(dest)
            except Exception as exc:
                extracted.metadata["copy_warning"] = str(exc)
        now = utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM source_fts WHERE source_id = ?", (source_id,))
            conn.execute(
                """
                INSERT INTO sources (
                    id, version, path, stored_path, kind, status, title, sha256, chars,
                    created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    version=excluded.version,
                    path=excluded.path,
                    stored_path=excluded.stored_path,
                    kind=excluded.kind,
                    status=excluded.status,
                    title=excluded.title,
                    sha256=excluded.sha256,
                    chars=excluded.chars,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    source_id,
                    version,
                    str(path),
                    stored_path,
                    extracted.kind,
                    "active",
                    extracted.title,
                    extracted.sha256,
                    len(extracted.text),
                    existing["created_at"] if existing else now,
                    now,
                    json.dumps(extracted.metadata, sort_keys=True),
                ),
            )
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{source_id}:{version}:{idx}"
                meta = {"source_version": version, "ordinal": idx}
                conn.execute(
                    "INSERT INTO chunks (id, source_id, ordinal, text, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, source_id, idx, chunk, json.dumps(meta, sort_keys=True)),
                )
                conn.execute(
                    "INSERT INTO source_fts (text, source_id, chunk_id) VALUES (?, ?, ?)",
                    (chunk, source_id, chunk_id),
                )
        return source_id

    def update_source_text(self, source_id: str, new_text: str) -> None:
        row = self.get_source(source_id)
        if not row:
            raise KeyError(source_id)
        path = Path(row["path"])
        self.add_or_update_source(path, manual_text=new_text)

    def delete_source(self, source_id: str) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute("UPDATE sources SET status = 'deleted', updated_at = ? WHERE id = ?", (now, source_id))
            conn.execute("DELETE FROM source_fts WHERE source_id = ?", (source_id,))

    def dataset_jsonl(self) -> Path:
        rows = self.active_chunks()
        out = self.datasets_dir / f"localmind_sft_{int(time.time())}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                text = row["text"].strip()
                if not text:
                    continue
                record = {
                    "text": (
                        "Reference material to integrate into the assistant's behavior and answers:\n\n"
                        + text
                    )
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out


@dataclasses.dataclass
class ExtractedSource:
    path: Path
    kind: str
    title: str
    text: str
    sha256: str
    metadata: dict[str, Any]

    @classmethod
    def from_manual(cls, path: Path, text: str | None) -> "ExtractedSource":
        content = text or ""
        return cls(
            path=path,
            kind="manual-edit",
            title=path.name,
            text=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            metadata={"extractor": "manual", "path": str(path), "edited_at": utc_now()},
        )


def extract_source(path: Path) -> ExtractedSource:
    suffix = path.suffix.lower()
    title = path.name
    metadata: dict[str, Any] = {"path": str(path), "suffix": suffix, "extractors": []}
    digest = sha256_file(path) if path.exists() and path.is_file() else ""

    if suffix in {".txt", ".md", ".markdown", ".json", ".jsonl", ".csv", ".html", ".htm"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata["extractors"].append("plain-text")
        return ExtractedSource(path, "text", title, text, digest, metadata)

    if suffix == ".pdf":
        text = extract_with_docling(path, metadata)
        if not text.strip():
            text = extract_pdf_with_pymupdf(path, metadata)
        if not text.strip():
            text = f"[PDF imported but no text extractor succeeded for {path.name}]"
            metadata["warning"] = "No PDF text extractor available or extraction returned no text."
        return ExtractedSource(path, "pdf", title, text, digest, metadata)

    if suffix in {".docx", ".pptx", ".xlsx"}:
        text = extract_with_docling(path, metadata)
        if not text.strip():
            text = f"[Document imported but Docling is not available or could not extract {path.name}]"
            metadata["warning"] = "Install/use the train env with docling for structured extraction."
        return ExtractedSource(path, "document", title, text, digest, metadata)

    if suffix in IMAGE_EXTS:
        text = extract_image_description(path, metadata)
        return ExtractedSource(path, "image", title, text, digest, metadata)

    capability = detect_file_capability(path)
    text = (
        f"{path.name}\n"
        f"Kind: {capability.kind}\n"
        f"Status: {capability.status}\n"
        f"Workflow: {capability.workflow}\n"
        f"Details: {json.dumps(capability.details, indent=2, sort_keys=True, default=str)}\n"
    )
    capability_data = dataclasses.asdict(capability)
    capability_data["path"] = str(capability.path)
    metadata["capability"] = capability_data
    return ExtractedSource(path, "metadata", title, text, digest, metadata)


def extract_with_docling(path: Path, metadata: dict[str, Any]) -> str:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        result = converter.convert(str(path))
        text = result.document.export_to_markdown()
        metadata["extractors"].append("docling")
        return text or ""
    except Exception as exc:
        metadata.setdefault("extractor_errors", {})["docling"] = str(exc)
        return ""


def extract_pdf_with_pymupdf(path: Path, metadata: dict[str, Any]) -> str:
    try:
        try:
            import pymupdf  # type: ignore
        except Exception:
            import fitz as pymupdf  # type: ignore

        with pymupdf.open(str(path)) as doc:
            pages = [page.get_text() for page in doc]
        metadata["extractors"].append("pymupdf")
        metadata["page_count"] = len(pages)
        return "\n\n".join(pages)
    except Exception as exc:
        metadata.setdefault("extractor_errors", {})["pymupdf"] = str(exc)
        return ""


def extract_image_description(path: Path, metadata: dict[str, Any]) -> str:
    lines = [f"Image source: {path.name}"]
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            metadata["image"] = {
                "format": img.format,
                "mode": img.mode,
                "size": list(img.size),
            }
            lines.append(f"Format: {img.format}")
            lines.append(f"Size: {img.size[0]} x {img.size[1]}")
            lines.append(f"Mode: {img.mode}")
            metadata["extractors"].append("pillow")
    except Exception as exc:
        metadata.setdefault("extractor_errors", {})["pillow"] = str(exc)
        lines.append("Image metadata extraction unavailable.")
    lines.append(
        "No local OCR/caption model has been applied yet. Add a manual caption or use a VLM "
        "training project to pair this image with instructions."
    )
    return "\n".join(lines)


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 180) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                start = max(end - overlap, end)
            continue
        candidate = (current + "\n\n" + paragraph).strip() if current else paragraph
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + paragraph).strip()
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def compact_summary(existing: str, older_history: list[dict[str, Any]]) -> str:
    parts = [existing.strip()] if existing.strip() else []
    for item in older_history[-40:]:
        role = item.get("role", "unknown")
        content = str(item.get("content", "")).strip().replace("\n", " ")
        if len(content) > 320:
            content = content[:320] + "..."
        if content:
            parts.append(f"{role}: {content}")
    summary = "\n".join(parts)
    if len(summary) > 12000:
        summary = summary[-12000:]
    return summary.strip()


def default_settings() -> dict[str, Any]:
    return {
        "mode": "normal",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.08,
        "context_length": 4096,
        "max_tokens": 512,
        "gpu_layers": -1,
        "threads": max(1, (os.cpu_count() or 4) - 1),
        "server_host": "127.0.0.1",
        "server_port": 8765,
        "system_prompt": read_prompt_file(),
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "epochs": 1.0,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "max_seq_length": 1024,
        "packing": False,
        "dtype": "auto",
        "load_4bit": True,
        "target_modules": "all-linear",
        "quant_preset": "Fast Q4_K_M",
        "export_format": "adapter+gguf",
        "offline_after_setup": True,
    }


def read_prompt_file() -> str:
    prompt = ROOT / "prompt.txt"
    if prompt.exists():
        try:
            return prompt.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass
    return "You are LocalMind, a local offline assistant. Use provided local context when relevant."


@dataclasses.dataclass
class CommandSpec:
    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None


class ToolRegistry:
    REQUIRED_TARGETS = ["llama-server", "llama-cli", "llama-quantize", "llama-imatrix"]

    def __init__(self):
        ensure_dirs()
        self.llama_root = TOOLS_DIR / "llama.cpp"
        self.llama_build = self.llama_root / "build"

    def build_dir(self, backend: str | None = None) -> Path:
        return self.llama_root / f"build-{backend or likely_llama_backend()}"

    def find_tool(self, name: str) -> Path | None:
        candidates: list[Path] = []
        for exe_name in executable_names(name):
            found = shutil.which(exe_name)
            if found:
                candidates.append(Path(found))
        build_dirs = []
        if self.llama_root.exists():
            build_dirs = unique_paths([self.build_dir(likely_llama_backend()), *sorted(self.llama_root.glob("build-*"))])
        bin_dirs = [
            self.llama_root / "build" / "bin",
            self.llama_root / "build",
            self.llama_root,
        ]
        for build_dir in build_dirs:
            bin_dirs.extend([build_dir / "bin", build_dir / "tools" / "main", build_dir])
        for directory in bin_dirs:
            for exe_name in executable_names(name):
                candidates.append(directory / exe_name)
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def llama_status(self) -> dict[str, Any]:
        backend = likely_llama_backend()
        return {
            "backend": backend,
            "root": str(self.llama_root),
            "build_dir": str(self.build_dir(backend)),
            "cuda_toolkit": cuda_toolkit_info() or {},
            "cuda_host_compiler_pair": list(cuda_host_compiler_pair() or ()),
            "gcc_major": compiler_major("gcc"),
            "server": str(self.find_tool("llama-server") or ""),
            "cli": str(self.find_tool("llama-cli") or ""),
            "quantize": str(self.find_tool("llama-quantize") or ""),
            "imatrix": str(self.find_tool("llama-imatrix") or ""),
            "convert_hf_to_gguf": str(self.llama_root / "convert_hf_to_gguf.py")
            if (self.llama_root / "convert_hf_to_gguf.py").exists()
            else "",
        }

    def backend_order(self) -> list[str]:
        preferred = likely_llama_backend()
        order = [preferred]
        for backend in ("cuda", "vulkan", "metal", "hip", "cpu"):
            if backend == "cuda" and not cuda_toolkit_info():
                continue
            if backend == "vulkan" and not shutil.which("vulkaninfo"):
                continue
            if backend == "metal" and platform.system().lower() != "darwin":
                continue
            if backend == "hip" and not rocm_available():
                continue
            if backend not in order:
                order.append(backend)
        return order

    def install_commands(self, backend: str | None = None) -> list[Any]:
        backend = backend or likely_llama_backend()
        env = os.environ.copy()
        cuda_info = None
        if backend == "cuda":
            cuda_info = cuda_toolkit_info()
            if cuda_info:
                env = cuda_build_env() or env
            elif shutil.which("vulkaninfo"):
                backend = "vulkan"
            else:
                backend = "cpu"
        build_dir = self.build_dir(backend)
        cmake_flags = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_WEBUI=OFF",
        ]
        if backend == "cuda":
            cuda_info = cuda_info or cuda_toolkit_info()
            if not cuda_info:
                raise RuntimeError(
                    "CUDA was selected, but no usable CUDA toolkit was found. "
                    "LocalMind requires nvcc and libcudart_static.a. Set LOCALMIND_CUDA_ROOT "
                    "or build with Vulkan/CPU."
                )
            cmake_flags.append("-DGGML_CUDA=ON")
            cmake_flags.append(f"-DCUDAToolkit_ROOT={cuda_info['root']}")
            cmake_flags.append(f"-DCMAKE_CUDA_COMPILER={cuda_info['nvcc']}")
            host_pair = cuda_host_compiler_pair()
            if host_pair:
                cmake_flags.append(f"-DCMAKE_C_COMPILER={host_pair[0]}")
                cmake_flags.append(f"-DCMAKE_CXX_COMPILER={host_pair[1]}")
                cmake_flags.append(f"-DCMAKE_CUDA_HOST_COMPILER={host_pair[1]}")
            elif not cuda_default_host_supported():
                cmake_flags.append("-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler")
        elif backend == "metal":
            cmake_flags.append("-DGGML_METAL=ON")
        elif backend == "hip":
            cmake_flags.append("-DGGML_HIP=ON")
        elif backend == "vulkan":
            cmake_flags.append("-DGGML_VULKAN=ON")
        clone = ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(self.llama_root)]
        configure = ["cmake", "-S", str(self.llama_root), "-B", str(build_dir), "--fresh", *cmake_flags]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
            "--target",
            *self.REQUIRED_TARGETS,
        ]
        commands: list[Any] = []
        if not (self.llama_root / "CMakeLists.txt").exists():
            commands.append(CommandSpec(clone, ROOT, os.environ.copy()))
        commands.extend([CommandSpec(configure, ROOT, env), CommandSpec(build, ROOT, env)])
        return commands


class LocalServer:
    def __init__(self, tools: ToolRegistry, line_cb: Callable[[str], None] | None = None):
        self.tools = tools
        self.line_cb = line_cb or (lambda line: None)
        self.process: subprocess.Popen[str] | None = None
        self.host = "127.0.0.1"
        self.port = 8765
        self.model_path: Path | None = None
        self._reader_thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, model_path: Path, settings: dict[str, Any]) -> None:
        if self.is_running():
            self.stop()
        exe = self.tools.find_tool("llama-server")
        if not exe:
            raise RuntimeError("llama-server not found. Install/build llama.cpp from the Jobs tab first.")
        self.host = str(settings.get("server_host") or "127.0.0.1")
        preferred_port = int(settings.get("server_port") or 8765)
        self.port = preferred_port if not is_port_open(self.host, preferred_port) else find_free_port(preferred_port + 1)
        ctx = str(int(settings.get("context_length") or 4096))
        gpu_layers = int(settings.get("gpu_layers", -1))
        threads = int(settings.get("threads", max(1, (os.cpu_count() or 4) - 1)))
        cmd = [
            str(exe),
            "-m",
            str(model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            ctx,
            "-t",
            str(threads),
        ]
        if gpu_layers != 0:
            cmd.extend(["-ngl", str(gpu_layers)])
        self.line_cb(f"Starting llama-server: {command_display(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        record_server_process(self.process.pid, cmd)
        self.model_path = model_path
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        deadline = time.time() + 25
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("llama-server exited before becoming ready")
            if self.health():
                self.line_cb(f"llama-server ready at http://{self.host}:{self.port}")
                return
            time.sleep(0.5)
        self.line_cb("llama-server did not answer health checks yet; it may still be loading.")

    def _read_output(self) -> None:
        proc = self.process
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            append_log(line, "llama-server.log")
            self.line_cb(line)

    def health(self) -> bool:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=0.5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status < 500
        except Exception:
            try:
                conn = http.client.HTTPConnection(self.host, self.port, timeout=0.5)
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                return resp.status < 500
            except Exception:
                return False

    def stop(self) -> None:
        if not self.process:
            return
        self.line_cb("Stopping llama-server")
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        clear_server_process(self.process.pid)
        self.process = None

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        if not self.is_running():
            raise RuntimeError("llama-server is not running")
        body = {
            "model": "localmind",
            "messages": messages,
            "temperature": float(settings.get("temperature", 0.7)),
            "top_p": float(settings.get("top_p", 0.95)),
            "max_tokens": int(settings.get("max_tokens", 512)),
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}/v1/chat/completions",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or choices[0].get("text") or "")
        return json.dumps(payload, indent=2)


def build_chat_messages(store: ProjectStore, user_message: str) -> list[dict[str, str]]:
    settings = store.settings()
    system = str(settings.get("system_prompt") or read_prompt_file())
    retrieved = store.search(user_message, limit=6)
    source_context = ""
    if retrieved:
        parts = []
        for idx, row in enumerate(retrieved, 1):
            title = row.get("title") or Path(str(row.get("path") or "")).name
            text = str(row.get("text") or "")
            if len(text) > 1400:
                text = text[:1400] + "..."
            parts.append(f"[Source {idx}: {title}]\n{text}")
        source_context = "\n\n".join(parts)
    summary = store.conversation_summary()
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if summary:
        messages.append({"role": "system", "content": "Conversation summary:\n" + summary})
    if source_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Relevant local knowledge from the user's imported sources follows. "
                    "Use it when it helps, and do not invent unsupported details.\n\n"
                    + source_context
                ),
            }
        )
    history = store.chat_history()[-18:]
    for item in history:
        role = str(item.get("role") or "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        messages.append({"role": role, "content": str(item.get("content") or "")})
    messages.append({"role": "user", "content": user_message})
    return messages


def write_training_script(job_dir: Path) -> Path:
    script = job_dir / "train_lora.py"
    script.write_text(
        r'''
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--packing", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    target_modules = args.target_modules
    if "," in target_modules:
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    train_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        max_length=args.max_seq_length,
        packing=args.packing,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "localmind_training_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"base_model": args.base_model, "dataset": args.dataset, "output": args.output}, fh, indent=2)


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )
    return script


def write_merge_script(job_dir: Path) -> Path:
    script = job_dir / "merge_adapter.py"
    script.write_text(
        r'''
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--safe-merge", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload(safe_merge=args.safe_merge)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "localmind_merge_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"base_model": args.base_model, "adapter": args.adapter, "output": args.output}, fh, indent=2)


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )
    return script


def run_console(args: argparse.Namespace) -> int:
    ensure_dirs()
    print(f"{APP_NAME} {APP_VERSION}")
    print(f"Workspace: {ROOT}")
    print(f"State: {STATE_DIR}")
    print("System:")
    for key, value in system_summary().items():
        print(f"  {key}: {value}")
    print("\nDetected model files:")
    for capability in scan_workspace_for_models():
        print(f"  - {capability.path.name}: {capability.kind} [{capability.status}]")
        print(f"    {capability.workflow}")
    print("\nPySide6 is not available in the active interpreter.")
    print("Run without --no-network to let LocalMind bootstrap its UI environment, or run --bootstrap-only first.")
    return 0


def run_headless_server(args: argparse.Namespace) -> int:
    store = ProjectStore(args.project)
    model = args.model or store.current_model()
    if not model:
        models = [cap for cap in scan_workspace_for_models() if cap.path.suffix.lower() == ".gguf"]
        if models:
            model = str(models[0].path)
    if not model:
        raise SystemExit("No GGUF model selected or discovered.")
    tools = ToolRegistry()
    server = LocalServer(tools, print)
    server.start(Path(model), store.settings())
    print(f"Serving {model} at http://{server.host}:{server.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


def import_qt_modules() -> dict[str, Any]:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal, QSize
    from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPixmap, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    return locals()


def run_ui_child(ui_py: Path, args: argparse.Namespace) -> int:
    cmd = [
        str(ui_py),
        str(Path(__file__).resolve()),
        "--run-app",
        "--project",
        args.project,
        "--python-version",
        args.python_version,
    ]
    if args.advanced:
        cmd.append("--advanced")
    if args.no_network:
        cmd.append("--no-network")

    platform_attempts: list[str | None] = [None]
    if platform.system().lower() == "linux" and not os.environ.get("QT_QPA_PLATFORM"):
        if os.environ.get("DISPLAY"):
            platform_attempts.append("xcb")
        if os.environ.get("WAYLAND_DISPLAY"):
            platform_attempts.append("wayland")

    deduped_attempts: list[str | None] = []
    for attempt in platform_attempts:
        if attempt not in deduped_attempts:
            deduped_attempts.append(attempt)

    last_code = 1
    for attempt in deduped_attempts:
        env = os.environ.copy()
        env["LOCALMIND_APP_ENV"] = "ui"
        env.setdefault("PYTHONFAULTHANDLER", "1")
        if attempt:
            env["QT_QPA_PLATFORM"] = attempt
        label = attempt or env.get("QT_QPA_PLATFORM") or "auto"
        append_log(f"launching UI child with Qt platform: {label}")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        try:
            last_code = proc.wait()
        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise
        if last_code not in (-11, 139):
            return last_code if last_code >= 0 else 128 + abs(last_code)
        append_log(f"UI child crashed with SIGSEGV using Qt platform {label}", "fatal.log")
        stop_recorded_server()
    return last_code if last_code >= 0 else 128 + abs(last_code)


def run_gui(args: argparse.Namespace) -> int:
    if not args.run_app and os.environ.get("LOCALMIND_APP_ENV") != "ui":
        bootstrap = Bootstrapper(no_network=args.no_network, line_cb=print, python_target=args.python_version)
        ui_py = bootstrap.bootstrap_ui()
        if ui_py and Path(ui_py).resolve() != Path(sys.executable).resolve():
            return run_ui_child(Path(ui_py), args)

    try:
        qt = import_qt_modules()
    except Exception:
        if args.no_network:
            return run_console(args)
        bootstrap = Bootstrapper(no_network=args.no_network, line_cb=print, python_target=args.python_version)
        ui_py = bootstrap.bootstrap_ui()
        if ui_py and Path(ui_py).resolve() != Path(sys.executable).resolve():
            return run_ui_child(Path(ui_py), args)
        return run_console(args)

    QApplication = qt["QApplication"]
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyleSheet(PITCH_BLACK_QSS)
    window = build_window_class(qt)(args)
    app.aboutToQuit.connect(window.stop_server)
    window.show()
    return int(app.exec())


PITCH_BLACK_QSS = """
QWidget {
    background: #000000;
    color: #f1f1f1;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background: #000000;
}
QTabWidget::pane, QFrame, QGroupBox {
    border: 1px solid #1f1f1f;
    border-radius: 4px;
}
QGroupBox {
    margin-top: 12px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #d9d9d9;
}
QPushButton, QToolButton {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 7px 10px;
}
QPushButton:hover, QToolButton:hover {
    background: #191919;
    border-color: #4a4a4a;
}
QPushButton:pressed, QToolButton:pressed {
    background: #242424;
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #050505;
    border: 1px solid #262626;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #3c6df0;
}
QTableWidget, QListWidget {
    background: #050505;
    border: 1px solid #262626;
    gridline-color: #171717;
    alternate-background-color: #090909;
}
QHeaderView::section {
    background: #0d0d0d;
    color: #e7e7e7;
    border: 1px solid #202020;
    padding: 6px;
}
QTabBar::tab {
    background: #050505;
    border: 1px solid #222222;
    padding: 8px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #151515;
    border-bottom-color: #151515;
}
QProgressBar {
    border: 1px solid #262626;
    border-radius: 4px;
    text-align: center;
    background: #050505;
}
QProgressBar::chunk {
    background: #3c6df0;
}
QSplitter::handle {
    background: #111111;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #050505;
    width: 12px;
    height: 12px;
}
QScrollBar::handle {
    background: #2a2a2a;
    border-radius: 4px;
}
"""


def build_window_class(qt: dict[str, Any]) -> type:
    QObject = qt["QObject"]
    Signal = qt["Signal"]
    Qt = qt["Qt"]
    QTimer = qt["QTimer"]
    QSize = qt["QSize"]
    QIcon = qt["QIcon"]
    QPixmap = qt["QPixmap"]
    QTextCursor = qt["QTextCursor"]
    QAction = qt["QAction"]
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QDoubleSpinBox = qt["QDoubleSpinBox"]
    QFileDialog = qt["QFileDialog"]
    QFormLayout = qt["QFormLayout"]
    QFrame = qt["QFrame"]
    QGridLayout = qt["QGridLayout"]
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QListWidget = qt["QListWidget"]
    QMainWindow = qt["QMainWindow"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QProgressBar = qt["QProgressBar"]
    QSizePolicy = qt["QSizePolicy"]
    QSpinBox = qt["QSpinBox"]
    QSplitter = qt["QSplitter"]
    QTableWidget = qt["QTableWidget"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    QTabWidget = qt["QTabWidget"]
    QTextEdit = qt["QTextEdit"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]

    class Bridge(QObject):
        line = Signal(str)
        status = Signal(str)
        finished = Signal(str, int)
        refresh_chat = Signal()
        refresh_sources = Signal()
        refresh_status = Signal()

    class LocalMindWindow(QMainWindow):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            self.args = args
            self.store = ProjectStore(args.project)
            if args.advanced:
                settings = self.store.settings()
                settings["mode"] = "advanced"
                self.store.update_settings(settings)
            self.main_thread_id = threading.get_ident()
            self.tools = ToolRegistry()
            self.bridge = Bridge()
            self.bridge.line.connect(self.append_log)
            self.bridge.status.connect(self.set_status)
            self.bridge.finished.connect(self.job_finished)
            self.bridge.refresh_chat.connect(self.render_chat)
            self.bridge.refresh_sources.connect(self.refresh_sources)
            self.bridge.refresh_status.connect(self.refresh_status)
            self.server = LocalServer(self.tools, self.bridge.line.emit)
            self.running_jobs: dict[str, threading.Thread] = {}
            self.selected_model: Path | None = None
            self.editing_source_id: str | None = None
            current = self.store.current_model()
            if current:
                self.selected_model = Path(current)
            self.setWindowTitle(f"{APP_NAME} - Offline AI Trainer")
            logo = ROOT / "LocalMind.png"
            if logo.exists():
                self.setWindowIcon(QIcon(str(logo)))
            self.resize(1280, 820)
            self._build_ui()
            self.refresh_all()
            self.status_timer = QTimer(self)
            self.status_timer.timeout.connect(self.refresh_status)
            self.status_timer.start(3000)

        def _build_ui(self) -> None:
            root = QWidget()
            main = QHBoxLayout(root)
            main.setContentsMargins(10, 10, 10, 10)
            main.setSpacing(10)

            sidebar = QFrame()
            sidebar.setFixedWidth(230)
            side = QVBoxLayout(sidebar)
            side.setContentsMargins(10, 10, 10, 10)
            side.setSpacing(8)
            logo_label = QLabel()
            logo_label.setAlignment(Qt.AlignCenter)
            logo_path = ROOT / "LocalMind.png"
            if logo_path.exists():
                pix = QPixmap(str(logo_path))
                logo_label.setPixmap(pix.scaled(QSize(112, 112), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                logo_label.setText("LocalMind")
            title = QLabel("LocalMind")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 22px; font-weight: 700;")
            subtitle = QLabel("offline trainer")
            subtitle.setAlignment(Qt.AlignCenter)
            subtitle.setStyleSheet("color: #9a9a9a;")
            side.addWidget(logo_label)
            side.addWidget(title)
            side.addWidget(subtitle)

            self.status_label = QLabel("Ready")
            self.status_label.setWordWrap(True)
            self.server_label = QLabel("Server: stopped")
            self.server_label.setWordWrap(True)
            self.model_label = QLabel("Model: none")
            self.model_label.setWordWrap(True)
            side.addSpacing(8)
            side.addWidget(self.status_label)
            side.addWidget(self.server_label)
            side.addWidget(self.model_label)

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["normal", "advanced"])
            self.mode_combo.setCurrentText(self.store.settings().get("mode", "normal"))
            self.mode_combo.currentTextChanged.connect(self.mode_changed)
            side.addWidget(QLabel("Mode"))
            side.addWidget(self.mode_combo)

            setup_btn = QPushButton("Bootstrap Core")
            setup_btn.clicked.connect(self.bootstrap_setup)
            train_setup_btn = QPushButton("Install Training Stack")
            train_setup_btn.clicked.connect(self.bootstrap_train_setup)
            install_llama_btn = QPushButton("Install llama.cpp")
            install_llama_btn.clicked.connect(self.install_llama_cpp)
            scan_btn = QPushButton("Rescan Models")
            scan_btn.clicked.connect(self.refresh_models)
            side.addWidget(setup_btn)
            side.addWidget(train_setup_btn)
            side.addWidget(install_llama_btn)
            side.addWidget(scan_btn)
            side.addStretch(1)

            self.tabs = QTabWidget()
            self.tabs.addTab(self._build_chat_tab(), "Chat")
            self.tabs.addTab(self._build_models_tab(), "Models")
            self.tabs.addTab(self._build_knowledge_tab(), "Knowledge")
            self.tabs.addTab(self._build_jobs_tab(), "Jobs")
            self.tabs.addTab(self._build_advanced_tab(), "Advanced")
            self.tabs.addTab(self._build_logs_tab(), "Logs")

            main.addWidget(sidebar)
            main.addWidget(self.tabs, 1)
            self.setCentralWidget(root)

            reload_action = QAction("Refresh", self)
            reload_action.triggered.connect(self.refresh_all)
            self.addAction(reload_action)

        def _build_chat_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            splitter = QSplitter(Qt.Vertical)
            self.chat_view = QTextEdit()
            self.chat_view.setReadOnly(True)
            self.chat_input = QPlainTextEdit()
            self.chat_input.setPlaceholderText("Message the local model. Imported sources are retrieved automatically.")
            self.chat_input.setFixedHeight(120)
            splitter.addWidget(self.chat_view)
            splitter.addWidget(self.chat_input)
            splitter.setStretchFactor(0, 4)
            splitter.setStretchFactor(1, 1)
            controls = QHBoxLayout()
            send_btn = QPushButton("Send")
            send_btn.clicked.connect(self.send_chat)
            start_btn = QPushButton("Start Server")
            start_btn.clicked.connect(self.start_server)
            stop_btn = QPushButton("Stop Server")
            stop_btn.clicked.connect(self.stop_server)
            controls.addWidget(start_btn)
            controls.addWidget(stop_btn)
            controls.addStretch(1)
            controls.addWidget(send_btn)
            layout.addWidget(splitter, 1)
            layout.addLayout(controls)
            return page

        def _build_models_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            self.model_table = QTableWidget(0, 5)
            self.model_table.setHorizontalHeaderLabels(["Name", "Kind", "Status", "Size", "Workflow"])
            self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
            self.model_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.model_table.itemSelectionChanged.connect(self.model_selection_changed)
            self.model_details = QPlainTextEdit()
            self.model_details.setReadOnly(True)
            buttons = QHBoxLayout()
            add_model_btn = QPushButton("Open Model/File")
            add_model_btn.clicked.connect(self.open_model_file)
            set_current_btn = QPushButton("Use Selected")
            set_current_btn.clicked.connect(self.use_selected_model)
            metadata_btn = QPushButton("Refresh Metadata")
            metadata_btn.clicked.connect(self.refresh_models)
            buttons.addWidget(add_model_btn)
            buttons.addWidget(set_current_btn)
            buttons.addWidget(metadata_btn)
            buttons.addStretch(1)
            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(self.model_table)
            splitter.addWidget(self.model_details)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)
            layout.addLayout(buttons)
            layout.addWidget(splitter, 1)
            return page

        def _build_knowledge_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            buttons = QHBoxLayout()
            add_btn = QPushButton("Add Source")
            add_btn.clicked.connect(self.add_source)
            edit_btn = QPushButton("Edit Source Text")
            edit_btn.clicked.connect(self.edit_source)
            del_btn = QPushButton("Delete Source")
            del_btn.clicked.connect(self.delete_source)
            dataset_btn = QPushButton("Build Dataset")
            dataset_btn.clicked.connect(self.build_dataset)
            buttons.addWidget(add_btn)
            buttons.addWidget(edit_btn)
            buttons.addWidget(del_btn)
            buttons.addWidget(dataset_btn)
            buttons.addStretch(1)
            self.source_table = QTableWidget(0, 6)
            self.source_table.setHorizontalHeaderLabels(["Title", "Kind", "Status", "Version", "Chars", "Path"])
            self.source_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.source_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
            self.source_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.source_table.itemSelectionChanged.connect(self.source_selection_changed)
            self.source_preview = QPlainTextEdit()
            self.source_preview.setReadOnly(True)
            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(self.source_table)
            splitter.addWidget(self.source_preview)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)
            layout.addLayout(buttons)
            layout.addWidget(splitter, 1)
            return page

        def _build_jobs_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            grid = QGridLayout()
            self.base_model_edit = QLineEdit()
            self.base_model_edit.setPlaceholderText("HF model folder or model id for LoRA training")
            browse_base = QPushButton("Browse Base")
            browse_base.clicked.connect(self.browse_base_model)
            train_btn = QPushButton("Train LoRA")
            train_btn.clicked.connect(self.train_lora)
            merge_btn = QPushButton("Merge Adapter")
            merge_btn.clicked.connect(self.merge_adapter)
            quant_btn = QPushButton("Quantize GGUF")
            quant_btn.clicked.connect(self.quantize_model)
            imatrix_btn = QPushButton("Build iMatrix")
            imatrix_btn.clicked.connect(self.build_imatrix)
            export_btn = QPushButton("Convert HF to GGUF")
            export_btn.clicked.connect(self.convert_hf_to_gguf)
            grid.addWidget(QLabel("Base model"), 0, 0)
            grid.addWidget(self.base_model_edit, 0, 1)
            grid.addWidget(browse_base, 0, 2)
            grid.addWidget(train_btn, 1, 0)
            grid.addWidget(merge_btn, 1, 1)
            grid.addWidget(export_btn, 1, 2)
            grid.addWidget(quant_btn, 2, 0)
            grid.addWidget(imatrix_btn, 2, 1)
            self.progress = QProgressBar()
            self.progress.setRange(0, 0)
            self.progress.hide()
            self.job_list = QListWidget()
            layout.addLayout(grid)
            layout.addWidget(self.progress)
            layout.addWidget(self.job_list, 1)
            return page

        def _build_advanced_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            form_box = QGroupBox("Generation")
            form = QFormLayout(form_box)
            settings = self.store.settings()
            self.temperature = self.double_spin(settings["temperature"], 0.0, 2.0, 0.05)
            self.top_p = self.double_spin(settings["top_p"], 0.0, 1.0, 0.01)
            self.max_tokens = self.spin(settings["max_tokens"], 1, 16384)
            self.context_length = self.spin(settings["context_length"], 512, 1048576)
            self.gpu_layers = self.spin(settings["gpu_layers"], -1, 999)
            self.threads = self.spin(settings["threads"], 1, 256)
            form.addRow("Temperature", self.temperature)
            form.addRow("Top P", self.top_p)
            form.addRow("Max tokens", self.max_tokens)
            form.addRow("Context length", self.context_length)
            form.addRow("GPU layers", self.gpu_layers)
            form.addRow("Threads", self.threads)

            train_box = QGroupBox("LoRA / QLoRA")
            train_form = QFormLayout(train_box)
            self.lora_rank = self.spin(settings["lora_rank"], 1, 1024)
            self.lora_alpha = self.spin(settings["lora_alpha"], 1, 4096)
            self.lora_dropout = self.double_spin(settings["lora_dropout"], 0.0, 0.9, 0.01)
            self.learning_rate = self.double_spin(settings["learning_rate"], 1e-7, 1e-1, 1e-5, decimals=7)
            self.epochs = self.double_spin(settings["epochs"], 0.01, 100.0, 0.25)
            self.batch_size = self.spin(settings["batch_size"], 1, 1024)
            self.gradient_accumulation = self.spin(settings["gradient_accumulation"], 1, 1024)
            self.max_seq_length = self.spin(settings["max_seq_length"], 128, 1048576)
            self.target_modules = QLineEdit(str(settings["target_modules"]))
            self.load_4bit = QCheckBox()
            self.load_4bit.setChecked(bool(settings["load_4bit"]))
            self.packing = QCheckBox()
            self.packing.setChecked(bool(settings["packing"]))
            self.quant_preset = QComboBox()
            self.quant_preset.addItems(list(QUANT_PRESETS))
            self.quant_preset.setCurrentText(str(settings.get("quant_preset", "Fast Q4_K_M")))
            train_form.addRow("Rank", self.lora_rank)
            train_form.addRow("Alpha", self.lora_alpha)
            train_form.addRow("Dropout", self.lora_dropout)
            train_form.addRow("Learning rate", self.learning_rate)
            train_form.addRow("Epochs", self.epochs)
            train_form.addRow("Batch size", self.batch_size)
            train_form.addRow("Grad accumulation", self.gradient_accumulation)
            train_form.addRow("Max sequence", self.max_seq_length)
            train_form.addRow("Target modules", self.target_modules)
            train_form.addRow("Load 4-bit", self.load_4bit)
            train_form.addRow("Packing", self.packing)
            train_form.addRow("Quant preset", self.quant_preset)

            prompt_box = QGroupBox("System Prompt")
            prompt_layout = QVBoxLayout(prompt_box)
            self.system_prompt = QPlainTextEdit()
            self.system_prompt.setPlainText(str(settings.get("system_prompt", read_prompt_file())))
            prompt_layout.addWidget(self.system_prompt)
            save_btn = QPushButton("Save Settings")
            save_btn.clicked.connect(self.save_settings)
            layout.addWidget(form_box)
            layout.addWidget(train_box)
            layout.addWidget(prompt_box, 1)
            layout.addWidget(save_btn)
            return page

        def _build_logs_tab(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            buttons = QHBoxLayout()
            refresh_btn = QPushButton("Refresh Logs")
            refresh_btn.clicked.connect(self.load_logs)
            clear_btn = QPushButton("Clear View")
            clear_btn.clicked.connect(self.log_view.clear)
            buttons.addWidget(refresh_btn)
            buttons.addWidget(clear_btn)
            buttons.addStretch(1)
            layout.addLayout(buttons)
            layout.addWidget(self.log_view, 1)
            return page

        def spin(self, value: int, minimum: int, maximum: int) -> Any:
            widget = QSpinBox()
            widget.setRange(minimum, maximum)
            widget.setValue(int(value))
            return widget

        def double_spin(
            self, value: float, minimum: float, maximum: float, step: float, decimals: int = 4
        ) -> Any:
            widget = QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setDecimals(decimals)
            widget.setValue(float(value))
            return widget

        def mode_changed(self, value: str) -> None:
            settings = self.store.settings()
            settings["mode"] = value
            self.store.update_settings(settings)
            self.set_status(f"Mode: {value}")

        def save_settings(self) -> None:
            settings = self.store.settings()
            settings.update(
                {
                    "temperature": self.temperature.value(),
                    "top_p": self.top_p.value(),
                    "max_tokens": self.max_tokens.value(),
                    "context_length": self.context_length.value(),
                    "gpu_layers": self.gpu_layers.value(),
                    "threads": self.threads.value(),
                    "lora_rank": self.lora_rank.value(),
                    "lora_alpha": self.lora_alpha.value(),
                    "lora_dropout": self.lora_dropout.value(),
                    "learning_rate": self.learning_rate.value(),
                    "epochs": self.epochs.value(),
                    "batch_size": self.batch_size.value(),
                    "gradient_accumulation": self.gradient_accumulation.value(),
                    "max_seq_length": self.max_seq_length.value(),
                    "target_modules": self.target_modules.text(),
                    "load_4bit": self.load_4bit.isChecked(),
                    "packing": self.packing.isChecked(),
                    "quant_preset": self.quant_preset.currentText(),
                    "system_prompt": self.system_prompt.toPlainText(),
                }
            )
            self.store.update_settings(settings)
            self.set_status("Settings saved")

        def refresh_all(self) -> None:
            self.refresh_models()
            self.refresh_sources()
            self.refresh_status()
            self.load_logs()
            self.render_chat()

        def refresh_status(self) -> None:
            self.server_label.setText(
                f"Server: running {self.server.host}:{self.server.port}" if self.server.is_running() else "Server: stopped"
            )
            model = self.store.current_model()
            self.model_label.setText(f"Model: {Path(model).name}" if model else "Model: none")

        def set_status(self, text: str) -> None:
            self.status_label.setText(text)
            append_log(text)

        def append_log(self, line: str) -> None:
            if not line:
                return
            if getattr(self, "main_thread_id", None) != threading.get_ident():
                if hasattr(self, "bridge"):
                    self.bridge.line.emit(line)
                else:
                    append_log(line)
                return
            self.log_view.appendPlainText(line) if hasattr(self, "log_view") else None
            cursor = self.log_view.textCursor() if hasattr(self, "log_view") else None
            if cursor:
                cursor.movePosition(QTextCursor.End)
                self.log_view.setTextCursor(cursor)
            append_log(line)

        def load_logs(self) -> None:
            if not hasattr(self, "log_view"):
                return
            lines: list[str] = []
            for name in ("localmind.log", "bootstrap.log", "commands.log", "llama-server.log"):
                path = LOG_DIR / name
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="replace")
                    lines.append(f"--- {name} ---\n{content[-12000:]}")
            self.log_view.setPlainText("\n\n".join(lines))

        def render_chat(self) -> None:
            lines = []
            if self.store.conversation_summary():
                lines.append("[compacted summary]\n" + self.store.conversation_summary() + "\n")
            for item in self.store.chat_history():
                role = item.get("role", "user").upper()
                lines.append(f"{role}: {item.get('content', '')}")
            self.chat_view.setPlainText("\n\n".join(lines))
            self.chat_view.moveCursor(QTextCursor.End)

        def send_chat(self) -> None:
            text = self.chat_input.toPlainText().strip()
            if not text:
                return
            self.chat_input.clear()
            self.store.add_chat("user", text)
            self.render_chat()

            def work() -> None:
                try:
                    messages = build_chat_messages(self.store, text)
                    if self.server.is_running():
                        answer = self.server.chat(messages, self.store.settings()).strip()
                    else:
                        hits = self.store.search(text, limit=3)
                        if hits:
                            answer = (
                                "llama-server is not running, so I searched the local knowledge pack instead.\n\n"
                                + "\n\n".join(str(hit.get("text", ""))[:1000] for hit in hits)
                            )
                        else:
                            answer = (
                                "llama-server is not running. Start a GGUF model server from the Chat or Models tab "
                                "to get live model responses."
                            )
                    self.store.add_chat("assistant", answer)
                    self.bridge.line.emit("Chat response received")
                except Exception as exc:
                    self.store.add_chat("assistant", f"LocalMind error: {exc}")
                    self.bridge.line.emit(traceback.format_exc())
                self.bridge.refresh_chat.emit()

            threading.Thread(target=work, daemon=True).start()

        def refresh_models(self) -> None:
            capabilities = scan_workspace_for_models()
            current = self.store.current_model()
            self.model_table.setRowCount(0)
            for capability in capabilities:
                row = self.model_table.rowCount()
                self.model_table.insertRow(row)
                values = [
                    capability.path.name,
                    capability.kind,
                    capability.status,
                    format_bytes(capability.details.get("size")),
                    capability.workflow,
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setData(Qt.UserRole, str(capability.path))
                    self.model_table.setItem(row, col, item)
                if current and Path(current) == capability.path:
                    for col in range(self.model_table.columnCount()):
                        self.model_table.item(row, col).setBackground(qt["QColor"]("#101a32") if "QColor" in qt else None)
            self.model_table.resizeRowsToContents()

        def model_selection_changed(self) -> None:
            rows = self.model_table.selectionModel().selectedRows()
            if not rows:
                return
            row = rows[0].row()
            item = self.model_table.item(row, 0)
            if not item:
                return
            path = Path(item.data(Qt.UserRole))
            capability = detect_file_capability(path)
            self.selected_model = path
            self.model_details.setPlainText(json.dumps(dataclasses.asdict(capability), indent=2, default=str))

        def open_model_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open model or AI file",
                str(ROOT),
                "AI files (*.gguf *.safetensors *.bin *.pt *.pth *.onnx *.tflite *.mlmodel);;All files (*)",
            )
            if path:
                capability = detect_file_capability(Path(path))
                self.model_details.setPlainText(json.dumps(dataclasses.asdict(capability), indent=2, default=str))
                if Path(path).suffix.lower() == ".gguf":
                    self.selected_model = Path(path)

        def use_selected_model(self) -> None:
            if not self.selected_model:
                self.model_selection_changed()
            if not self.selected_model:
                QMessageBox.warning(self, "LocalMind", "Select a model first.")
                return
            self.store.set_current_model(str(self.selected_model))
            self.set_status(f"Current model: {self.selected_model.name}")
            self.refresh_status()

        def start_server(self) -> None:
            model = self.selected_model or (Path(self.store.current_model()) if self.store.current_model() else None)
            if not model or not model.exists():
                QMessageBox.warning(self, "LocalMind", "Select a GGUF model first.")
                return
            if model.suffix.lower() != ".gguf":
                QMessageBox.warning(self, "LocalMind", "llama-server requires a GGUF model.")
                return
            self.store.set_current_model(str(model))

            def work() -> None:
                try:
                    self.server.start(model, self.store.settings())
                    self.bridge.status.emit("Server ready")
                except Exception:
                    self.bridge.line.emit(traceback.format_exc())
                    self.bridge.status.emit("Server failed")
                self.bridge.refresh_status.emit()

            threading.Thread(target=work, daemon=True).start()

        def stop_server(self) -> None:
            self.server.stop()
            self.refresh_status()

        def refresh_sources(self) -> None:
            rows = self.store.list_sources()
            self.source_table.setRowCount(0)
            for source in rows:
                row = self.source_table.rowCount()
                self.source_table.insertRow(row)
                values = [
                    source["title"],
                    source["kind"],
                    source["status"],
                    source["version"],
                    source["chars"],
                    source["path"],
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setData(Qt.UserRole, source["id"])
                    self.source_table.setItem(row, col, item)
            self.source_table.resizeRowsToContents()

        def selected_source_id(self) -> str | None:
            rows = self.source_table.selectionModel().selectedRows()
            if not rows:
                return None
            item = self.source_table.item(rows[0].row(), 0)
            return str(item.data(Qt.UserRole)) if item else None

        def source_selection_changed(self) -> None:
            source_id = self.selected_source_id()
            if not source_id:
                return
            chunks = []
            with self.store._connect() as conn:
                rows = conn.execute(
                    "SELECT text FROM chunks WHERE source_id = ? ORDER BY ordinal LIMIT 20",
                    (source_id,),
                ).fetchall()
                chunks = [row["text"] for row in rows]
            self.source_preview.setPlainText("\n\n--- chunk ---\n\n".join(chunks))

        def add_source(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Add knowledge sources",
                str(ROOT),
                "Sources (*.txt *.md *.markdown *.pdf *.docx *.pptx *.xlsx *.csv *.html *.json *.jsonl *.png *.jpg *.jpeg *.webp *.bmp *.tiff);;All files (*)",
            )
            if not files:
                return

            def work() -> None:
                for file in files:
                    try:
                        source_id = self.store.add_or_update_source(Path(file))
                        self.bridge.line.emit(f"Imported source {Path(file).name} as {source_id}")
                    except Exception:
                        self.bridge.line.emit(traceback.format_exc())
                self.bridge.refresh_sources.emit()

            threading.Thread(target=work, daemon=True).start()

        def edit_source(self) -> None:
            source_id = self.selected_source_id()
            if not source_id:
                QMessageBox.warning(self, "LocalMind", "Select a source first.")
                return
            if self.source_preview.isReadOnly() or self.editing_source_id != source_id:
                self.editing_source_id = source_id
                self.source_preview.setReadOnly(False)
                self.source_preview.setFocus()
                self.set_status("Editing source preview. Click Edit Source Text again to save.")
                return
            reply = QMessageBox.question(self, "LocalMind", "Save this edited text as a new source version?")
            if reply == QMessageBox.Yes:
                self.store.update_source_text(source_id, self.source_preview.toPlainText())
                self.source_preview.setReadOnly(True)
                self.editing_source_id = None
                self.refresh_sources()
                self.set_status("Source updated")
            else:
                self.source_preview.setReadOnly(True)
                self.editing_source_id = None

        def delete_source(self) -> None:
            source_id = self.selected_source_id()
            if not source_id:
                QMessageBox.warning(self, "LocalMind", "Select a source first.")
                return
            reply = QMessageBox.question(
                self,
                "LocalMind",
                "Delete this source from active knowledge? Existing exported models cannot be surgically unlearned; rebuild/retrain from base to exclude it.",
            )
            if reply == QMessageBox.Yes:
                self.store.delete_source(source_id)
                self.refresh_sources()
                self.set_status("Source deleted from active knowledge")

        def build_dataset(self) -> None:
            path = self.store.dataset_jsonl()
            if path.stat().st_size == 0:
                QMessageBox.warning(self, "LocalMind", "No active source chunks are available for a dataset.")
                self.set_status("Dataset build skipped: no active source chunks")
                return
            self.set_status(f"Dataset written: {path}")
            self.append_log(f"Dataset written: {path}")

        def browse_base_model(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select Hugging Face base model folder", str(ROOT))
            if path:
                self.base_model_edit.setText(path)

        def bootstrap_setup(self) -> None:
            def work() -> None:
                try:
                    bootstrap = Bootstrapper(
                        no_network=self.args.no_network,
                        line_cb=self.bridge.line.emit,
                        python_target=self.args.python_version,
                    )
                    bootstrap.bootstrap_all_light()
                    self.bridge.status.emit("Core bootstrap complete")
                except Exception:
                    self.bridge.line.emit(traceback.format_exc())
                    self.bridge.status.emit("Bootstrap failed")

            self.run_thread_job("bootstrap", work)

        def bootstrap_train_setup(self) -> None:
            def work() -> None:
                try:
                    bootstrap = Bootstrapper(
                        no_network=self.args.no_network,
                        line_cb=self.bridge.line.emit,
                        python_target=self.args.python_version,
                    )
                    bootstrap.bootstrap_train()
                    self.bridge.status.emit("Training stack ready")
                except Exception:
                    self.bridge.line.emit(traceback.format_exc())
                    self.bridge.status.emit("Training setup failed")

            self.run_thread_job("training setup", work)

        def install_llama_cpp(self) -> None:
            if self.args.no_network:
                QMessageBox.warning(
                    self,
                    "LocalMind",
                    "llama.cpp installation needs network access. Restart without --no-network to install it.",
                )
                self.set_status("llama.cpp install skipped: --no-network")
                return
            self.run_thread_job("install llama.cpp", self.install_llama_cpp_with_fallbacks)

        def install_llama_cpp_with_fallbacks(self) -> None:
            failures: list[str] = []
            for backend in self.tools.backend_order():
                self.bridge.line.emit(f"Trying llama.cpp backend: {backend}")
                try:
                    for command in self.tools.install_commands(backend):
                        cmd = command.argv if isinstance(command, CommandSpec) else command
                        cwd = command.cwd if isinstance(command, CommandSpec) else ROOT
                        env = command.env if isinstance(command, CommandSpec) else None
                        self.bridge.line.emit(f"$ {command_display(cmd)}")
                        run_checked(cmd, cwd=cwd or ROOT, env=env, line_cb=self.bridge.line.emit)
                    self.bridge.status.emit(f"llama.cpp ready ({backend})")
                    return
                except Exception as exc:
                    failures.append(f"{backend}: {exc}")
                    self.bridge.line.emit(f"Backend {backend} failed: {exc}")
                    if requested_llama_backend():
                        break
            raise RuntimeError("All llama.cpp backend builds failed:\n" + "\n".join(failures))

        def run_thread_job(self, name: str, target: Callable[[], None]) -> None:
            self.progress.show()
            self.job_list.addItem(f"{utc_now()} started: {name}")

            def wrapped() -> None:
                code = 0
                try:
                    target()
                except Exception:
                    code = 1
                    self.bridge.line.emit(traceback.format_exc())
                self.bridge.finished.emit(name, code)

            thread = threading.Thread(target=wrapped, daemon=True)
            self.running_jobs[name] = thread
            thread.start()

        def run_command_job(self, name: str, commands: list[Any]) -> None:
            def target() -> None:
                for command in commands:
                    if isinstance(command, CommandSpec):
                        cmd = command.argv
                        cwd = command.cwd or ROOT
                        env = command.env
                    else:
                        cmd = command
                        cwd = ROOT
                        env = None
                    self.bridge.line.emit(f"$ {command_display(cmd)}")
                    run_checked(cmd, cwd=cwd, env=env, line_cb=self.bridge.line.emit)

            self.run_thread_job(name, target)

        def job_finished(self, name: str, code: int) -> None:
            self.progress.hide()
            status = "finished" if code == 0 else "failed"
            self.job_list.addItem(f"{utc_now()} {status}: {name}")
            self.set_status(f"{name} {status}")
            self.refresh_models()

        def train_lora(self) -> None:
            base = self.base_model_edit.text().strip()
            if not base:
                QMessageBox.warning(self, "LocalMind", "Choose a Hugging Face base model folder or model id first.")
                return
            settings = self.store.settings()
            dataset = self.store.dataset_jsonl()
            if dataset.stat().st_size == 0:
                QMessageBox.warning(self, "LocalMind", "No active source chunks are available for training.")
                return
            train_py = python_in_venv(ENVS_DIR / "train")
            if not train_py.exists():
                QMessageBox.warning(
                    self,
                    "LocalMind",
                    "Train environment is not installed yet. Run Bootstrap Setup with network enabled.",
                )
                return
            job_dir = self.store.jobs_dir / f"train_{int(time.time())}"
            job_dir.mkdir(parents=True, exist_ok=True)
            script = write_training_script(job_dir)
            output = self.store.exports_dir / f"adapter_{int(time.time())}"
            cmd = [
                str(train_py),
                str(script),
                "--base-model",
                base,
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--rank",
                str(settings["lora_rank"]),
                "--alpha",
                str(settings["lora_alpha"]),
                "--dropout",
                str(settings["lora_dropout"]),
                "--learning-rate",
                str(settings["learning_rate"]),
                "--epochs",
                str(settings["epochs"]),
                "--batch-size",
                str(settings["batch_size"]),
                "--gradient-accumulation",
                str(settings["gradient_accumulation"]),
                "--max-seq-length",
                str(settings["max_seq_length"]),
                "--target-modules",
                str(settings["target_modules"]),
            ]
            if settings.get("load_4bit"):
                cmd.append("--load-4bit")
            if settings.get("packing"):
                cmd.append("--packing")
            self.run_command_job("train LoRA", [cmd])

        def merge_adapter(self) -> None:
            base = self.base_model_edit.text().strip()
            if not base:
                QMessageBox.warning(self, "LocalMind", "Choose the base model first.")
                return
            adapter = QFileDialog.getExistingDirectory(self, "Select PEFT adapter folder", str(self.store.exports_dir))
            if not adapter:
                return
            train_py = python_in_venv(ENVS_DIR / "train")
            if not train_py.exists():
                QMessageBox.warning(self, "LocalMind", "Train environment is not installed.")
                return
            job_dir = self.store.jobs_dir / f"merge_{int(time.time())}"
            job_dir.mkdir(parents=True, exist_ok=True)
            script = write_merge_script(job_dir)
            output = self.store.exports_dir / f"merged_{int(time.time())}"
            cmd = [
                str(train_py),
                str(script),
                "--base-model",
                base,
                "--adapter",
                adapter,
                "--output",
                str(output),
                "--safe-merge",
            ]
            self.run_command_job("merge adapter", [cmd])

        def convert_hf_to_gguf(self) -> None:
            model_dir = QFileDialog.getExistingDirectory(self, "Select merged/HF model folder", str(self.store.exports_dir))
            if not model_dir:
                return
            convert = self.tools.llama_root / "convert_hf_to_gguf.py"
            if not convert.exists():
                QMessageBox.warning(self, "LocalMind", "llama.cpp converter was not found. Install llama.cpp first.")
                return
            train_py = python_in_venv(ENVS_DIR / "train")
            py = train_py if train_py.exists() else Path(sys.executable)
            outfile = self.store.exports_dir / f"{Path(model_dir).name}.f16.gguf"
            cmd = [str(py), str(convert), model_dir, "--outfile", str(outfile), "--outtype", "f16"]
            self.run_command_job("convert HF to GGUF", [cmd])

        def quantize_model(self) -> None:
            model = self.selected_model or (Path(self.store.current_model()) if self.store.current_model() else None)
            if not model or not model.exists() or model.suffix.lower() != ".gguf":
                path, _ = QFileDialog.getOpenFileName(self, "Select GGUF to quantize", str(ROOT), "GGUF (*.gguf)")
                if not path:
                    return
                model = Path(path)
            quant = self.tools.find_tool("llama-quantize")
            if not quant:
                QMessageBox.warning(self, "LocalMind", "llama-quantize was not found. Install llama.cpp first.")
                return
            settings = self.store.settings()
            preset = QUANT_PRESETS.get(settings.get("quant_preset", "Fast Q4_K_M"), QUANT_PRESETS["Fast Q4_K_M"])
            quant_type = preset["type"]
            output = self.store.exports_dir / f"{model.stem}.{quant_type.lower()}.gguf"
            cmd = [str(quant)]
            imatrix = ROOT / "imatrix.dat"
            if preset.get("imatrix") and imatrix.exists():
                cmd.extend(["--imatrix", str(imatrix)])
            cmd.extend([str(model), str(output), quant_type, str(settings.get("threads", 4))])
            self.run_command_job(f"quantize {quant_type}", [cmd])

        def build_imatrix(self) -> None:
            model = self.selected_model or (Path(self.store.current_model()) if self.store.current_model() else None)
            if not model or not model.exists() or model.suffix.lower() != ".gguf":
                QMessageBox.warning(self, "LocalMind", "Select a GGUF model first.")
                return
            imatrix_tool = self.tools.find_tool("llama-imatrix")
            if not imatrix_tool:
                QMessageBox.warning(self, "LocalMind", "llama-imatrix was not found. Install llama.cpp first.")
                return
            corpus = ROOT / "prompt.txt"
            dataset = self.store.dataset_jsonl()
            if dataset.exists() and dataset.stat().st_size > 0:
                corpus = dataset
            output = self.store.exports_dir / f"{model.stem}.imatrix.dat"
            cmd = [str(imatrix_tool), "-m", str(model), "-f", str(corpus), "-o", str(output)]
            self.run_command_job("build iMatrix", [cmd])

    return LocalMindWindow


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalMind offline AI trainer")
    parser.add_argument("--no-network", action="store_true", help="Do not download or install anything.")
    parser.add_argument("--bootstrap-only", action="store_true", help="Run bootstrap checks/setup and exit.")
    parser.add_argument("--full-setup", action="store_true", help="With --bootstrap-only, also install the training stack.")
    parser.add_argument("--reset-envs", action="store_true", help="Delete LocalMind managed envs before continuing.")
    parser.add_argument("--advanced", action="store_true", help="Start in advanced mode.")
    parser.add_argument(
        "--python-version",
        default=os.environ.get("LOCALMIND_PYTHON", PYTHON_TARGET),
        help="Managed venv Python version, for example 3.13 or 3.12.",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project name.")
    parser.add_argument("--serve-local", action="store_true", help="Serve the selected/default GGUF model without GUI.")
    parser.add_argument("--model", help="Model path for --serve-local.")
    parser.add_argument("--run-app", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_dirs()
    clear_dead_server_record()
    if args.reset_envs and ENVS_DIR.exists():
        shutil.rmtree(ENVS_DIR)
        ENVS_DIR.mkdir(parents=True, exist_ok=True)
        append_log("reset managed envs")
    if args.bootstrap_only:
        bootstrap = Bootstrapper(no_network=args.no_network, line_cb=print, python_target=args.python_version)
        if args.full_setup:
            bootstrap.bootstrap_all_full()
        else:
            bootstrap.bootstrap_all_light()
        return 0
    if args.serve_local:
        return run_headless_server(args)
    return run_gui(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        append_log(traceback.format_exc(), "fatal.log")
        print(f"LocalMind failed: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)
