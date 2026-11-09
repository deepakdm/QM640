"""Record the computing environment for the abstract and Methods (rubric: "technology used, incl. CPU/GPU").

    python scripts/env_report.py          # -> outputs/final/environment.json

Run it on the machine that built the embeddings and ran the re-ranker. The LLM ran on the
provider's hosted infrastructure through the API, so no local GPU is needed for generation.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ["anthropic", "sentence-transformers", "torch", "faiss-cpu", "rank-bm25", "numpy", "pandas",
            "scipy", "scikit-learn", "statsmodels", "matplotlib", "python-docx", "pydantic"]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        if m:
            return m.group(1).strip()
    if sys.platform == "win32":
        out = _run(["wmic", "cpu", "get", "name"]).splitlines()
        if len(out) > 1:
            return out[1].strip()
    return platform.processor() or "unknown"


def ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
            return round(kb / 1024**2, 1)
        if sys.platform == "darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 1024**3, 1)
        if sys.platform == "win32":
            import ctypes

            class M(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("t", ctypes.c_ulonglong)] + \
                           [(f"x{i}", ctypes.c_ulonglong) for i in range(6)]
            s = M(); s.l = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.t / 1024**3, 1)
    except Exception:
        return None
    return None


def gpu() -> dict:
    info = {"cuda": False, "mps": False, "name": None}
    try:
        import torch
        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if info["cuda"]:
            info["name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["note"] = "torch not importable"
    return info


def main() -> None:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    env = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
        "cpu": cpu_model(), "logical_cores": os.cpu_count(), "ram_gb": ram_gb(), "gpu": gpu(),
        "python": platform.python_version(), "packages": versions,
        "generation": "Hosted LLM via provider API (no local GPU used for generation)",
    }
    g = env["gpu"]
    accel = g["name"] if g["cuda"] else ("Apple GPU (MPS) available" if g["mps"] else "no GPU used")
    env["sentence_for_report"] = (
        f"Retrieval (embedding, BM25 and cross-encoder re-ranking) and all analysis ran on one "
        f"{env['cpu']} CPU ({env['logical_cores']} logical cores, {env['ram_gb']} GB RAM; {accel}) "
        f"under {env['os']} with Python {env['python']}; the language model ran on the provider's hosted "
        f"infrastructure through its API.")
    out = Path("outputs/final/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2))
    print(env["sentence_for_report"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
