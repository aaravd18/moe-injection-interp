"""Environment and version reporting (used by `doctor` and stored in metadata.json)."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(pkg: str) -> str | None:
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return None


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Commit hash of this code checkout (with a '-dirty' suffix if there are uncommitted changes)."""
    repo_dir = Path(repo_dir or Path(__file__).resolve().parent.parent)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "safetensors": _version("safetensors"),
        "accelerate": _version("accelerate"),
        "numpy": _version("numpy"),
        "code_git_commit": git_commit(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "in_colab": "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ,
    }
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name, "total_memory_gb": round(p.total_memory / 1e9, 2),
                         "compute_capability": f"{p.major}.{p.minor}"})
        info["gpus"] = gpus
        try:
            info["bf16_supported"] = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def doctor_report(output_dir: str | Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """Return (info, warnings) describing whether this machine can run experiments."""
    info = environment_info()
    warnings: list[str] = []
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    info["hf_home"] = hf_home
    info["hf_token_set"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    try:
        p = Path(hf_home)
        p.mkdir(parents=True, exist_ok=True)
        info["hf_home_free_gb"] = round(shutil.disk_usage(p).free / 1e9, 1)
        if info["hf_home_free_gb"] < 20:
            warnings.append(f"Only {info['hf_home_free_gb']} GB free in HF_HOME ({hf_home}); OLMoE needs ~14 GB, Mellum2.1 ~24 GB.")
    except Exception:
        pass
    if output_dir:
        p = Path(output_dir)
        existing = p if p.exists() else next((q for q in p.parents if q.exists()), Path("."))
        info["output_dir_free_gb"] = round(shutil.disk_usage(existing).free / 1e9, 1)
    if not info["cuda_available"]:
        warnings.append("CUDA is not available: fine for analysis/tests, but the 7B model needs a GPU (~14 GB in bf16).")
    else:
        mem = max(g["total_memory_gb"] for g in info["gpus"])
        if mem < 16:
            warnings.append(f"Largest GPU has {mem} GB; bf16 weights need ~14 GB (OLMoE) or ~24 GB (Mellum2.1) plus activations.")
        if not info.get("bf16_supported"):
            warnings.append("GPU lacks native bf16 (pre-Ampere, e.g. T4); dtype=auto will use float16.")
    return info, warnings
