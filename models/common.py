"""Shared modelling utilities: device selection, seeding, MLflow, config I/O.

Kept deliberately tiny and dependency-light so both the acoustic (Phase 2) and
vibration (Phase 4) pipelines share one source of truth for the handful of
cross-cutting concerns (reproducibility, the MPS/CPU device dance, and where
MLflow writes its runs).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Repo root: models/ -> repo root. Every path helper anchors here so the code
# behaves identically regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = REPO_ROOT / "mlruns"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse a YAML config file into a plain dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if imported) PyTorch for reproducible runs.

    Torch is imported lazily so pure-NumPy callers (e.g. the dataset builder and
    most unit tests) do not pay the multi-second import cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        # MPS shares the CPU generator seed; this covers both backends.
        if torch.cuda.is_available():  # pragma: no cover - no CUDA on this laptop
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dep, defensive only
        pass


def select_device(prefer: str = "auto") -> str:
    """Return the torch device string to train on.

    ``auto`` picks Apple-Silicon ``mps`` when available and falls back to ``cpu``.
    A ``prefer`` of ``"cpu"``/``"mps"`` forces that choice (used by tests and to
    work around backend-specific issues). We never assume CUDA here — this is a
    laptop pipeline.
    """
    import torch

    if prefer == "cpu":
        return "cpu"
    if prefer == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if prefer != "auto":
        raise ValueError(f"unknown device preference {prefer!r}")
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class MlflowRun:
    """Handle to an active-or-noop MLflow run context.

    We wrap MLflow so a run can be disabled (``enabled=False``) without peppering
    the training code with conditionals — handy in tests and smoke runs.
    """

    enabled: bool
    experiment: str


def mlflow_setup(experiment: str, *, enabled: bool = True) -> MlflowRun:
    """Point MLflow at the repo-local ``./mlruns`` store and select an experiment.

    Using a file-based tracking URI keeps every run on disk under the repo (the
    directory is gitignored) with no server to stand up.
    """
    if enabled:
        # MLflow >= 3 gates the file-based tracking store behind an opt-in flag
        # (it is in maintenance mode). We keep the repo-local ./mlruns store the
        # task asks for — no tracking server, everything on disk — by opting in.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        MLRUNS_DIR.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
        mlflow.set_experiment(experiment)
    return MlflowRun(enabled=enabled, experiment=experiment)
