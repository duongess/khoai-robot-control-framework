"""Safe, atomic local persistence for complete SAC learner checkpoints."""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 1
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def validate_model_name(value: str) -> str:
    """Return a safe checkpoint identifier, rejecting file-system paths."""
    name = value.strip()
    if not name:
        raise ValueError("a model name is required; use letters, digits, '.', '_' or '-'")
    if not _MODEL_NAME.fullmatch(name):
        raise ValueError("model name must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")
    return name


def checkpoint_path(checkpoint_dir: str | Path, model_name: str) -> Path:
    """Resolve a model identifier beneath the configured checkpoint directory."""
    root = Path(checkpoint_dir).expanduser().resolve()
    return root / f"{validate_model_name(model_name)}.pt"


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a locally-created checkpoint and validate its outer structure.

    ``weights_only=True`` deliberately refuses arbitrary pickle globals. Do not
    load a checkpoint from an untrusted source.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f"could not load checkpoint {path.name}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"checkpoint {path.name} has an unsupported format")
    if not isinstance(payload.get("learner_config"), dict) or not isinstance(payload.get("agent"), dict):
        raise ValueError(f"checkpoint {path.name} is missing learner configuration or SAC state")
    return payload


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically so an interrupted save preserves the old model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except (OSError, RuntimeError, ValueError) as error:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise ValueError(f"could not save checkpoint {path.name}: {error}") from error
