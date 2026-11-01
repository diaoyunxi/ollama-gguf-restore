"""High-level discovery helpers built on :mod:`manifest`."""

from __future__ import annotations

import os
from typing import List, Optional

from .manifest import ModelRef, default_models_dir, find_model, list_models


def models_dir(explicit: Optional[str] = None) -> str:
    """Return the Ollama models directory to operate on.

    Resolution order: explicit argument, ``OLLAMA_MODELS`` env var, then the
    platform default (``~/.ollama/models``).
    """
    if explicit:
        return os.path.abspath(explicit)
    return default_models_dir()


def discover(explicit_dir: Optional[str] = None) -> tuple[List[ModelRef], str]:
    """List all models; returns ``(models, models_dir_used)``."""
    base = models_dir(explicit_dir)
    return list_models(base), base


def resolve(reference: str, explicit_dir: Optional[str] = None) -> tuple[ModelRef, str]:
    """Resolve a ``name:tag`` reference; returns ``(ref, models_dir_used)``."""
    base = models_dir(explicit_dir)
    ref = find_model(base, reference)
    if ref is None:
        available = [m.reference for m in list_models(base)]
        hint = f" (available: {', '.join(available) or 'none'})"
        raise LookupError(
            f"model {reference!r} not found under {base}{hint}"
        )
    return ref, base
