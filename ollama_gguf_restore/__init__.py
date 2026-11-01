"""ollama-gguf-restore: recover standalone GGUF files from Ollama's blob storage.

Ollama stores pulled models as content-addressed blobs under
``~/.ollama/models``:

* ``manifests/<registry>/<ns>/<name>/<tag>``  -- JSON manifests
* ``blobs/sha256-<hex>``                        -- the actual layer bytes

The layer with media type ``application/vnd.ollama.image.model`` is a complete
GGUF file (or one shard of a multi-file GGUF). This package parses the
manifests, locates those blobs, verifies the GGUF structure and writes clean
``.gguf`` files out.
"""

from .discovery import discover, resolve, models_dir
from .manifest import (
    ModelRef,
    ModelConfig,
    Layer,
    default_models_dir,
    list_models,
    find_model,
    blob_name,
)
from .restore import restore_model, default_output_name, default_output_path, RestoreResult
from .gguf import parse_gguf_header, is_gguf, sniff_header, GgufInfo

__all__ = [
    "discover",
    "resolve",
    "models_dir",
    "ModelRef",
    "ModelConfig",
    "Layer",
    "default_models_dir",
    "list_models",
    "find_model",
    "blob_name",
    "restore_model",
    "default_output_name",
    "default_output_path",
    "RestoreResult",
    "parse_gguf_header",
    "is_gguf",
    "sniff_header",
    "GgufInfo",
]

__version__ = "0.1.0"
