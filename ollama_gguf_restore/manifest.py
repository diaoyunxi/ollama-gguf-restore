"""Discovery of Ollama models on disk.

Ollama's on-disk layout (an OCI-style content store):

.. code-block:: text

    <models>/
      manifests/
        <registry>/           # e.g. registry.ollama.ai, ghcr.io, ...
          <namespace>/        # e.g. library, or an org/user
            <model>/<tag>     # a JSON manifest file
      blobs/
        sha256-<hex>         # every referenced layer, keyed by digest

A manifest looks like::

    {
      "schemaVersion": 2,
      "config": {"digest": "sha256:...", "size": 123, "mediaType": "..."},
      "layers": [
        {"mediaType": "application/vnd.ollama.image.model",
         "digest": "sha256:...", "size": ...},
        {"mediaType": "application/vnd.ollama.image.license", ...},
        ...
      ],
      "model": "registry.ollama.ai/library/llama3"   # optional, newer Ollama
    }

The ``config`` blob is a small JSON describing the model (architecture,
``file_type`` quantisation, parameters). Each ``model`` layer is a complete
GGUF file (or one shard of a multi-file GGUF).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
CONFIG_MEDIA_TYPES = (
    "application/vnd.ollama.image.model",
    "application/vnd.docker.container.image.v1+json",
)

DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), ".ollama", "models"
)


def default_models_dir() -> str:
    """Resolve the Ollama models directory, honouring ``OLLAMA_MODELS``."""
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    return DEFAULT_MODELS_DIR


def blob_name(digest: str) -> str:
    """Map ``sha256:<hex>`` to the on-disk blob filename ``sha256-<hex>``."""
    alg, _, hexpart = digest.partition(":")
    return f"{alg}-{hexpart}"


@dataclass
class Layer:
    media_type: str
    digest: str
    size: int
    blob_path: Optional[str] = None
    blob_exists: bool = False

    @property
    def filename(self) -> str:
        return blob_name(self.digest)

    @property
    def is_model(self) -> bool:
        return self.media_type == MODEL_MEDIA_TYPE


@dataclass
class ModelConfig:
    """Parsed contents of the Ollama config blob (JSON)."""

    raw: dict = field(default_factory=dict)
    digest: Optional[str] = None
    path: Optional[str] = None

    @property
    def architecture(self) -> str:
        return str(self.raw.get("architecture", "unknown"))

    @property
    def model_type(self) -> str:
        return str(self.raw.get("model_type", self.raw.get("model", "unknown")))

    @property
    def quantization(self) -> str:
        return str(self.raw.get("file_type", "unknown"))


@dataclass
class ModelRef:
    """A discovered model: its manifest plus the layers that back it."""

    manifest_path: str
    registry: str
    namespace: str
    name: str
    tag: str
    model_field: Optional[str] = None
    layers: List[Layer] = field(default_factory=list)
    config: Optional[ModelConfig] = None
    size_bytes: int = 0

    @property
    def reference(self) -> str:
        """``name:tag`` reference, used for filenames."""
        return f"{self.name}:{self.tag}"

    @property
    def qualified_name(self) -> str:
        if self.model_field:
            return self.model_field
        return f"{self.registry}/{self.namespace}/{self.name}"

    @property
    def model_layers(self) -> List[Layer]:
        return [l for l in self.layers if l.is_model]

    @property
    def model_size(self) -> int:
        return sum(l.size for l in self.model_layers)


def _read_config(blobs_dir: str, manifest: dict) -> Optional[ModelConfig]:
    cfg = manifest.get("config") or {}
    digest = cfg.get("digest")
    if not digest:
        return None
    path = os.path.join(blobs_dir, blob_name(digest))
    config = ModelConfig(digest=digest, path=path)
    if not os.path.exists(path):
        return config
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config.raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # Some configs may be binary or absent; leave raw empty.
        pass
    return config


def _model_name_from_manifest(manifest_path: str, manifest: dict) -> tuple:
    """Return (name, tag, model_field) for a manifest file."""
    tag = os.path.basename(manifest_path)
    parent = os.path.dirname(manifest_path)
    name = os.path.basename(parent)
    model_field = manifest.get("model")
    # Ollama's "model" field is the fully-qualified registry reference; the
    # last path component is the local name we care about.
    if model_field:
        canonical = model_field.rsplit("/", 1)[-1]
        if canonical:
            name = canonical
    return name, tag, model_field


def list_models(models_dir: str) -> List[ModelRef]:
    """Enumerate every model manifest under ``models_dir``.

    Missing directories simply yield an empty list; malformed manifest JSON
    is skipped (reported on stderr elsewhere) rather than fatal.
    """
    manifests_dir = os.path.join(models_dir, "manifests")
    blobs_dir = os.path.join(models_dir, "blobs")
    found: List[ModelRef] = []
    if not os.path.isdir(manifests_dir):
        return found

    for dirpath, _dirs, filenames in os.walk(manifests_dir):
        for fname in filenames:
            manifest_path = os.path.join(dirpath, fname)
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue

            # registry / namespace are the two path components between
            # manifests/ and the model directory.
            rel = os.path.relpath(manifest_path, manifests_dir)
            parts = rel.split(os.sep)
            registry = parts[0] if len(parts) >= 2 else ""
            namespace = parts[1] if len(parts) >= 3 else ""

            name, tag, model_field = _model_name_from_manifest(manifest_path, manifest)

            layers: List[Layer] = []
            for layer_info in manifest.get("layers", []) or []:
                layer = Layer(
                    media_type=layer_info.get("mediaType", ""),
                    digest=layer_info.get("digest", ""),
                    size=int(layer_info.get("size", 0)),
                )
                layer.blob_path = os.path.join(blobs_dir, layer.filename)
                layer.blob_exists = os.path.exists(layer.blob_path)
                layers.append(layer)

            ref = ModelRef(
                manifest_path=manifest_path,
                registry=registry,
                namespace=namespace,
                name=name,
                tag=tag,
                model_field=model_field,
                layers=layers,
            )
            ref.config = _read_config(blobs_dir, manifest)
            ref.size_bytes = ref.model_size
            found.append(ref)
    return found


def find_model(models_dir: str, reference: str) -> Optional[ModelRef]:
    """Resolve a ``name:tag`` or bare ``name`` reference to a :class:`ModelRef`."""
    reference = reference.strip()
    want_tag = ""
    if ":" in reference:
        want_name, want_tag = reference.rsplit(":", 1)
        reference = want_name
    else:
        want_name = reference
        want_tag = ""  # any tag

    for ref in list_models(models_dir):
        if ref.name != want_name:
            continue
        if want_tag and ref.tag != want_tag:
            continue
        return ref
    return None
