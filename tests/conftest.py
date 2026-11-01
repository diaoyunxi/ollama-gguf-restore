"""Shared fixtures: build a minimal *valid* GGUF and a synthetic Ollama tree."""

from __future__ import annotations

import json
import os
import struct
import hashlib
import tempfile
import pytest

from ollama_gguf_restore.gguf import parse_gguf_header, is_gguf


def build_minimal_gguf(
    arch: str = "llama",
    name: str = "mini",
    quant: str = "q8_0",
    ctx: int = 4096,
    data_size: int = 128,
) -> bytes:
    """Produce a structurally valid GGUF v3 file with a single tensor.

    This is what Ollama's ``application/vnd.ollama.image.model`` layer looks
    like, distilled to the minimum a header parser accepts.
    """
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 1)  # n_tensors
    out += struct.pack("<Q", 3)  # n_kv
    out += struct.pack("<Q", 0)  # placeholder metadata_offset

    kv_start = len(out)  # 32
    out[24:32] = struct.pack("<Q", kv_start)  # patch metadata_offset

    def align() -> None:
        nonlocal out
        delta = len(out) - kv_start
        out += b"\0" * ((-delta) % 8)

    def add_str_kv(key: str, value: str) -> None:
        nonlocal out
        out += key.encode() + b"\0"
        out += struct.pack("<i", 7)  # TYP_STR
        out += struct.pack("<Q", len(value))
        out += value.encode()
        align()

    def add_i64_kv(key: str, value: int) -> None:
        # GGUF pads only *string* values to 8 bytes; primitives are packed.
        nonlocal out
        out += key.encode() + b"\0"
        out += struct.pack("<i", 5)  # TYP_I64
        out += struct.pack("<q", value)

    add_str_kv("general.architecture", arch)
    add_str_kv("general.name", name)
    add_i64_kv("llama.context_length", ctx)

    # ---- tensor info section (starts right after the last kv value) ----
    out += b"embd\0"
    out += struct.pack("<I", 1)  # n_dims
    out += struct.pack("<Q", 4)  # dims[0]
    out += struct.pack("<I", 0)  # ggml_type = f32
    out += struct.pack("<Q", 0)  # offset

    data_start = len(out)
    out += b"\0" * data_size
    assert data_start <= len(out)
    return bytes(out)


def _sha(name_blob: bytes) -> str:
    """Return a digest string we use to name the blob file in tests."""
    return "sha256:" + hashlib.sha256(name_blob).hexdigest()[:16]


@pytest.fixture
def gguf_bytes() -> bytes:
    return build_minimal_gguf()


@pytest.fixture
def ollama_tree(tmp_path, gguf_bytes):
    """Create a synthetic ~/.ollama/models style tree under tmp_path.

    Returns a dict with keys used by the tests.
    """
    models_dir = tmp_path / "models"
    blobs_dir = models_dir / "blobs"
    blobs_dir.mkdir(parents=True)

    # 1) model (GGUF) blob
    gguf_blob = blobs_dir / "sha256-ggufmini"
    gguf_blob.write_bytes(gguf_bytes)
    gguf_digest = "sha256:ggufmini"

    # 2) config blob (JSON)
    cfg = {
        "architecture": "llama",
        "model_type": "llama",
        "file_type": "q8_0",
        "parameters": "2",
    }
    cfg_bytes = json.dumps(cfg).encode()
    cfg_blob = blobs_dir / "sha256-cfg"
    cfg_blob.write_bytes(cfg_bytes)
    cfg_digest = "sha256:cfg"

    # 3) license blob
    lic_blob = blobs_dir / "sha256-lic"
    lic_blob.write_bytes(b"MIT")

    # 4) manifest
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": cfg_digest, "size": len(cfg_bytes),
                   "mediaType": "application/vnd.ollama.image.model"},
        "layers": [
            {"digest": gguf_digest, "mediaType": "application/vnd.ollama.image.model",
             "size": len(gguf_bytes)},
            {"digest": "sha256:lic", "mediaType": "application/vnd.ollama.image.license",
             "size": 3},
        ],
        "model": "registry.ollama.ai/library/mini",
    }
    manifest_dir = models_dir / "manifests" / "registry.ollama.ai" / "library" / "mini"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "latest").write_text(json.dumps(manifest))

    return {
        "models_dir": str(models_dir),
        "gguf_digest": gguf_digest,
        "gguf_blob": str(gguf_blob),
        "gguf_bytes": gguf_bytes,
        "gguf_size": len(gguf_bytes),
    }


def test_gguf_parser_accepts_synthetic(ollama_tree):
    data = ollama_tree["gguf_bytes"]
    info = parse_gguf_header(data)
    assert info.ok, info.error
    assert info.version == 3
    assert info.n_tensors == 1
    assert info.n_kv == 3
    keys = [kv.key for kv in info.kvs]
    assert "general.architecture" in keys
    assert "llama.context_length" in keys
    assert len(info.tensors) == 1
    assert info.tensors[0].name == "embd"
    assert is_gguf(data)
