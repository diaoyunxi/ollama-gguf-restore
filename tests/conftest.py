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
    ctx: int = 4096,
    data_size: int = 128,
) -> bytes:
    """Build a structurally valid GGUF v3 file in the *Ollama* encoding.

    This mirrors what Ollama's ``application/vnd.ollama.image.model`` layer
    looks like on disk:

    * header is 24 bytes: magic, version (u32), n_tensors (u64), n_kv (u64);
    * the KV section starts at offset 24;
    * keys and string values are ``u64 length + exactly length bytes`` (no
      null terminator, no 8-byte padding);
    * the value type table is Ollama's shifted table:  uint8=0, int8=1,
      uint16=2, int16=3, uint32=4, int32=5, float32=6, bool=7, string=8,
      array=9, uint64=10, int64=11, float64=12.
    """
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 1)  # n_tensors
    out += struct.pack("<Q", 3)  # n_kv
    assert len(out) == 24

    def key(s: str) -> None:
        nonlocal out
        out += struct.pack("<Q", len(s))
        out += s.encode()

    def str_kv(k: str, v: str) -> None:
        nonlocal out
        key(k)
        out += struct.pack("<I", 8)  # Ollama typeString
        out += struct.pack("<Q", len(v))
        out += v.encode()

    def u32_kv(k: str, v: int) -> None:
        nonlocal out
        key(k)
        out += struct.pack("<I", 4)  # Ollama typeUint32
        out += struct.pack("<I", v & 0xFFFFFFFF)

    str_kv("general.architecture", arch)
    str_kv("general.name", name)
    u32_kv("llama.context_length", ctx)

    # ---- tensor info section ----
    key("embd")
    out += struct.pack("<I", 1)  # n_dims
    out += struct.pack("<Q", 4)  # dims[0]
    out += struct.pack("<I", 0)  # ggml_type (f32)
    out += struct.pack("<Q", 0)  # offset

    out += b"\0" * data_size
    return bytes(out)


def build_standard_gguf(
    arch: str = "llama",
    ctx: int = 4096,
    data_size: int = 128,
) -> bytes:
    """Build a GGUF v3 file in the *standard* (gguf.c) encoding.

    Standard differs from Ollama: null-terminated keys, string values are
    ``u64 length + bytes`` **8-byte aligned**, and the value type table is
    f32=0, f64=1, i8=2, i16=3, i32=4, i64=5, bool=6, string=7, array=8.
    """
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 1)  # n_tensors
    out += struct.pack("<Q", 2)  # n_kv
    assert len(out) == 24
    kv_start = 24

    def align() -> None:
        nonlocal out
        delta = len(out) - kv_start
        out += b"\0" * ((-delta) % 8)

    def str_kv(k: str, v: str) -> None:
        nonlocal out
        out += k.encode() + b"\0"
        out += struct.pack("<i", 7)  # standard TYP_STR
        out += struct.pack("<Q", len(v))
        out += v.encode()
        align()

    def u32_kv(k: str, v: int) -> None:
        nonlocal out
        out += k.encode() + b"\0"
        out += struct.pack("<i", 4)  # standard TYP_I32
        out += struct.pack("<i", v)

    str_kv("general.architecture", arch)
    u32_kv("llama.context_length", ctx)

    # ---- tensor info section ----
    out += b"embd\0"
    out += struct.pack("<I", 1)
    out += struct.pack("<Q", 4)
    out += struct.pack("<I", 0)
    out += struct.pack("<Q", 0)

    out += b"\0" * data_size
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



