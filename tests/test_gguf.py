"""Unit tests for the GGUF header parser across both encoding families."""

from __future__ import annotations

from conftest import build_minimal_gguf, build_standard_gguf
from ollama_gguf_restore.gguf import parse_gguf_header, is_gguf


def test_parser_accepts_ollama_synthetic():
    data = build_minimal_gguf()
    info = parse_gguf_header(data)
    assert info.ok, info.error
    assert info.version == 3
    assert info.n_tensors == 1
    assert info.n_kv == 3
    assert info.encoding == "ollama"
    assert info.kv_start == 24
    keys = [kv.key for kv in info.kvs]
    assert keys == ["general.architecture", "general.name", "llama.context_length"]
    assert len(info.tensors) == 1
    assert info.tensors[0].name == "embd"
    assert info.tensors[0].dims == [4]
    assert is_gguf(data)


def test_parser_detects_standard_family():
    data = build_standard_gguf()
    info = parse_gguf_header(data)
    assert info.ok, info.error
    assert info.encoding == "standard"
    assert info.kv_start == 24
    keys = [kv.key for kv in info.kvs]
    assert "general.architecture" in keys
    assert "llama.context_length" in keys
    assert info.tensors[0].name == "embd"
    assert is_gguf(data)


def test_parser_rejects_non_gguf():
    info = parse_gguf_header(b"NOTAGGUF...")
    assert not info.ok
    assert "bad magic" in info.error


def test_parser_handles_truncated_sniff():
    data = build_minimal_gguf()
    # give only the header prefix; the parser should report incomplete, not crash
    info = parse_gguf_header(data[:32])
    assert info.version == 3
    assert info.parse_complete is False
