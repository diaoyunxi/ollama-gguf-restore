"""End-to-end tests: discover + restore from a synthetic Ollama tree."""

from __future__ import annotations

import os

from ollama_gguf_restore.discovery import discover, resolve
from ollama_gguf_restore.manifest import ModelRef
from ollama_gguf_restore.restore import restore_model
from ollama_gguf_restore.gguf import parse_gguf_header


def test_discover_finds_single_model(ollama_tree):
    models, base = discover(ollama_tree["models_dir"])
    assert len(models) == 1
    ref = models[0]
    assert ref.reference == "mini:latest"
    assert ref.qualified_name == "registry.ollama.ai/library/mini"
    assert ref.name == "mini"
    assert ref.tag == "latest"
    assert ref.config is not None
    assert ref.config.architecture == "llama"
    assert ref.config.quantization == "q8_0"
    # exactly one model layer
    assert len(ref.model_layers) == 1
    assert ref.model_layers[0].blob_exists is True
    assert ref.model_size == ollama_tree["gguf_size"]


def test_resolve_by_name(ollama_tree):
    ref, base = resolve("mini", ollama_tree["models_dir"])
    assert ref.reference == "mini:latest"

    # bare name resolves even with a different tag form
    ref2, _ = resolve("mini:latest", ollama_tree["models_dir"])
    assert ref2 is not None


def test_restore_single_layer(ollama_tree, tmp_path):
    ref, _ = resolve("mini", ollama_tree["models_dir"])
    out_dir = str(tmp_path / "out")
    result = restore_model(ref, out_dir, link="copy")

    assert result.verified, result.note
    assert os.path.exists(result.output_path)
    # Output is byte-identical to the source blob.
    with open(result.output_path, "rb") as fh:
        restored = fh.read()
    assert restored == ollama_tree["gguf_bytes"]
    # And it still parses as GGUF.
    assert parse_gguf_header(restored).ok

    # default filename includes the quantisation
    assert result.output_path.endswith("mini-q8_0.gguf")


def test_restore_refuses_missing_blob(ollama_tree, tmp_path):
    ref, _ = resolve("mini", ollama_tree["models_dir"])
    # Remove the blob to simulate an incomplete download.
    for layer in ref.model_layers:
        if os.path.exists(layer.blob_path):
            os.remove(layer.blob_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        restore_model(ref, str(tmp_path))


def test_default_output_name(ollama_tree):
    ref, _ = resolve("mini", ollama_tree["models_dir"])
    from ollama_gguf_restore.restore import default_output_name

    assert default_output_name(ref) == "mini-q8_0.gguf"


def test_default_output_path_is_abs(ollama_tree):
    ref, _ = resolve("mini", ollama_tree["models_dir"])
    from ollama_gguf_restore.restore import default_output_path

    p = default_output_path(ref, "out")
    assert p.endswith(os.path.join("out", "mini-q8_0.gguf"))
    assert os.path.isabs(p)
