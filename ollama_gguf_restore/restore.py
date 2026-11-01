"""Restore standalone ``.gguf`` files from an Ollama :class:`ModelRef`.

Ollama stores the model weights in one or more layers with media type
``application/vnd.ollama.image.model``. Each layer is a complete,
self-contained GGUF file. For single-file models there is exactly one such
layer; for sharded models Ollama stores several layers that together make up
a multi-file GGUF.

Strategy:

* Single ``model`` layer: the blob is already a valid GGUF, so we just copy
  (or hard-/symbolic-link, if the caller opts in) it to
  ``<out>/<name>-<quant>.gguf`` and verify the header.
* Multiple ``model`` layers: concatenate them in manifest order to reproduce
  the original multi-shard GGUF byte stream, then verify the header.

We never re-encode or modify tensor bytes; the output is byte-identical to
what Ollama downloaded (plus the ordering Ollama expects).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

from .gguf import parse_gguf_header, is_gguf
from .manifest import ModelRef

# Maximum bytes we will inspect for a post-restore header sanity check. We do
# not read whole multi-GB files just to verify; the header is near the start.
_HEADER_PEEK = 8 * 1024 * 1024


@dataclass
class RestoreResult:
    model: ModelRef
    output_paths: List[str]
    total_bytes: int
    verified: bool
    note: str = ""

    @property
    def output_path(self) -> str:
        return self.output_paths[0] if self.output_paths else ""

    def describe(self) -> str:
        label = self.model.reference
        head = f"{label} -> {', '.join(self.output_paths)}"
        if self.note:
            head += f" ({self.note})"
        return head


def _safe_filename_component(*parts) -> str:
    cleaned = []
    for p in parts:
        if p is None:
            continue
        token = str(p)
        # Ollama layer names can include ':' and '/'; sanitise them.
        for ch in (":", "/", "\\", "\x00"):
            token = token.replace(ch, "_")
        if token:
            cleaned.append(token)
    return "-".join(cleaned) if cleaned else "model"


def default_output_name(model: ModelRef) -> str:
    """``<name>-<quant>.gguf`` mirroring what a typical user expects."""
    quant = model.config.quantization if model.config else "unknown"
    base = _safe_filename_component(model.name)
    return f"{base}-{_safe_filename_component(quant)}.gguf"


def default_output_path(model: ModelRef, out_dir: str) -> str:
    target = out_dir
    if not os.path.isabs(target):
        target = os.path.abspath(target)
    return os.path.join(target, default_output_name(model))


def _verify_gguf(path: str) -> str:
    """Return an empty string when the header looks valid, else a reason."""
    if not os.path.exists(path):
        return f"missing: {path}"
    if not is_gguf(path):
        return "not a GGUF file (magic mismatch)"
    with open(path, "rb") as fh:
        head = fh.read(_HEADER_PEEK)
    info = parse_gguf_header(head)
    if info.error and "data section start" not in info.error:
        # Only treat structural problems (bad magic/version/count) as hard
        # errors. A truncated header peek on a huge file is acceptable.
        if info.version in (2, 3) and info.parse_complete:
            return "header complete but unexpected"
        if "bad magic" in info.error or "unsupported GGUF version" in info.error:
            return info.error
    return ""


def restore_model(
    model: ModelRef,
    out_dir: str,
    *,
    link: str = "copy",
    overwrite: bool = False,
    max_shards: int = 8,
    verbose: bool = False,
) -> RestoreResult:
    """Write a ``.gguf`` file for ``model`` into ``out_dir``.

    :param link: ``"copy"`` (default) makes real files; ``"hard"`` /
        ``"symlink"`` avoids duplicating bytes when the blob fits.
    :param max_shards: refuse multi-layer restores above this to avoid
        writing absurdly large files from a mis-parsed manifest.
    """
    model_layers = model.model_layers
    if not model_layers:
        raise FileNotFoundError(
            f"model {model.reference!r} has no "
            "'application/vnd.ollama.image.model' layer"
        )

    missing = [l for l in model_layers if not l.blob_exists]
    if missing:
        raise FileNotFoundError(
            "missing blob(s) for {}: {}".format(
                model.reference,
                ", ".join(l.filename for l in missing),
            )
        )

    if len(model_layers) > max_shards:
        raise ValueError(
            f"{model.reference}: {len(model_layers)} model layers exceed "
            f"max_shards={max_shards}"
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = default_output_path(model, out_dir)
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"output exists (use overwrite=True): {out_path}")

    total_bytes = 0

    if len(model_layers) == 1:
        layer = model_layers[0]
        _emit_layer(layer.blob_path, out_path, link)
        total_bytes = layer.size or os.path.getsize(out_path)
    else:
        # Sharded GGUF: concatenate shards in manifest order.
        if link != "copy":
            link = "copy"  # concatenation requires a copy
        with open(out_path, "wb") as out:
            for layer in model_layers:
                with open(layer.blob_path, "rb") as src:
                    while True:
                        chunk = src.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                total_bytes += layer.size or os.path.getsize(layer.blob_path)
                if verbose:
                    print(f"  + {layer.filename}")

    note = ""
    reason = _verify_gguf(out_path)
    verified = not reason
    if not verified:
        note = reason
    elif link != "copy":
        note = f"link={link}"
    if total_bytes <= 0:
        total_bytes = os.path.getsize(out_path)

    result = RestoreResult(
        model=model,
        output_paths=[out_path],
        total_bytes=total_bytes,
        verified=verified,
        note=note,
    )
    return result


def _emit_layer(src: str, dst: str, link: str) -> None:
    if link in ("hard", "symlink"):
        try:
            _link_file(src, dst, hard=(link == "hard"))
            return
        except OSError:
            # Fall back to a plain copy; we never want to lose the result
            # just because the filesystem disallowed a link.
            pass
    shutil.copyfile(src, dst)


def _link_file(src: str, dst: str, hard: bool) -> None:
    if hard:
        os.link(src, dst, strict=False)
    else:
        os.symlink(src, dst, target_is_directory=False)
