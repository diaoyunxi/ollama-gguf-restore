"""Command-line interface.

Commands
--------

* ``list``       enumerate discovered Ollama models.
* ``inspect``    show the layers, config and GGUF header of one model.
* ``restore``    write a clean ``.gguf`` file out of the model's blobs.
* ``restore-all`` restore every discovered model.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .discovery import discover, resolve
from .gguf import parse_gguf_header
from .manifest import ModelRef
from .restore import restore_model, default_output_path, _verify_gguf

_GGUF_PEEK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _human_size(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while num >= 1024 and i < len(units) - 1:
        num /= 1024
        i += 1
    if i == 0:
        return f"{int(num)} {units[i]}"
    return f"{num:.1f} {units[i]}"


def _print_model_summary(ref: ModelRef) -> None:
    cfg = ref.config
    quant = cfg.quantization if cfg else "?"
    arch = cfg.architecture if cfg else "?"
    n_model_layers = len(ref.model_layers)
    print(f"{ref.reference:32} quant={quant:8} arch={arch:12} "
          f"model-layers={n_model_layers} size={_human_size(ref.model_size)}")
    missing = [l for l in ref.model_layers if not l.blob_exists]
    if missing:
        print(f"  ! {len(missing)} model blob(s) missing on disk")


def _dump_gguf_info(ref: ModelRef) -> None:
    layers = ref.model_layers
    if not layers:
        print("  (no model layer found)")
        return
    # Inspect the first model layer's header.
    target = layers[0].blob_path
    if not target:
        print("  (no blob path resolved)")
        return
    with open(target, "rb") as fh:
        head = fh.read(_GGUF_PEEK)
    info = parse_gguf_header(head)
    print(f"  blob   : {layers[0].filename}")
    print(info.summary())
    # Dump a few interesting keys.
    interesting = {
        "general.architecture",
        "general.name",
        "general.size_label",
        "general.quantized_by_according_to",
        "llama.context_length",
    }
    seen = 0
    for kv in info.kvs:
        if kv.key in interesting:
            seen += 1
            print(f"    {kv.key} (vtype={kv.vtype})")
        if seen > 8:
            break


# ---------------------------------------------------------------------------
# command handlers
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    models, base = discover(args.models_dir)
    print(f"Ollama models dir: {base}\n")
    if not models:
        print("No models found.")
        return 1
    for ref in models:
        _print_model_summary(ref)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    ref, base = resolve(args.model, args.models_dir)
    print(f"Model  : {ref.reference}")
    print(f"  dir  : {base}")
    print(f"  qual : {ref.qualified_name}")
    print(f"  manifest: {ref.manifest_path}")
    print(f"  layers ({len(ref.layers)}):")
    for layer in ref.layers:
        mark = "ok" if layer.blob_exists else "MISSING"
        print(f"    - {layer.media_type or '(?mediaType)'} "
              f"[{layer.size:>12} b] {layer.filename} ({mark})")
    print(f"  model bytes: {_human_size(ref.model_size)}")
    if args.json:
        payload = {
            "reference": ref.reference,
            "qualified": ref.qualified_name,
            "manifest": ref.manifest_path,
            "layers": [
                {
                    "mediaType": l.media_type,
                    "digest": l.digest,
                    "size": l.size,
                    "blob_exists": l.blob_exists,
                }
                for l in ref.layers
            ],
            "config": ref.config.raw if ref.config else None,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.no_gguf:
        _dump_gguf_info(ref)
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    ref, base = resolve(args.model, args.models_dir)
    out = args.out or "out"
    result = restore_model(
        ref,
        out,
        link=args.link,
        overwrite=args.overwrite,
        max_shards=args.max_shards,
        verbose=args.verbose,
    )
    status = "OK" if result.verified else "WARN"
    print(f"[{status}] {result.describe()}")
    print(f"  -> {result.output_path}  ({_human_size(result.total_bytes)})")
    if result.note:
        print(f"  note: {result.note}")
    return 0 if result.verified else 2


def cmd_restore_all(args: argparse.Namespace) -> int:
    models, base = discover(args.models_dir)
    if not models:
        print("No models found; nothing to restore.")
        return 1
    out = args.out or "out"
    ok = 0
    warn = 0
    fail = 0
    for ref in models:
        try:
            result = restore_model(
                ref,
                out,
                link=args.link,
                overwrite=args.overwrite,
                max_shards=args.max_shards,
                verbose=False,
            )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            print(f"[FAIL] {ref.reference}: {exc}")
            fail += 1
            continue
        status = "OK" if result.verified else "WARN"
        print(f"[{status}] {result.describe()}")
        if result.verified:
            ok += 1
        else:
            warn += 1
    print(f"\nrestored {ok} ok, {warn} with warnings, {fail} failed "
          f"(of {len(models)}) into {out}")
    return 0 if fail == 0 else 2


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-gguf-restore",
        description=(
            "Recover standalone GGUF files from Ollama's on-disk model blobs."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # common options used by list / inspect / restore / restore-all
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-d", "--models-dir",
            default=None,
            help="Ollama models dir (default: $OLLAMA_MODELS or ~/.ollama/models)",
        )

    p_list = sub.add_parser("list", help="list discovered models")
    add_common(p_list)
    p_list.set_defaults(func=cmd_list)

    p_insp = sub.add_parser("inspect", help="inspect one model")
    p_insp.add_argument("model", help="name:tag or bare name")
    add_common(p_insp)
    p_insp.add_argument("--json", action="store_true", help="dump manifest JSON")
    p_insp.add_argument(
        "--no-gguf", action="store_true", help="skip GGUF header summary"
    )
    p_insp.set_defaults(func=cmd_inspect)

    p_res = sub.add_parser("restore", help="restore one model to .gguf")
    p_res.add_argument("model", help="name:tag or bare name")
    add_common(p_res)
    p_res.add_argument(
        "-o", "--out", default=None,
        help="output directory for the .gguf file (default: ./out)",
    )
    p_res.add_argument(
        "--link", choices=("copy", "hard", "symlink"), default="copy",
        help="how to materialise the file (default: copy)",
    )
    p_res.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing output file",
    )
    p_res.add_argument(
        "--max-shards", type=int, default=8,
        help="refuse more than this many model layers (default: 8)",
    )
    p_res.add_argument(
        "-v", "--verbose", action="store_true",
        help="print each concatenated shard",
    )
    p_res.set_defaults(func=cmd_restore)

    p_all = sub.add_parser(
        "restore-all", help="restore every discovered model"
    )
    add_common(p_all)
    p_all.add_argument(
        "-o", "--out", default=None,
        help="output directory for .gguf files (default: ./out)",
    )
    p_all.add_argument(
        "--link", choices=("copy", "hard", "symlink"), default="copy",
    )
    p_all.add_argument("--overwrite", action="store_true")
    p_all.add_argument("--max-shards", type=int, default=8)
    p_all.set_defaults(func=cmd_restore_all)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
