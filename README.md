# ollama-gguf-restore

Recover standalone **GGUF** files from Ollama's on-disk model blobs.

Ollama does not save a `model.gguf` you can point at `llama.cpp` /
`llama-server` / `llama-factory` / vLLM-ollama. It stores every pulled model
as content-addressed pieces under `~/.ollama/models`:

```
~/.ollama/models/
  manifests/registry.ollama.ai/library/<model>/<tag>   # JSON manifest
  blobs/sha256-<hex>                                   # layer bytes
```

The layer with media type
`application/vnd.ollama.image.model` **is a complete GGUF file** (or one shard
of a sharded GGUF). This tool reads the manifest, locates those blobs, verifies
the GGUF header and writes a clean `.gguf` you can use anywhere.

> No network, no Ollama server, no re-download — just the files Ollama
> already put on disk.

## Features

* **Cross-platform** — Linux, macOS, Windows (native paths, no deps).
* **Stdlib only** — Python 3.8+, no external packages required.
* **Verifies output** — every restored file is checked against a real GGUF
  header parser (magic, version, kv, tensor counts).
* **Single- and multi-shard** — one model layer → one file; several →
  concatenated in manifest order (reproduces Ollama's sharded GGUF).
* **Safe** — refuses to write over an existing file without `--overwrite`;
  reports missing blobs instead of corrupting.
* **CLI** — `list`, `inspect`, `restore`, `restore-all`.
* **Library** — importable from Python for scripting.

## Install

No install is required; the package is importable from the repo checkout:

```bash
git clone https://github.com/diaoyunxi/ollama-gguf-restore
cd ollama-gguf-restore
python -m ollama_gguf_restore --help
```

Or install as a tool (creates an `ollama-gguf-restore` entry point):

```bash
pip install -e .
ollama-gguf-restore --help
```

## Where Ollama puts your models

| OS       | Default models dir            |
|----------|-------------------------------|
| Linux    | `~/.ollama/models`            |
| macOS    | `~/.ollama/models`            |
| Windows  | `C:\Users\<you>\.ollama\models` |

Override the location via the `OLLAMA_MODELS` env var, or pass
`-d/--models-dir` on every command.

## Usage

```bash
# list every model Ollama has downloaded
python -m ollama_gguf_restore list

# inspect one model: layers, config, GGUF header summary
python -m ollama_gguf_restore inspect llama3 --json

# restore a single model to a clean .gguf
python -m ollama_gguf_restore restore llama3 -o ./restored

# restore every model
python -m ollama_gguf_restore restore-all -o ./restored
```

### Restore options

| flag            | default | meaning                                            |
|-----------------|---------|----------------------------------------------------|
| `-o / --out`    | `./out` | output directory for the `.gguf` files             |
| `--link`        | `copy`  | `copy` / `hard` / `symlink` — how the bytes are materialised |
| `--overwrite`   | off     | replace an existing output file                     |
| `--max-shards`  | `8`     | refuse manifests with more model layers than this   |
| `-v/--verbose`  | off     | print each concatenated shard                       |

Output filename: `<name>-<quant>.gguf` (e.g. `llama3-q8_0.gguf`).

## How it works

1. **Discover** — walk `manifests/` and read each manifest. The model name is
   derived from the manifest path (…/library/<name>/<tag>) or the `model`
   field if present.
2. **Read config** — the manifest's `config` blob (a small JSON) is loaded to
   get `architecture`, `model_type`, and `file_type` (the quantisation).
3. **Locate model layers** — every layer whose `mediaType` is
   `application/vnd.ollama.image.model` maps to `blobs/sha256-<hex>`.
4. **Materialise** —
   * one model layer: `copyfile` / `link` / `symlink` it out;
   * several: open each and `write()` to a single file in manifest order.
5. **Verify** — run the in-tree GGUF header parser over the result. A
   structurally valid header (correct magic, version, tensor + kv counts,
   data section within bounds) marks the output `verified`.

## Python API

```python
from ollama_gguf_restore import discover, restore_model, parse_gguf_header

models, base = discover()                       # all Ollama models
ref = models[0]
print(ref.reference, ref.config.quantization, ref.model_size)

result = restore_model(ref, "./restored")
print(result.output_path, result.verified)
```

`discover()` returns every :class:`ModelRef` it can find; `restore_model()`
writes one `.gguf` and returns a :class:`RestoreResult` describing the output
path, total bytes and whether the header verified.

## Notes & limitations

* Ollama's model blobs are already valid GGUFs — we do **not** re-quantise,
  re-pack or re-encode tensor data. We only re-assemble shards and verify
  the header, so output is byte-identical to what Ollama downloaded.
* Sharded models: Ollama's manifest lists each shard as a separate
  `application/vnd.ollama.image.model` layer. We concatenate them in
  manifest order, which is the order the GGUF writer expects.
* If a blob is missing on disk (incomplete download) we raise
  `FileNotFoundError` rather than produce a truncated file.

## Tests

```bash
python -m pytest tests
```

The suite builds a synthetic Ollama tree (valid GGUF blob + config + manifest)
in `tmp_path`, so it runs on any OS and never needs a real Ollama install.

## License

MIT. See [LICENSE](LICENSE.md).
