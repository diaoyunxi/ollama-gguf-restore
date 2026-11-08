> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「Features」章节拆分而来。

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
