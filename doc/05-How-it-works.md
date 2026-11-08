> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「How it works」章节拆分而来。

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
