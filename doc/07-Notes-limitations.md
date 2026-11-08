> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「Notes & limitations」章节拆分而来。

## Notes & limitations

* Ollama's model blobs are already valid GGUFs — we do **not** re-quantise,
  re-pack or re-encode tensor data. We only re-assemble shards and verify
  the header, so output is byte-identical to what Ollama downloaded.
* Sharded models: Ollama's manifest lists each shard as a separate
  `application/vnd.ollama.image.model` layer. We concatenate them in
  manifest order, which is the order the GGUF writer expects.
* If a blob is missing on disk (incomplete download) we raise
  `FileNotFoundError` rather than produce a truncated file.
