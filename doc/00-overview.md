# 概览

**项目**：ollama-gguf-restore

Recover standalone GGUF files from Ollama's on-disk model blobs.

## 原文前言

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
