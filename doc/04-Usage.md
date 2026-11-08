> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「Usage」章节拆分而来。

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
