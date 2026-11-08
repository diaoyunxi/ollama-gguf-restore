> [← 返回文档目录](../README.md)

# 命令行参数（`ollama_gguf_restore/cli.py`）

子命令：`list`、`inspect`、`restore`、`restore-all`。

| 参数 | 说明 |
|---|---|
| `--version` | 打印版本并退出（全局） |
| `-d` / `--models-dir` | 覆盖 Ollama models 目录（也可用 `OLLAMA_MODELS` 环境变量） |
| `-o` / `--out` | 输出目录，默认 `./out` |
| `--link` | 材料化方式：`copy` / `hard` / `symlink` |
| `--overwrite` | 覆盖已存在的输出文件 |
| `--max-shards` | 分片数上限，默认 `8` |
| `-v` / `--verbose` | 打印每个分片 |
| `--json` | `inspect` 时输出 manifest JSON |
| `--no-gguf` | `inspect` 时跳过 GGUF 头摘要 |
| `--help` / `-h` | 显示帮助 |

> 以上参数均来自 `cli.py::build_parser()` 与 `add_common()`。
