> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「Where Ollama puts your models」章节拆分而来。

## Where Ollama puts your models

| OS       | Default models dir            |
|----------|-------------------------------|
| Linux    | `~/.ollama/models`            |
| macOS    | `~/.ollama/models`            |
| Windows  | `C:\Users\<you>\.ollama\models` |

Override the location via the `OLLAMA_MODELS` env var, or pass
`-d/--models-dir` on every command.
