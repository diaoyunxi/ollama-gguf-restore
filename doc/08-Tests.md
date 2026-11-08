> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「Tests」章节拆分而来。

## Tests

```bash
python -m pytest tests
```

The suite builds a synthetic Ollama tree (valid GGUF blob + config + manifest)
in `tmp_path`, so it runs on any OS and never needs a real Ollama install.
