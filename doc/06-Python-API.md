> [← 返回文档目录](../README.md) ｜ 本分册由 `README.md` 的「Python API」章节拆分而来。

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
